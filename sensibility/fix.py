#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2017 Eddie Antonio Santos <easantos@ualberta.ca>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import tempfile
from typing import (
    Iterable, Iterator, SupportsFloat, Sequence,
    List, NamedTuple, TextIO, cast,
)

import numpy as np
from numpy.linalg import norm

from .corpus import Corpus
from .edit import Edit, Insertion, Deletion, Substitution
from .predictions import Predictions, Contexts
from .source_file import SourceFile
from .source_vector import SourceVector
from .tokenize_js import tokenize_file, check_syntax_file
from .vectorize_tokens import serialize_tokens
from .vectors import Vectors
from .vocabulary import vocabulary, Vind


class IndexResult(SupportsFloat):
    """
    Provides results for EACH INDIVIDUAL INDEX in a file.
    """
    __slots__ = (
        'index',
        'cosine_similarity', 'indexed_prob',
        'mutual_info', 'total_variation'
    )

    def __init__(self, index: int, program: SourceVector,
                 a: np.ndarray, b: np.ndarray) -> None:
        assert 0 <= index < len(program)
        self.index = index

        # Categorical distributions MUST have |x|_1 == 1.0
        assert is_normalized_vector(a, p=1) and is_normalized_vector(b, p=1)

        # P(token | prefix AND token | suffix)
        # 1.0 == both models completely agree the token should be here.
        # .25 == lukewarm---models kind of think this token should be here
        # 0.0 == at least one model finds this token absolutely unlikely
        self.indexed_prob = a[program[index]] * b[program[index]]

        # How similar are the two categorical distributions?
        # 1.0 == Exactly similar -- pointing in the same direction
        # 0.0 == Not similar --- pointing in orthogonal direction
        self.cosine_similarity = (a @ b) / (norm(a) * norm(b))

        # Use averaged KL-divergence?
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.4480&rep=rep1&type=pdf
        # https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/metrics/cluster/supervised.py#L531

        # Total variation distance:
        # http://onlinelibrary.wiley.com/doi/10.1111/j.1751-5823.2002.tb00178.x/epdf
        # https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
        self.total_variation = .5 * norm(a - b, ord=1)

        # TODO: Store the forwards and backward predictions?
        #   => useful for visual debugging later.

    @property
    def comp_total_variation(self):
        """
        The complement of the total variation.
        Flips the semantics of total variation such that:

         * 1.0 means the distribtions are similar.
         * 0.0 means the distribtions are different.

        As such, its meanings are the same as cosine similarity and
        indexed_probability.
        """
        assert 0.0 <= self.total_variation <= 1.0
        return 1.0 - self.total_variation

    def __float__(self) -> float:
        """
        Returns the score between the two elements.

        0.0 means syntax error;
        1.0 means valid, natural code.

        Smaller value => more likely to be a syntax error.
        """
        # We can tweak λ to weigh local and global factors differently,
        # but for now, weigh them equally.
        # TODO: use sklearn's Lasso regression to find this coefficient?
        # TODO:                 `-> fit_intercept=False
        λ = .5
        score = λ * self.indexed_prob + (1 - λ) * self.comp_total_variation
        assert 0 <= score <= 1
        return score


class FixResult(NamedTuple):
    # The results of detecting and fixing syntax errors.
    ranks: Sequence[IndexResult]
    fixes: Sequence[Edit]


class Sensibility:
    """
    Detects and fixes syntax errors in JavaScript files.
    """
    context_length = 20

    def __init__(self, fold: int) -> None:
        # TODO: this fold business is weird. Get rid of it.
        self.predictions = Predictions(fold)

    def rank_and_fix(self, filename: str, k: int=4) -> FixResult:
        """
        Rank the syntax error location (in token number) and returns a possible
        fix for the given filename.
        """

        # Get file vector for the incorrect file.
        with open(filename, 'rt', encoding='UTF-8') as script:
            tokens = tokenize_file(script)
        file_vector = serialize_tokens(tokens)
        assert len(file_vector) > 0

        # Holds the lowest agreement at each point in the file.
        results: List[IndexResult] = []

        # These will hold the TOP predictions at a given point.
        forwards_predictions: List[Vind] = []  # XXX: https://github.com/PyCQA/pycodestyle/pull/640
        backwards_predictions: List[Vind] = []
        contexts = enumerate(self.contexts(file_vector))

        for index, ((prefix, token), (suffix, _)) in contexts:
            assert token == file_vector[index], (
                str(token) + ' ' + str(file_vector[index])
            )

            # Fetch predictions.
            prefix_pred = np.array(self.predictions.predict_forwards(prefix))  # type: ignore
            suffix_pred = np.array(self.predictions.predict_backwards(suffix))  # type: ignore

            result = IndexResult(index, file_vector, prefix_pred, suffix_pred)
            results.append(result)

            # Store the TOP prediction from each model.
            # TODO: document corner cases!
            top_next_prediction = prefix_pred.argmax()
            top_prev_prediction = suffix_pred.argmax()
            assert 0 <= top_next_prediction <= len(vocabulary)
            assert 0 <= top_prev_prediction <= len(vocabulary)
            forwards_predictions.append(cast(Vind, top_next_prediction))
            backwards_predictions.append(cast(Vind, top_prev_prediction))
            assert top_next_prediction == forwards_predictions[index]
            assert top_prev_prediction == backwards_predictions[index]

        # Rank the results by some metric of similarity defined by IndexResult
        # (the top rank will be LEAST similar).
        ranked_results = tuple(sorted(results, key=float))

        # For the top-k disagreements, synthesize fixes.
        # NOTE: k should be determined by the MRR of finding the syntax error!
        fixes = Fixes(file_vector)
        for disagreement in ranked_results[:k]:
            pos = disagreement.index

            likely_next: Vind = forwards_predictions[pos]
            likely_prev: Vind = backwards_predictions[pos]

            # Note: the order of these operations SHOULDN'T matter,
            # but typically we only report the first fix that works.
            # Because missing tokens are usually a bigger issue,
            # we'll try to insert tokens first, THEN delete.

            # Assume a deletion. Let's try inserting some tokens.
            fixes.try_insert(pos, likely_next)
            fixes.try_insert(pos, likely_prev)

            # Assume an addition. Let's try removing the offensive token.
            fixes.try_delete(pos)

            # Assume a substitution. Let's try swapping the token.
            fixes.try_substitute(pos, likely_next)
            fixes.try_substitute(pos, likely_prev)

        return FixResult(ranks=ranked_results, fixes=tuple(fixes))

    def contexts(self, file_vector: SourceVector) -> Contexts:
        """
        Yield every context (prefix, suffix) in the given file vector.
        """
        return self.predictions.contexts(cast(Sequence[Vind], file_vector))

    @staticmethod
    def is_okay(filename: str) -> bool:
        """
        Check if the syntax is okay.
        """
        with open(filename, 'rt') as source_file:
            return check_syntax_file(source_file)


# TODO: Move this to a different file, probably
class Fixes(Iterable[Edit]):
    def __init__(self, vector: SourceVector) -> None:
        self.vector = vector
        self.fixes: List[Edit] = []

    def try_insert(self, index: int, token: Vind) -> None:
        self._try_edit(Insertion.create_mutation(self.vector, index, token))

    def try_delete(self, index: int) -> None:
        self._try_edit(Deletion.create_mutation(self.vector, index))

    def try_substitute(self, index: int, token: Vind) -> None:
        edit = Substitution.create_mutation(self.vector, index, token)
        self._try_edit(edit)

    def _try_edit(self, edit: Edit) -> None:
        """
        Actually apply the edit to the file. Add it to the fixes if it works.
        """
        with temporary_program(edit.apply(self.vector)) as mutant_file:
            if check_syntax_file(mutant_file):
                self.fixes.append(edit)

    def __bool__(self) -> bool:
        return len(self.fixes) > 0

    def __iter__(self) -> Iterator[Edit]:
        return iter(self.fixes)


def is_normalized_vector(x: np.ndarray, p: int=2, tolerance=0.01) -> bool:
    """
    Returns whether the vector is normalized.

    >>> is_normalized_vector(np.array([ 0.,  0.,  1.]))
    True
    >>> is_normalized_vector(np.array([ 0.5 ,  0.25,  0.25]))
    False
    >>> is_normalized_vector(np.array([ 0.5 ,  0.25,  0.25]), p=1)
    True
    """
    return math.isclose(norm(x, p), 1.0, rel_tol=tolerance)


# TODO: move to a more appropriate file
def temporary_program(program: SourceVector) -> TextIO:
    """
    Returns a temporary file of the given program.
    """
    raw_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='UTF-8')
    mutated_file = cast(TextIO, raw_file)
    try:
        program.print(file=mutated_file)
        mutated_file.flush()
        mutated_file.seek(0)
    except IOError as error:
        raw_file.close()
        raise error
    return mutated_file
