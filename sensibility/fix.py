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

"""
Implements the logic to attempt to fix syntax errors.
"""

from typing import Iterable, Iterator, List, NamedTuple, Sequence, SupportsFloat
from typing import cast

import numpy as np
from numpy.linalg import norm  # noqa

from sensibility import (
    Edit, Insertion, Deletion, Substitution,
    Token,
    Vind,
    language,
)
from sensibility.model.lstm import DualLSTMModel
from sensibility.source_vector import SourceVector, to_source_vector
from sensibility.vocabulary import NoSourceRepresentationError


class LSTMFixerUpper:
    """
    Suggests fixes for syntax errors in a file with a dual LSTM model (which
    you must provide).

    TODO: Make an abc, probably.
    """

    def __init__(self, model: DualLSTMModel, k: int=4) -> None:
        self.model = model
        self.k = k

    def fix(self, file: bytes) -> Sequence[Edit]:
        """
        Produces a ranked sequence of possible edits that will fix the file.
        If there are no possible fixes, the sequence will be empty.
        """
        # Get file vector for the error'd file.
        file_vector = to_source_vector(file, oov_to_unk=True)
        tokens = tuple(language.tokenize(file))
        predictions = self.model.predict_file(file_vector)

        # Holds the lowest agreement at each point in the file.
        results: List[IndexResult] = []

        # These will hold the TOP predictions at a given index.
        forwards_predictions: List[Vind] = []
        backwards_predictions: List[Vind] = []

        for index, pred in enumerate(predictions):
            vind = file_vector[index]
            token = tokens[index]
            prefix_pred = pred.forwards
            suffix_pred = pred.backwards

            # Figure out the agreement between models, and against the ground
            # truth.
            result = IndexResult(index, file_vector, prefix_pred, suffix_pred, token)
            results.append(result)

            # Store the TOP prediction from each model.
            # TODO: document corner cases!
            top_next_prediction = prefix_pred.argmax()
            top_prev_prediction = suffix_pred.argmax()

            assert 0 <= top_next_prediction <= len(language.vocabulary)
            assert 0 <= top_prev_prediction <= len(language.vocabulary)
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
        for disagreement in ranked_results[:self.k]:
            pos = disagreement.index

            likely_next: Vind = forwards_predictions[pos]
            likely_prev: Vind = backwards_predictions[pos]

            # Note: the order of these operations SHOULDN'T matter,
            # but typically we only report the first fix that works.
            # Because missing tokens are the most common
            # we'll try to insert tokens first, THEN delete.

            # Assume a deletion. Let's try inserting some tokens.
            fixes.try_insert(pos, likely_next)
            fixes.try_insert(pos, likely_prev)

            # Assume an insertion. Let's try removing the offensive token.
            fixes.try_delete(pos)

            # Assume a substitution. Let's try swapping the token.
            fixes.try_substitute(pos, likely_next)
            fixes.try_substitute(pos, likely_prev)

        # TODO: sort by how fixed the result is after applying the fix.
        return tuple(fixes)


class IndexResult(SupportsFloat):
    """
    Provides results for EACH INDIVIDUAL INDEX in a file.
    """
    __slots__ = (
        'index',
        'cosine_similarity', 'indexed_prob',
        'mutual_info', 'total_variation', 'token'
    )

    def __init__(self, index: int, program: SourceVector,
                 a: np.ndarray, b: np.ndarray, token: Token) -> None:
        assert 0 <= index < len(program)
        self.index = index
        self.token = token

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
        # Clamp between 0.0 an 1.0 because due to the weirdness of floating
        # point, norm can return slightly above 1.0...
        from sensibility.utils import clamp
        self.total_variation = clamp(.5 * norm(a - b, ord=1))

        # TODO: Store the forwards and backward predictions?
        #   => useful for visual debugging later.

    def __repr__(self) -> str:
        return (f'IndexResult(index={self.index!r}, token={self.token!r}, '
                f'score={float(self)})')

    @property
    def line_no(self) -> int:
        return self.token.line

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
        # XXX: Move this import elsewhere.
        try:
            source_code = edit.apply(self.vector).to_source_code()
        except NoSourceRepresentationError:
            import logging
            logger = logging.getLogger(type(self).__name__)
            logger.info(f"Tried applying %r", edit)
            return
        if language.check_syntax(source_code):
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
    from math import isclose
    return isclose(norm(x, p), 1.0, rel_tol=tolerance)
