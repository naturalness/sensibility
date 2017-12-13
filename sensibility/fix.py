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

import logging
from typing import (Iterable, Iterator, List, NamedTuple, Sequence, Set,
                    SupportsFloat, cast)

import numpy as np
from numpy.linalg import norm  # noqa

from sensibility import (Deletion, Edit, Insertion, Substitution, Token, Vind,
                         language)
from sensibility.model.lstm import DualLSTMModel
from sensibility.source_vector import SourceVector, to_source_vector
from sensibility.vocabulary import NoSourceRepresentationError

# The smallest positive (non-zero) float32.
epsilon = np.nextafter(np.float32(0), np.float32(1))


class LSTMFixerUpper:
    """
    Suggests fixes for syntax errors in a file with a dual LSTM model (which
    you must provide).

    TODO: Make an abc, probably.
    """

    def __init__(self, model: DualLSTMModel, k: int=3) -> None:
        """
        Set k to the ceil(average perplexity of model).
        """
        self.model = model
        self.k = k

    def fix(self, source_file: bytes) -> Sequence[Edit]:
        """
        Produces a ranked sequence of possible edits that will fix the file.
        If there are no possible fixes, the sequence will be empty.
        """
        # Get file vector for the error'd file.
        file_vector = to_source_vector(source_file, oov_to_unk=True)
        tokens = tuple(language.tokenize(source_file))
        predictions = self.model.predict_file(file_vector)

        # Holds the lowest agreement at each point in the file.
        results: List[IndexResult] = []

        for index, pred in enumerate(predictions):
            vind = file_vector[index]
            token = tokens[index]
            prefix_pred = pred.forwards
            suffix_pred = pred.backwards

            # Figure out the agreement between models, and against the ground
            # truth.
            result = IndexResult(index, file_vector, prefix_pred, suffix_pred, token, vind)
            results.append(result)

        # Rank the results by some metric of similarity defined by IndexResult
        # (the top rank will be LEAST similar).
        ranked_results = tuple(sorted(results, key=float))

        # For the top-k disagreements, synthesize fixes.
        # NOTE: k should be determined by the xentropy of the models!
        fixes = Fixes(file_vector)
        for disagreement in ranked_results[:self.k]:
            pos = disagreement.index

            likely_tokens = disagreement.best_suggestions()

            # Note: the order of these operations SHOULDN'T matter,
            # but typically we only report the first fix that works.
            # Because missing tokens are the most common
            # we'll try to insert tokens first, THEN delete.

            # Assume a deletion. Let's try inserting some tokens.
            for likely_token in likely_tokens:
                fixes.try_insert(pos, likely_token)

            # Assume an insertion. Let's try removing the offensive token.
            fixes.try_delete(pos)

            # Assume a substitution. Let's try swapping the token.
            for likely_token in likely_tokens:
                fixes.try_substitute(pos, likely_token)

        return tuple(fixes)


class IndexResult(SupportsFloat):
    """
    Provides results for EACH INDIVIDUAL INDEX in a file.
    """

    def __init__(self, index: int, program: SourceVector,
                 a: np.ndarray, b: np.ndarray,
                 token: Token, vind: Vind) -> None:
        self.a = a
        self.b = b
        assert 0 <= index < len(program)
        self.index = index
        self.token = token

        # Categorical distributions MUST have |x|_1 == 1.0
        assert is_normalized_vector(a, p=1) and is_normalized_vector(b, p=1)

        # P(token | prefix AND token | suffix)
        # 1.0 == both models completely agree the token should be here.
        # .25 == lukewarm---models kind of think this token should be here
        # 0.0 == at least one model finds this token absolutely unlikely
        self.indexed_prob = float(a[program[index]] * b[program[index]])

        # How similar are the two categorical distributions?
        # 1.0 == Exactly similar -- pointing in the same direction
        # 0.0 == Not similar --- pointing in orthogonal direction
        self.cosine_similarity = float((a @ b) / (norm(a) * norm(b)))

        # Cross-entropy
        p = one_hot(vind, len(a))
        self.xentropy = float(cross_entropy(p, a) + cross_entropy(p, b))

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
        # The original paper had this metric:
        # Agreement: elementwise harmonic mean of two vectors
        return -self.xentropy

    def __str__(self) -> str:
        """
        Prints an elaborate debug display of metrics.
        """
        (f1, f1t), (f2, f2t), (f3, f3t) = self._maxes(self.a)
        (b1, b1t), (b2, b2t), (b3, b3t) = self._maxes(self.b)
        token = self.token.name

        return f"""
                         ⎧{f1t:10}, {f1:6.2f}%                      ⎧{b1t:10}, {b1:6.2f}%
        f({token:>11}) = ⎨{f2t:10}, {f2:6.2f}%     b({token:>11}) = ⎨{b2t:10}, {b2:6.2f}%
                         ⎩{f3t:10}, {f3:6.2f}%                      ⎩{b3t:10}, {b3:6.2f}%

                               xentropy = {self.xentropy:5}
                              total_var = {self.total_variation:5}
                             index_prob = {self.indexed_prob:5}
                             cosine_sim = {self.cosine_similarity:5}
        """

    def best_suggestions(self) -> Set[Vind]:
        return set(self.top_forwards) | set(self.top_backwards)

    @property
    def top_forwards(self):
        return self._top(self.a)

    @property
    def top_backwards(self):
        return self._top(self.b)

    def _top(self, vector, k=3) -> np.ndarray:
        return vector.argpartition(-k)[-k:][::-1]

    def _maxes(self, vector, k=3):
        """
        Yields percentage, and token text of top-k entries.
        """
        from sensibility import current_language
        for idx in self._top(vector):  # for idx in vector.argsort()[-1:-4:-1]:
            yield 100. * vector[idx], current_language.to_text(idx)


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
        logger = logging.getLogger(type(self).__name__)
        try:
            logger.info(f"Applying %r", edit)
            source_code = edit.apply(self.vector).to_source_code()
        except NoSourceRepresentationError:
            logger.warn(f"No source representation for %r", edit)
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


def zap_zeros_inplace(dist: np.ndarray) -> np.ndarray:
    """
    Ensure the distribution has no zeros
    """
    dist[dist == 0] = epsilon
    return dist


def cross_entropy(true_dist: np.ndarray, est_dist: np.ndarray) -> float:
    """
    Calculate the average bits needed to describe true distribution using
    estimated distribution.
    """
    assert len(true_dist) == len(est_dist)
    return -(true_dist * np.log(zap_zeros_inplace(est_dist))).sum()


def one_hot(idx: Vind, size: int) -> np.ndarray:
    from sensibility import current_language
    dist: np.ndarray = np.zeros((size,), dtype=np.float32)
    dist[idx] = 1.0
    return dist
