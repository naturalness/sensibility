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
Evaluates the performance of the detecting and fixing syntax errors.
"""

import array
import csv
import functools
import logging
import numpy as np
import sqlite3
import sys
import traceback

from sensibility.sentences import Sentence, T, forward_sentences, backward_sentences
from abc import ABC, abstractmethod
from numpy.linalg import norm  # noqa
from pathlib import Path
from sensibility.edit import Edit, Insertion, Deletion, Substitution
from sensibility.evaluation.distance import FixEvent
from sensibility.language import language
from sensibility.source_file import SourceFile
from sensibility.source_vector import SourceVector, to_source_vector
from sensibility.vocabulary import Vind
from tqdm import tqdm
from typing import Iterable, Iterator, Optional, Sequence, TextIO, Tuple
from typing import NamedTuple
from typing import Sequence, Tuple, Union, TextIO, Iterable, cast
from typing import SupportsFloat  # noqa
from typing import cast
from typing import Set
from typing import List  # noqa
from sensibility.model import Model

from sensibility._paths import MODEL_DIR

# TODO: fetch from the database instead.
# TODO: make language-agnostic (with priority for Java)


# A type that neatly summarizes the double contexts.
Contexts = Iterable[Tuple[Sentence[Vind], Sentence[Vind]]]


class Predictions:
    """
    Caches predictions.
    """

    SCHEMA = r"""
    PRAGMA encoding = "UTF-8";

    CREATE TABLE IF NOT EXISTS prediction (
        model   TEXT NOT NULL,      -- model that created the prediction
        context BLOB NOT NULL,      -- input of the prediction, as an array
        vector  BLOB NOT NULL,      -- prediction data, as an array

        PRIMARY KEY (model, context)
    );
    """

    def __init__(self, partition: int) -> None:
        from sensibility._paths import get_lstm_path, get_cache_path
        assert 0 <= partition < 5
        self.forwards_model = Model.from_filename(get_lstm_path('f', partition))
        self.backwards_model = Model.from_filename(get_lstm_path('b', partition), backwards=True)
        # XXX: Hard code the context length!
        self.context_length = 20
        self._conn = self._connect(get_cache_path())

        forwards = f'f{partition}'
        backwards = f'b{partition}'

        def _predict(name: str, model: Model,
                     tuple_context: Sequence[Vind]) -> array.array:
            """
            Does prediction, consulting the database first before consulting
            the model.
            """
            context = array.array('B', tuple_context).tobytes()
            try:
                return self.get_prediction(name, context)
            except KeyError:
                prediction = model.predict(tuple_context)
                self.add_prediction(name, context, prediction)
                return prediction  # type: ignore

        # > SELECT MAX(n_tokens) FROM vectorized_source;
        # 1809948
        # ceil(log(1809948, 2)) -- this will be like... 2 GiB PER CACHE.
        cache_size = 2**21

        # Create prediction functions with in-memory caching.
        @functools.lru_cache(maxsize=cache_size)
        def predict_forwards(prefix: Tuple[Vind, ...]) -> array.array:
            return _predict(forwards, self.forwards_model, prefix)

        @functools.lru_cache(maxsize=cache_size)
        def predict_backwards(suffix: Tuple[Vind, ...]) -> array.array:
            return _predict(backwards, self.backwards_model, suffix)

        self.predict_forwards = predict_forwards
        self.predict_backwards = predict_backwards

    def predict(self, source: bytes) -> None:
        """
        Predicts at each position in the file, writing predictions to
        database.
        """

        # Get file vector for this (incorrect) file.
        file_vector = to_source_vector(source)

        # Calculate and store predictions.
        for (prefix, _), (suffix, _) in self.contexts(file_vector):
            self.predict_forwards(tuple(prefix))
            self.predict_backwards(tuple(suffix))

    def clear_cache(self):
        self.predict_forwards.cache_clear()
        self.predict_backwards.cache_clear()

    def contexts(self, file_vector: Sequence[Vind]) -> Contexts:
        """
        Yield every context (prefix, suffix) in the given file vector.
        """
        sent_forwards = forward_sentences(file_vector,
                                          context=self.context_length)
        sent_backwards = backward_sentences(file_vector,
                                            context=self.context_length)
        return zip(sent_forwards, sent_backwards)

    def add_prediction(self, name: str, context: bytes,
                       prediction: np.ndarray) -> None:
        """
        Add the prediction (model, context) -> prediction
        """
        assert self._conn

        vector: bytes = array.array('f', prediction).tobytes()
        with self._conn:
            self._conn.execute('BEGIN')
            self._conn.execute(r'''
                INSERT INTO prediction(model, context, vector)
                VALUES (:model, :context, :vector)
            ''', dict(model=name, context=context, vector=vector))

    def get_prediction(self, name: str, context: bytes) -> array.array:
        """
        Try to fetch the prediction from the database.

        Returns None if the entry is not found.
        """
        assert self._conn
        cur = self._conn.execute(r'''
            SELECT vector
            FROM prediction
            WHERE model = :model AND context = :context
        ''', dict(model=name, context=context))
        result = cur.fetchall()

        if not result:
            # Prediction not found!
            raise KeyError(f'{name}/{context!r}')
        else:
            # Return the precomputed prediction.
            output = array.array('f')
            output.frombytes(result[0][0])
            return output

    def _connect(self, filename: Path) -> sqlite3.Connection:
        # Connect to the database
        conn = sqlite3.connect(str(filename))
        # Initialize the database.
        with conn:
            conn.executescript(self.SCHEMA)
            # Some speed optimizations:
            # http://codificar.com.br/blog/sqlite-optimization-faq/
            conn.executescript(r'''
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = normal;
            ''')
        return conn


class IndexResult(SupportsFloat):
    """
    Provides results for EACH INDIVIDUAL INDEX in a file.
    """
    __slots__ = (
        'index',
        'cosine_similarity', 'indexed_prob',
        'mutual_info', 'total_variation', 'line_no'
    )

    def __init__(self, index: int, program: SourceVector,
                 a: np.ndarray, b: np.ndarray, line_no: int) -> None:
        assert 0 <= index < len(program)
        self.index = index
        self.line_no = line_no

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
        from sensibility.language import language
        source_code = self.vector.to_source_code()
        if language.check_syntax(source_code):
            self.fixes.append(edit)

    def __bool__(self) -> bool:
        return len(self.fixes) > 0

    def __iter__(self) -> Iterator[Edit]:
        return iter(self.fixes)


class EvaluationFile(ABC):
    id: str  # Uniquely identifies the file
    error: Edit  # Error is a mistake or a mutation
    error_line: int  # Line number of the error
    true_fix: Edit  # The "true" fix that reversed the mistake
    source: bytes  # The source code of the error
    n_lines: int  # Number of SLOC in the file
    n_tokens: int  # Number of tokens in the file.


class Mistake(EvaluationFile):
    """
    Mistake from BlackBox data
    """

    def __init__(self, id: str, source: bytes, event: FixEvent) -> None:
        self.id = id
        self.error = event.mistake
        self.error_line = event.line_no
        self.true_fix = event.fix
        summary = language.summarize(source)
        self.source = source
        self.n_lines = summary.sloc
        self.n_tokens = summary.n_tokens


def subset_from_file(filename: Path) -> Set[Tuple[int, int]]:
    def gen():
        with open(filename) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                a, b = line.split(',')
                yield int(a), int(b)
    return set(gen())


def fetch_mistakes(subset: Set[Tuple[int, int]]) -> Iterator[EvaluationFile]:
    from sensibility._paths import get_mistakes_path
    conn = sqlite3.connect(str(get_mistakes_path()))
    query = '''
        SELECT mistake.source_file_id,
               mistake.before_id,
               before as source,
               line_no,
               fix, fix_index, new_token, old_token
          FROM mistake NATURAL JOIN edit
    '''
    for row in conn.execute(query):
        sfid, meid, source, line_no, f_kind, f_ind, f_new, f_old = row
        if (sfid, meid) not in subset:
            continue
        fix = Edit.deserialize(f_kind, int(f_ind), to_index(f_new), to_index(f_old))
        yield Mistake(f"{sfid}/{meid}", source, FixEvent(fix, line_no))


def to_index(token: Optional[str]) -> Optional[Vind]:
    return language.vocabulary.to_index(token) if token is not None else None


class Mutant(EvaluationFile):
    """
    Sythetic mutant.
    """
    # TODO:


class LSTMPartition(Model):
    """
    Detects and fixes syntax errors in JavaScript files.
    """
    context_length = 20

    def __init__(self, partition: int) -> None:
        assert partition in {0, 1, 2, 3, 4}
        self.id = f'lstm{partition}'
        # Load the model
        self.predictions = Predictions(partition)

    def fix(self, source: bytes, k: int=4) -> FixResult:
        """
        Rank the syntax error location (in token number) and returns a possible
        fix for the given filename.
        """

        vocabulary = language.vocabulary

        # Get file vector for the error'd file.
        file_vector = to_source_vector(source)

        from sensibility.lexical_analysis import Token
        all_toks: List[Token] = list(language.tokenize(source))

        assert len(file_vector) > 0

        # Holds the lowest agreement at each point in the file.
        results: List[IndexResult] = []

        # These will hold the TOP predictions at a given point.
        forwards_predictions = []  # type: List[Vind]
        backwards_predictions: List[Vind] = []
        contexts = enumerate(self.contexts(file_vector))

        for index, ((prefix, token), (suffix, _)) in contexts:
            assert token == file_vector[index], f'{token} != {file_vector[index]}'
            line_no = all_toks[index].line

            # Fetch predictions.
            prefix_pred = np.array(self.predictions.predict_forwards(prefix))  # type: ignore
            suffix_pred = np.array(self.predictions.predict_backwards(suffix))  # type: ignore

            result = IndexResult(index, file_vector, prefix_pred, suffix_pred, line_no)
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

            # Assume an insertion. Let's try removing the offensive token.
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


# TODO: Deal with the fact that some mistakes may have NO fixes,
# TODO: and some may have SEVERAL fixes

from typing import NewType  # noqa


RLogical = NewType('RLogical', str)


class EvaluationResult(NamedTuple):
    model: str  # 'lstm{i}'
    mode: str  # 'mutant' | 'mistake'
    n_lines: int  # source lines of code
    n_tokens: int  # number of tokens in the error file
    correct_line: int  # Line of the syntax error
    error: Edit  # What was the error (mistake or mutation)?
    fixed: bool
    fixes: Sequence[Edit]  # What is a possible fix?
    line_of_top_ranked: int  # The line number of the top ranked error location
    rank_of_correct_line: Optional[int]  # Rank of the first token on the erro line line

    # I wanted cool methods to add to NamedTuple, but they don't work
    # for some reason so :/


from typing import IO  # noqa
EvaluationFiles = Iterator[EvaluationFile]


class Evaluation:
    FIELDS = '''
        model mode file n.lines n.tokens
        m.kind m.loc m.token m.old
        correct.line line.top.rank rank.correct.line
        fixed true.fix
        f.kind f.loc f.token f.old
    '''.split()

    def __init__(self, mode: str, model: LSTMPartition) -> None:
        self.model = model
        self.mode = mode

    def evaluate_file(self, file: EvaluationFile) -> EvaluationResult:
        results = self.model.fix(file.source)
        top_rank = results.ranks[0]
        locations = tuple(language.token_locations(file.source))
        rank_correct_line = first_with_line_no(results.ranks, file.error_line)

        if rank_correct_line is None:
            import warnings
            line_nums = sorted(set(x.line_no for x in results.ranks))
            warnings.warn(f"{file.id}: Did not find line number "
                          f"{file.error_line} in {line_nums}")

        return EvaluationResult(
            model=self.model.id,
            mode=self.mode,
            n_lines=file.n_lines,
            n_tokens=file.n_tokens,
            correct_line=file.error_line,
            error=file.error,
            fixed=len(results.fixes) > 0,
            fixes=results.fixes,
            line_of_top_ranked=locations[top_rank.index].line,
            rank_of_correct_line=rank_correct_line,
        )

    def evaluate(self, files: EvaluationFiles, output: IO[str]) -> None:
        writer = csv.DictWriter(output, self.FIELDS)
        writer.writeheader()
        for file in files:
            self._evaluate_file(file, writer)

    def _evaluate_file(self, file: EvaluationFile, writer: csv.DictWriter) -> None:
        result = self.evaluate_file(file)
        m_kind, m_loc, m_token, m_old = result.error.serialize()
        # Compute
        #  - was it fixed?
        #  - fix
        #  - line of the top ranked fix
        #  - rank of the first location on the correct line
        #       => assert locations line up with file
        row = {
            'model': result.model,
            'mode': result.mode,
            'file': file.id,
            'n.lines': result.n_lines,
            'n.tokens': result.n_tokens,
            'm.kind': m_kind,
            'm.loc': m_loc,
            'm.token': to_text(m_token),
            'm.old': to_text(m_old),
            'line.top.rank': result.line_of_top_ranked,
            'correct.line': result.correct_line,
            'rank.correct.line':  result.rank_of_correct_line
        }

        if result.fixed:
            row['fixed'] = piratize(True)
            # Create an output for each row.
            for fix in result.fixes:
                f_kind, f_loc, f_token, f_old = fix.serialize()
                row.update({
                    'f.kind': f_kind,
                    'f.loc': f_loc,
                    'f.token': to_text(f_token),
                    'f.old': to_text(f_old),
                    'true.fix': piratize(fix == file.true_fix)
                })
                writer.writerow(row)
        else:
            row['fixed'] = piratize(False)
            writer.writerow(row)


def piratize(value: bool) -> RLogical:
    """
    Convert a Python `bool` into an R `logical`.
    """
    return RLogical('TRUE' if value else 'FALSE')


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


def to_text(token: Optional[Vind]) -> Optional[str]:
    """
    Converts the token to its textual representation, if it exists.
    """
    return None if token is None else language.vocabulary.to_text(token)


def first_with_line_no(ranked_results: Sequence[IndexResult],
                       correct_line: int) -> Optional[int]:
    """
    Return the first result with the given correct line number.

    Sometimes this fails and I'm not sure why!
    """
    for rank, token_result in enumerate(ranked_results, start=1):
        if token_result.line_no == correct_line:
            return rank
    return None
