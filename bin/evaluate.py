#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Evaluates the performance of the detecting and fixing syntax errors.
"""

import csv
import tempfile
import math
import sys
import json
from pathlib import Path
from typing import (
    Iterable, Iterator, SupportsFloat, Sequence,
    List, Optional,
    NamedTuple, Tuple,
    TextIO, cast,
)

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from sensibility import (
    SourceFile,
    Edit, Insertion, Deletion, Substitution,
    SourceVector,
    Corpus, Vectors,
    Vind,
    vectorize_tokens, vocabulary,
)
from sensibility.tokenize_js import tokenize_file, check_syntax_file
from sensibility.mutations import Mutations
from sensibility.predictions import Predictions, Contexts
from sensibility._paths import VECTORS_PATH, SOURCES_PATH, DATA_DIR

# TODO: make an IndexResult object including:
#   - harmonic mean
#   - sum (???)


class IndexResult(SupportsFloat):
    """
    Provides results for EACH INDIVIDUAL INDEX in a file.
    """
    __slots__ = (
        'index',
        'cosine_similarity', 'squared_euclidean_distance',
        'harmonic_mean', 'indexed_sum'
    )

    def __init__(self, index: int, program: SourceVector,
                 a: np.ndarray, b: np.ndarray) -> None:
        assert 0 <= index < len(program)
        self.index = index

        # Categorical distributions MUST have |x|_1 == 1.0
        assert is_normalized_vector(a, p=1) and is_normalized_vector(b, p=1)

        self.harmonic_mean = ...
        self.squared_euclidean_distance = ((a - b) ** 2).sum()

        # How probable is the token at this position?
        # 2.0 == both models absoultely think this token should be here.
        # 1.0 == lukewarm---possibly one model thinks this token should be here
        # 0.0 == both models are perplexed by this token.
        self.indexed_sum = a[program[index]] + b[program[index]]

        # How similar are the two vectors?
        self.cosine_similarity = (a @ b) / (norm(a) * norm(b))

    def __float__(self) -> float:
        """
        Returns the score between the two elements.
        """
        # We can tweak λ to weigh local and global factors differently,
        # but for now, weigh them equally.
        # TODO: use sklearn's Lasso regression to find this coefficient?
        # TODO:                 `-> fit_intercept=False
        λ = .5
        score = λ * self.indexed_sum / 2. + (1. - λ) * self.cosine_similarity
        assert 0. <= score <= 1.
        return score


class FixResult(NamedTuple):
    # The results of detecting and fixing syntax errors.
    ranks: Sequence[IndexResult]
    fixes: Sequence[Edit]


class SensibilityForEvaluation:
    """
    Detects and fixes syntax errors in JavaScript files.
    """
    context_length = 20

    def __init__(self, fold: int) -> None:
        self.predictions = Predictions(fold)

    def rank_and_fix(self, filename: str, k: int=4) -> FixResult:
        """
        Rank the syntax error location (in token number) and returns a possible
        fix for the given filename.
        """

        # Get file vector for the incorrect file.
        with cast(TextIO, open(filename, 'rt', encoding='UTF-8')) as script:
            tokens = tokenize_file(script)
        file_vector = SourceVector(vectorize_tokens(tokens))
        assert len(file_vector) > 0

        padding = self.context_length

        # Holds the lowest agreement at each point in the file.
        results: List[IndexResult] = []

        # These will hold the TOP predictions at a given point.
        forwards_predictions: List[Vind] = []
        backwards_predictions: List[Vind] = []
        contexts = enumerate(self.contexts(file_vector))

        for index, ((prefix, token), (suffix, _)) in contexts:
            assert token == file_vector[index], (
                str(token) + ' ' + str(file_vector[index])
            )

            # Fetch predictions.
            prefix_pred = np.array(self.predictions.predict_forwards(prefix))
            suffix_pred = np.array(self.predictions.predict_backwards(suffix))

            result = IndexResult(index, file_vector, prefix_pred, suffix_pred)
            results.append(result)

            # Store the TOP prediction from each model.
            # TODO: document corner cases!
            top_next_prediction = prefix_pred[prefix_pred.argmax()]
            forwards_predictions.append(cast(Vind, top_next_prediction))
            top_prev_prediction = suffix_pred[suffix_pred.argmax()]
            backwards_predictions.append(cast(Vind, top_prev_prediction))
            assert top_next_prediction == forwards_predictions[index]

        # Rank the results by some metric of similarity defined by IndexResult
        # (the top rank will be LEAST similar).
        ranked_results = tuple(sorted(results, key=float))

        # For the top disagreements, synthesize fixes.
        fixes = Fixes(file_vector)
        for disagreement in ranked_results[:k]:
            pos = disagreement.index

            # Assume an addition. Let's try removing the offensive token.
            fixes.try_delete(pos)

            likely_next: Vind = forwards_predictions[pos]
            likely_prev: Vind = backwards_predictions[pos]

            # Assume a deletion. Let's try inserting some tokens.
            fixes.try_insert(pos, likely_next)
            fixes.try_insert(pos, likely_prev)

            # Assume it's a substitution. Let's try swapping the token.
            if likely_next is not None:
                fixes.try_substitute(pos, likely_next)
            if likely_prev is not None:
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
            return check_syntax_file(cast(TextIO, source_file))


def is_normalized_vector(x: np.ndarray, p: int=2, tolerance=0.01) -> bool:
    """
    Returns whether the vector is normalized

    >>> is_normalized_vector(np.array([ 0.,  0.,  1.]))
    True
    >>> is_normalized_vector(np.array([ 0.5 ,  0.25,  0.25]))
    False
    >>> is_normalized_vector(np.array([ 0.5 ,  0.25,  0.25]), p=1)
    True
    """
    return math.isclose(norm(x, p), 1.0, rel_tol=tolerance)


class Evaluation:
    FIELDS = '''
        fold filehash n.lines n.tokens
        m.kind m.loc m.token m.old
        correct.line line.top.rank rank.correct.line
        fixed true.fix
        f.kind f.loc f.token f.old
    '''.split()

    def __init__(self, fold: int) -> None:
        assert 0 <= fold < 5
        self.fold = fold
        self._filename = f'results.{fold}.csv'

    def __enter__(self) -> None:
        self._file = open(self._filename, 'w')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def __exit__(self, *exc_info) -> None:
        self._file.close()

    def write(self, *,
              program: SourceFile,
              mutation: Edit,
              fixes: Sequence[Edit],
              correct_line: int,
              line_of_top_rank: int,
              rank_of_correct_line: int) -> None:

        kind, loc, new_tok, old_tok = mutation.serialize()
        row = {
            # Meta information
            "fold": fold,
            "filehash": program.file_hash,
            "n.lines": program.sloc,
            "n.tokens": len(program.vector),
            # Mutation information
            "m.kind": kind,
            "m.loc": loc,
            "m.token": to_text(new_tok),
            "m.old": to_text(old_tok),

            # Fault locatization information.
            "line.top.rank": line_of_top_rank,
            "correct.line": correct_line,
            "rank.correct.line": rank_of_correct_line,
        }

        # Information about the fix (if at least one exists).
        if len(fixes) == 0:
            row.update({
                "fixed": False,
                "true.fix": None,
                "f.kind": None,
                "f.loc": None,
                "f.token": None,
                "f.old": None,
            })
        else:
            fix = fixes[0]
            kind, loc, new_tok, old_tok = fix.serialize()
            row.update({
                "fixed": True,
                "true.fix": fix == mutation.additive_inverse(),
                "f.kind": kind,
                "f.loc": loc,
                "f.token": to_text(new_tok),
                "f.old": to_text(old_tok),
            })

        self._writer.writerow(row)
        self._file.flush()

    def run(self) -> None:
        """
        Run the evaluation.
        """
        SourceFile.vectors = Vectors.connect_to(VECTORS_PATH)
        SourceFile.corpus = Corpus.connect_to(SOURCES_PATH)

        with open(DATA_DIR / 'test_set_hashes.{self.fold}') as f:
            hashes = frozenset(s.strip() for s in f.readlines() if len(s) > 2)

        with self, Mutations(read_only=True) as all_mutations:
            mutations = (m for m in all_mutations if m[0].file_hash in hashes)
            for program, mutation in tqdm(mutations):
                self.evaluate_mutant(program, mutation)

    def evaluate_mutant(self, program: SourceFile, mutation: Edit) -> None:
        """
        Evaluate one particular mutant.
        """
        # Figure out the line of the mutation in the original file.
        correct_line = program.line_of_index(mutation.index, mutation)

        # Apply the original mutation.
        mutant = mutation.apply(program.vector)
        with temporary_program(mutant) as mutated_file:
            # Do the (canned) prediction...
            ranked_locations, fixes = rank_and_fix(fold, mutated_file)
        assert len(ranked_locations) > 0

        # Figure out the rank of the actual mutation.
        top_error_index = ranked_locations[0].index
        line_of_top_rank = program.source_tokens[top_error_index].line
        rank_of_correct_line = first_with_line_no(ranked_locations, mutation,
                                                  correct_line, program)

        self.write(program=program,
                   mutation=mutation, fixes=fixes,
                   correct_line=correct_line,
                   line_of_top_rank=line_of_top_rank,
                   rank_of_correct_line=rank_of_correct_line)


def temporary_program(program: SourceVector) -> TextIO:
    """
    Returns a temporary file that with the contents of the file written to it.
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


def to_text(token: Optional[Vind]) -> Optional[str]:
    """
    Converts the token to its textual representation, if it exists.
    """
    return None if token is None else vocabulary.to_text(token)


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


def fix_zeros(preds, epsilon=sys.float_info.epsilon):
    """
    Replace zeros with really small values...
    """
    for i, pred in enumerate(preds):
        if math.isclose(pred, 0.0):
            preds[i] = epsilon


def harmonic_mean_agreement(prefix_pred, suffix_pred):
    # Avoid NaNs
    fix_zeros(prefix_pred)
    fix_zeros(suffix_pred)

    mean = harmonic_mean(prefix_pred, suffix_pred)
    # Rank the values from lowest to top (???)
    paired_rankings = rank(mean)
    # Get the value with the minimum probability (???)
    _, min_prob = paired_rankings[0]

    return min_prob


def rank_and_fix(fold: int, mutated_file: TextIO) -> FixResult:
    sensibility = SensibilityForEvaluation(fold)
    return sensibility.rank_and_fix(mutated_file.name)


def first_with_line_no(ranked_locations: Sequence[IndexResult],
                       mutation: Edit,
                       correct_line: int,
                       program: SourceFile) -> int:
    """
    Return the first result with the given correct line number.
    """
    for rank, location in enumerate(ranked_locations, start=1):
        if program.line_of_index(location.index, mutation) == correct_line:
            return rank
    raise ValueError(f'Could not find any location on {correct_line}')


if __name__ == '__main__':
    fold = int(sys.argv[1])
    Evaluation(fold).run()
