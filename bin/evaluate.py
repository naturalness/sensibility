#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Evaluates the performance of the detecting and fixing syntax errors.
"""

import csv
import math
import sys
import tempfile
import traceback
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
    serialize_tokens, vocabulary,
)
from sensibility.tokenize_js import tokenize_file, check_syntax_file
from sensibility.mutations import Mutations
from sensibility.predictions import Predictions, Contexts
from sensibility._paths import VECTORS_PATH, SOURCES_PATH, DATA_DIR


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
        self.total_variation = .5 * norm(a - b, p=1)

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


class SensibilityForEvaluation:
    """
    Detects and fixes syntax errors in JavaScript files.
    """
    context_length = 20

    def __init__(self, fold: int) -> None:
        self.predictions = Predictions(fold)

    def rank_and_fix(self, filename: str, k: int=3) -> FixResult:
        """
        Rank the syntax error location (in token number) and returns a possible
        fix for the given filename.
        """

        # Get file vector for the incorrect file.
        with cast(TextIO, open(filename, 'rt', encoding='UTF-8')) as script:
            tokens = tokenize_file(script)
        file_vector = serialize_tokens(tokens)
        assert len(file_vector) > 0

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

            # Assume an addition. Let's try removing the offensive token.
            fixes.try_delete(pos)

            # Assume a deletion. Let's try inserting some tokens.
            fixes.try_insert(pos, likely_next)
            fixes.try_insert(pos, likely_prev)

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
            return check_syntax_file(cast(TextIO, source_file))


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
        self.sensibility = SensibilityForEvaluation(fold)

    def __enter__(self) -> None:
        self._file = open(self._filename, 'w')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def __exit__(self, *exc_info) -> None:
        self._file.close()

    def run(self) -> None:
        """
        Run the evaluation.
        """
        SourceFile.vectors = Vectors.connect_to(VECTORS_PATH)
        SourceFile.corpus = Corpus.connect_to(SOURCES_PATH)

        with self:
            mutations = self.filter_mutations()
            for program, mutation in tqdm(mutations):
                self.evaluate_mutant(program, mutation)

    def filter_mutations(self) -> Iterator[Tuple[SourceFile, Edit]]:
        """
        Filter only the relevant mutations.
        """

        # Figure out which hashes are acceptable.
        with open(DATA_DIR / f'test_set_hashes.{self.fold}') as f:
            hashes = frozenset(s.strip() for s in f.readlines()
                               if len(s) > 2)

        with Mutations(read_only=True) as all_mutations:
            i = 0
            for entry in all_mutations:
                program, mutation = entry
                if program.file_hash not in hashes:
                    continue
                yield entry

    def evaluate_mutant(self, program: SourceFile, mutation: Edit) -> None:
        try:
            self._evaluate_mutant(program, mutation)
        except Exception:
            self.log_exception(program, mutation)

    def _evaluate_mutant(self, program: SourceFile, mutation: Edit) -> None:
        """
        Evaluate one particular mutant.
        """
        # Figure out the line of the mutation in the original file.
        correct_line = program.line_of_index(mutation.index, mutation)

        # Apply the original mutation.
        mutant = mutation.apply(program.vector)
        with temporary_program(mutant) as mutated_file:
            # Do the (canned) prediction...
            ranked_locations, fixes = self.rank_and_fix(mutated_file)
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

    def rank_and_fix(self, mutated_file: TextIO) -> FixResult:
        """
        Try to fix the given source file and return the results.
        """
        return self.sensibility.rank_and_fix(mutated_file.name)

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
                "fixed": piratize(False),
            })
        else:
            fix = fixes[0]
            kind, loc, new_tok, old_tok = fix.serialize()
            row.update({
                "fixed": piratize(True),
                "true.fix": piratize(fix == mutation.additive_inverse()),
                "f.kind": kind,
                "f.loc": loc,
                "f.token": to_text(new_tok),
                "f.old": to_text(old_tok),
            })

        self._writer.writerow(row)
        self._file.flush()

    def log_exception(self, program: SourceFile, mutation: Edit) -> None:
        with open('failures.txt', 'at') as failures:
            line = '=' * 78
            failures.write(f"{line}\n")
            failures.write(f"Error evaluating {mutation!r} on {program!r}\n")
            traceback.print_exc(file=failures)
            failures.write(f"{line}\n\n")


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


def to_text(token: Optional[Vind]) -> Optional[str]:
    """
    Converts the token to its textual representation, if it exists.
    """
    return None if token is None else vocabulary.to_text(token)


def piratize(value: bool) -> str:
    """
    Convert a Python `bool` into an R `logical`.
    """
    return 'TRUE' if value else 'FALSE'


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


def fix_zeros(preds, epsilon=sys.float_info.epsilon):
    """
    Replace zeros with very small values to avoid NaNing out.
    """
    for i, pred in enumerate(preds):
        if math.isclose(pred, 0.0):
            preds[i] = epsilon


def first_with_line_no(ranked_locations: Sequence[IndexResult],
                       mutation: Edit,
                       correct_line: int,
                       program: SourceFile) -> int:
    """
    Return the first result with the given correct line number.

    Sometimes this fails and I'm not sure why!
    """
    for rank, location in enumerate(ranked_locations, start=1):
        if program.line_of_index(location.index, mutation) == correct_line:
            return rank
    raise ValueError(f'Could not find any token on {correct_line}')


if __name__ == '__main__':
    fold = int(sys.argv[1])
    Evaluation(fold).run()
