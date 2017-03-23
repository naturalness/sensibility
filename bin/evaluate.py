#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Evaluates the performance of the fixer thing.
"""

# TODO: make a big EvaluationResult object -> allow it to be encoded as JSON.
# TODO: try cosine distance      -,
# TODO: try indexing and adding  -'

import csv
import tempfile
import math
import sys
import json
from pathlib import Path
from typing import Optional, TextIO, Sequence

from tqdm import tqdm

from sensibility import (
    SourceFile, Edit, Corpus, Vectors, Agreement, Vind,
    vectorize_tokens, vocabulary,
)
from sensibility.mutations import Mutations
from sensibility.predictions import Predictions
from sensibility._paths import VECTORS_PATH, SOURCES_PATH


def fix_zeros(preds, epsilon=sys.float_info.epsilon):
    """
    Replace zeros with really small values..
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


def squared_error_agreement(prefix_pred, suffix_pred):
    """
    Return the agreement (probability) of this token using sum of squared
    errors.
    """
    # Pretend the sum of squared error is like the cross-entropy of
    # prefix and suffix.
    entropy = ((prefix_pred - suffix_pred) ** 2).sum()
    return -entropy


class SensibilityForEvaluation:
    sentence_length = 20

    def __init__(self, fold_no):
        name = 'javascript-{dir}-300-20.{fold}.5.h5'
        forwards = ModelRecipe.from_string(name.format(dir='f', fold=fold_no))
        backwards = ModelRecipe.from_string(name.format(dir='b', fold=fold_no))
        db = mutations
        self.forwards_predict = lambda prefix: db.get_prediction(model_recipe=forwards, context=prefix)
        self.backwards_predict = lambda suffix: db.get_prediction(model_recipe=backwards, context=suffix)

    def rank_and_fix(self, filename, k=3):
        """
        Rank the syntax error location (in token number) and returns a possible
        fix for the given filename.
        """

        # Get file vector for the incorrect file.
        with open(str(filename), 'rt', encoding='UTF-8') as script:
            # NOTE! INDICIES IN tokens ARE OFFSET BY ONE (- 1 from other
            # mentions of "index")
            tokens = tokenize_file(script)
        file_vector = vectorize_tokens(tokens)

        padding = self.sentence_length

        # Holds the lowest agreement at each point in the file.
        least_agreements = []

        # These will hold the TOP predictions at a given point.
        forwards_predictions = [None] * padding
        backwards_predictions = [None] * padding
        contexts = enumerate(self.contexts(file_vector), start=padding)

        for index, ((prefix, token), (suffix, _)) in contexts:
            assert token == file_vector[index], (
                str(token) + ' ' + str(file_vector[index])
            )

            # Fetch predictions.
            prefix_pred = self.forwards_predict(prefix)
            suffix_pred = self.backwards_predict(suffix)

            assert math.isclose(sum(prefix_pred), 1.0, rel_tol=0.01)
            assert math.isclose(sum(suffix_pred), 1.0, rel_tol=0.01)

            # Store the TOP prediction from both models.
            top_next_prediction = index_of_max(prefix_pred)
            forwards_predictions.append(top_next_prediction)
            top_prev_prediction = index_of_max(suffix_pred)
            backwards_predictions.append(top_prev_prediction)
            assert top_next_prediction == forwards_predictions[index]

            agreement = Agreement(
                squared_error_agreement(prefix_pred, suffix_pred),
                index
            )
            least_agreements.append(agreement)

        # The loop may not have executed at all -- return empty.
        if not least_agreements:
            return [], None

        fixes = Fixes(tokens, offset=-1)

        # For the top disagreements, synthesize fixes.
        least_agreements.sort()
        for disagreement in least_agreements[:k]:
            pos = disagreement.index

            # Assume an addition. Let's try removing some tokens.
            fixes.try_remove(pos)

            likely_next = id_to_token(forwards_predictions[pos])
            likely_prev = id_to_token(backwards_predictions[pos])

            # Assume a deletion. Let's try inserting some tokens.
            if likely_next is not None:
                fixes.try_insert(pos, likely_next)
            if likely_prev is not None:
                fixes.try_insert(pos, likely_prev)

            # Assume it's a substitution. Let's try swapping the token.
            if likely_next is not None:
                fixes.try_substitute(pos, likely_next)
            if likely_prev is not None:
                fixes.try_substitute(pos, likely_prev)

        fix = None if not fixes else tuple(fixes)[0]
        return least_agreements, fix

    def contexts(self, file_vector):
        """
        Yield every context (prefix, suffix) in the given file vector.
        """
        sent_forwards = Sentences(file_vector,
                                  size=self.sentence_length,
                                  backwards=False)
        sent_backwards = Sentences(file_vector,
                                   size=self.sentence_length,
                                   backwards=True)
        return zip(sent_forwards, chop_prefix(sent_backwards))

    @staticmethod
    def is_okay(filename):
        """
        Check if the syntax is okay.
        """
        with open(filename, 'rb') as source_file:
            return check_syntax_file(source_file)


def to_text(token: Optional[Vind]) -> Optional[str]:
    return None if token is None else vocabulary.to_text(token)


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
              program: SourceFile,  # filehash, n.lines, n.tokens
              mutation: Edit,  # m.kind m.pos m.token m.old
              fix: Optional[Edit], # fixed true.fix f.kind f.pos f.token f.fold
              line_of_top_rank: int,
              rank_correct_line: int) -> None:

        kind, loc, new_tok, old_tok = mutation.serialize()
        row = {
            "fold": fold,
            "filehash": program.file_hash,
            "n.lines": program.sloc,
            "n.tokens": len(program.source_tokens),
            "m.kind": kind,
            "m.loc": loc,
            "m.token": to_text(new_tok),
            "m.old": to_text(old_tok),
        }

        if fix is None:
            row.update({
                "fixed": False,
                "true.fix": None,
                "f.kind": None,
                "f.loc": None,
                "f.token": None,
                "f.old": None,
            })
        else:
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

        with self, Mutations() as mutations:
            for program, mutation in tqdm(mutations.for_fold(self.fold)):
                self.evaluate_mutant(program, mutation)

    def evaluate_mutant(self, program: SourceFile, mutation: Edit) -> None:
        """
        Evaluate one particular mutant.
        """
        # Figure out the line of the mutation in the original file.
        correct_line = program.line_of_token(mutation.index)

        # Apply the original mutation.
        with apply_mutation(mutation, program) as mutated_file:
            # Do the (canned) prediction...
            ranked_locations, fix = rank_and_fix(fold, mutated_file)

        if not ranked_locations:
            # rank_and_fix() can occasional return a zero-sized list.
            # In which case, return early.
            return

        # Figure out the rank of the actual mutation.
        top_error_index = ranked_locations[0].index
        line_of_top_location = program.source_tokens[top_error_index].line
        rank_correct_line = first_with_line_no(ranked_locations, mutation,
                                               correct_line, program)

        self.write(program=program,
                   mutation=mutation,
                   fix=fix,
                   line_of_top_rank=line_of_top_location,
                   rank_correct_line=rank_correct_line)


def apply_mutation(mutation, program):
    mutated_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='UTF-8')
    # Apply the mutatation and write it to disk.
    mutation.format(program, mutated_file)
    mutated_file.flush()
    return mutated_file


def rank_and_fix(fold: int, mutated_file: TextIO):
    sensibility = SensibilityForEvaluation(fold)
    return sensibility.rank_and_fix(mutated_file.name)


def first_with_line_no(ranked_locations: Sequence[Agreement],
                       mutation: Edit,
                       correct_line: int,
                       program: SourceFile) -> int:
    for rank, location in enumerate(ranked_locations, start=1):
        if program.line_of_token(location.index, mutation) == correct_line:
            return rank
    raise ValueError(f'Could not find any location on {correct_line}')



if __name__ == '__main__':
    fold = int(sys.argv[1])
    Evaluation(fold).run()
