#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Evaluates the performance of the detecting and fixing syntax errors.
"""

import csv
import sys
import traceback
from pathlib import Path
from typing import Iterator, Optional, Sequence, TextIO, Tuple

from tqdm import tqdm

from sensibility import (
    Corpus, Edit, Sensibility, SourceFile, Vectors, Vind,
    vocabulary
)
from sensibility.tokenize_js import tokenize_file, check_syntax_file
from sensibility.mutations import Mutations
from sensibility._paths import VECTORS_PATH, SOURCES_PATH, DATA_DIR
from sensibility.fix import FixResult, IndexResult, temporary_program


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
        self.sensibility = Sensibility(fold)

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
