#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Evaluates the performance of the fixer thing.
"""

import csv

from tqdm import tqdm

from mutate import Persistence as Mutations
from corpus import Corpus


class Results:
    FIELDS = '''
        fold file
        mkind mpos mtoken
        correct_line
        rank_correct_line
        fixed fkind fpos ftoken same_fix
    '''.split()

    def __enter__(self):
        self._file = open('results.csv', 'w')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def __exit__(self, *exc_info):
        self._file.close()

    def write(self, **kwargs):
        self._writer.writerow(kwargs)


FOLDS = {}

def populate_folds():
    """
    Create a mapping between a file hash and the fold it came from.
    """
    for fold in 0, 1, 2, 3, 4:
        with open('test_set_hashes.' + str(fold)) as hash_file:
            for file_hash in hash_file:
                FOLDS[file_hash.strip()] = fold


def apply_mutation(mutation, tokens):
    raise NotImplementedError


def rank_and_fix(mutation):
    raise NotImplementedError
    return [], None


def first_with_line_no(ranked_locations, correct_line, tokens):
    raise NotImplementedError


if __name__ == '__main__':
    corpus = Corpus.connect_to('javascript-sources.sqlite3')
    populate_folds()

    with Mutations() as mutations, Results() as results:
        for file_hash, mutation in tqdm(mutations):
            print(mutation)
            # Figure out what fold it is.
            fold_no = FOLDS[file_hash]
            tokens = corpus.get_tokens(file_hash)
            # Apply the mutation and figure out the line of the mutation in the original file.
            mutated_file, correct_line = apply_mutation(mutation, tokens)
            # Do the (canned) prediction...
            ranked_locations, fix = rank_and_fix(mutated_file)
            rank_correct_line = first_with_line_no(ranked_locations,
                                                   correct_line, tokens)
            results.write(
                fold=fold_no,
                mkind=mutation.name,
                mtoken=mutation.token,
                mpos=mutation.location,
                correct_line=correct_line,
                rank_correct_line=rank_correct_line,
                fixed=bool(fix),
                fkind=fix.kind,
                fpos=fix.location,
                ftoken=fix.token,
                same_fix=...
            )
