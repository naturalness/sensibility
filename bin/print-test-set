#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Prints the hashes of files elligible to form the test set.

These files are:

    - not from projects in the train/validation set
    - not minimized files
    - have defined vectors in the given vectors file

"""

raise NotImplementedError("""TODO:
 - must search from USABLE source files
""")

import sys

from fnmatch import fnmatch
from pathlib import Path

from sensibility import Corpus, Vectors


def is_minified(path_name):
    return fnmatch(path_name, '*.min.js')


if __name__ == '__main__':
    _, vector_filename, corpus_filename = sys.argv
    assert Path(vector_filename).exists()
    assert Path(corpus_filename).exists()

    vectors = Vectors.connect_to(vector_filename)
    corpus = Corpus.connect_to(corpus_filename)

    assert vectors.has_fold_assignments

    for file_hash in vectors.unassigned_files:
        _, _, filename = corpus.file_info(file_hash)

        # Skip minified files.
        if is_minified(filename):
            continue

        print(file_hash)
