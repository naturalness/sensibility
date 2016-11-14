#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2016 Eddie Antonio Santos <easantos@ualberta.ca>
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
Creates matrix-corpus.sqlite3 from a full corpus stored as an SQLite database.

The file contains one-hot encoded matrices.
"""

import logging

import os
import sys

from tqdm import tqdm
from path import Path

from corpus import Corpus
from condensed_corpus import CondensedCorpus


def main():
    _, filename, min_rowid, max_rowid = sys.argv
    min_rowid = int(min_rowid)
    max_rowid = int(max_rowid)
    assert min_rowid <= max_rowid

    dest_filename = Path('matrix-corpus-{}.sqlite3'.format(os.getpid()))
    assert not dest_filename.exists()

    corpus = Corpus.connect_to(filename)
    destination = CondensedCorpus.connect_to(dest_filename)

    # Insert every file in the given subset.
    files = corpus.iterate(min_rowid=min_rowid,
                           max_rowid=max_rowid,
                           with_hash=True)
    for file_hash, tokens in tqdm(files, total=max_rowid - min_rowid):
        if len(tokens) == 0:
            logging.warn('Skipping empty file: %s', file_hash)
        else:
            destination.insert(file_hash, tokens)


if __name__ == '__main__':
    log_name = "{:s}.{:d}.log".format(__file__, os.getpid())
    logging.basicConfig(filename=log_name)
    main()
