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
import argparse

import os
import sys

from tqdm import tqdm
from path import Path

from corpus import Corpus
from condensed_corpus import CondensedCorpus


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=Path)
parser.add_argument('min_rowid', nargs='?', type=int, default=None)
parser.add_argument('max_rowid', nargs='?', type=int, default=None)


def main():
    args = parser.parse_args()
    corpus = Corpus.connect_to(args.filename)

    min_rowid = args.min_rowid if args.min_rowid is not None else 1
    max_rowid = int(max_rowid) if args.max_rowid is not None else len(corpus)
    assert min_rowid <= max_rowid

    dest_filename = Path('vector-corpus-{}.sqlite3'.format(os.getpid()))
    assert not dest_filename.exists(), dest_filename
    destination = CondensedCorpus.connect_to(dest_filename)

    # Insert every file in the given subset.
    files = corpus.iterate(min_rowid=min_rowid,
                           max_rowid=max_rowid,
                           with_hash=True)
    progress_bar = tqdm(files, initial=min_rowid, total=max_rowid)

    for file_hash, tokens in progress_bar:
        try:
            if len(tokens) == 0:
                logging.warn('Skipping empty file: %s', file_hash)
            else:
                progress_bar.set_description('Processing %s' % (file_hash,))
                destination.insert(file_hash, tokens)
        except KeyboardInterrupt:
            logging.exception('Last file before interrupt: %s',
                              file_hash)
            break


if __name__ == '__main__':
    log_name = "{:s}.{:d}.log".format(__file__, os.getpid())
    logging.basicConfig(filename=log_name)
    main()
