#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2016, 2017 Eddie Antonio Santos <easantos@ualberta.ca>
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
Places files into folds attempting to equalize the token length.

                            ,                       :::
                                     ,:::::::        ::::,
       ,,,,;;;;;;;;;,,                 ,:::::::,     ::::::,
   ,,,;::::::::::::;,,,                ,::::::::,    :::::,,,
  ;:::::::::::,,,,,                    ,::::::::,,   :::::,,,        +
 ,:::::::,,,,                          ::::::::,,,   ::::,,,      ,
 ,::::;,,,                             :::::::,,,   ~::::,,,
 ,::::::,                              :::::::,,,   ,::::,,         ,:::,
  :::::::,                             ::::::,,, :::::::,,,        :::::::,
  :::::::::,                           ::::::,,::::,::::,,I        :::::::::,
  ,:::::::::,,                 :,      :::::,,::::,,:::,,,        ,:::::::::,,
   :::::::::,,,                ::::    :::::,,:::,,,:::,,,        ,::::::::,,,,
    ::::::::,,,      ::::,,   ::::::, +::::,,:::,,,,:::,,,        :::::::,,,,
    ::::::::,,,    ,:::::,,  ::::::,:,,::::,,:::,,, ::::,,,:,,    :::::::,,,
     :::::::,,,   ,::::::,, ,:::::,,::,::::,,::::,,~:::::::,,,    ::::::,,,
      :::::::,, ::::::::,,, ::::::,,::,:::,,,:::::::,:::::,,,     :::::,,,    +
      ,::::::,::::::::::,,, :::::,,,::,:::,,,::::::,,,::::,,,     :::::,,,
       ::::::::::::,::,,,   ::::,,, ::,::::,,::::::,,,I,,,,,      ,:::,,,
        :::::::::,,,,,      ::::,,  ::,::::::,,,,,,,,::::::I      ~:::,,,
         ::::::,,,,,       ,:::::, :::,:::::,,,  :,? ,:::::,,,     :::,,
     I   ,:::::,,,        ,+::::::::::,,:::,,,,      ?:::::::::,,  :::,,
          :::::::,          :::::::::,,, :::::, ::::::::::,,,,,,,,+::,,,
           :::::::,          ::::::::,,,  ::::::,:::::::::,,,      ,:,,,
            ::::::,,         `;::::,,,,  ,:::::,,::::,::::,,        ::,,
            ,::::::,,        , ,,,,,,,  ::,,,,,,,:::,,::::,,   ::,, ,:,,
             ::::::,,            ,,~    :::::     ,,,,:::::, ,::,,,, :,,
              ::::::,,                  :::::::,    ,:,::::::::,,,  , ,,
              ,:::::,,                  ,::::::,,   ::,:::::::,,,   ,::
               :::::,,I                  ::::::,,  ,:,,,:::::,,,    :::::::
               ,::::,,,                  ,::::::,   ,,,  +,,,,,     :::::::,,
         ,       ,,,,,,                   ::::::::::       ,,      :::::::,,,
                                          ,:::::::,,,,             :::::,,,,,
                                           ::::::,,,,               ,,,,,,,
        I                                  ,::::,,,
                              ?             ,:,,,,                    +
                                              ,,,                         ,
                                                                           +

"""

import argparse
import heapq
import math
import operator
import random
import statistics
import sys

from pathlib import Path
from functools import partial
from itertools import islice
from fnmatch import fnmatch

from tqdm import tqdm

from sensibility import Corpus, Vectors

stderr = partial(print, file=sys.stderr)

parser = argparse.ArgumentParser(description='Divides the corpus into folds.')
parser.add_argument('filename', type=Path, metavar='corpus')
parser.add_argument('-o', '--offset', type=int, default=0, help='default: 0')
parser.add_argument('-k', '--folds', type=int, default=10, help='default: 10')
parser.add_argument('-n', '--min-tokens', type=int, default=None)
parser.add_argument('-f', '--overwrite', action='store_true')

MAIN_CORPUS = Path('javascript-sources.sqlite3')


def main():
    # Creates variables: filename, offset, folds, min_tokens, overwrite
    globals().update(vars(parser.parse_args()))
    assert filename.exists()
    assert folds >= 1
    assert offset >= 0

    vectors = Vectors.connect_to(str(filename))
    corpus = Corpus.connect_to(str(MAIN_CORPUS))

    # We will be assigning to these folds.
    new_fold_ids = tuple(range(offset, offset + folds))

    conflict = set(vectors.fold_ids) & set(new_fold_ids)
    if conflict:
        if overwrite:
            vectors.destroy_fold_assignments()
        else:
            stderr('Not overwriting existing fold assignments:', conflict)
            exit(-1)

    # We maintain a priority queue of folds. At the top of the heap is the
    # fold with the fewest tokens.
    heap = [(0, fold_no) for fold_no in new_fold_ids]
    heapq.heapify(heap)

    # This is kinda dumb, but:
    # Shuffle a list of ALL project...
    stderr('Shuffling projects...')
    shuffled_projects = list(corpus.projects)
    random.shuffle(shuffled_projects)

    hashes_seen = set()
    previously_assigned = set()

    # Add all existing fold assignments to hashes seen.
    for fold_no in vectors.fold_ids:
        for file_hash in vectors.hashes_in_fold(fold_no):
            previously_assigned.add(file_hash)

    # A series of helper functions.
    def appropriate_files(project):
        """
        Yields a random project from the main corpus.
        """
        for file_hash, path in corpus.filenames_from_project(project):
            # Although all files are only stored once in the database,
            # multiple projects could have THE SAME FILE. Skip 'em if that's
            # the case.
            if file_hash in hashes_seen:
                stderr('Ignoring duplicate file:', path, file_hash)
                continue

            if file_hash in previously_assigned:
                # This project has already been assigned.
                break

            hashes_seen.add(file_hash)
            try:
                yield vectors[file_hash]
            except TypeError:
                stderr('Could not find', file_hash)
                continue

    def tokens_in_smallest_fold():
        fewest_tokens, _ = heap[0]
        return fewest_tokens

    def normalish(getter=operator.itemgetter(0)):
        """
        Rule of thumb calculation for normality.
        """
        sample_mean = statistics.mean(map(getter, heap))
        sample_sd = statistics.stdev(map(getter, heap))

        progress.set_description("mean: {}, stddev: {}".format(
            sample_mean, sample_sd
        ))

        def t_statisitic(observation):
            return (observation - sample_mean) / sample_sd

        # Check if the minimum is too far away (it probably isn't).
        min_tokens, _ = heap[0]
        if abs(t_statisitic(min_tokens)) >= 3.0:
            return False

        # Check if the maximum is too far away (it just might be).
        max_tokens, _ = max(heap, key=getter)
        if abs(t_statisitic(max_tokens)) >= 3.0:
            return False

        return True

    def pop():
        return heapq.heappop(heap)

    def push(n_tokens, fold_no):
        assert isinstance(fold_no, int)
        return heapq.heappush(heap, (n_tokens, fold_no))

    if min_tokens is None:
        # Use globals().update() to avoid an UnboundLocal error:
        globals().update(min_tokens=math.inf)

    # Assign a project to each fold...
    for project in tqdm(shuffled_projects):
        # Assign this project to the smallest fold
        tokens_in_fold, fold_no = pop()
        n_tokens = 0

        # Add every file in the project...
        for file_hash, tokens in appropriate_files(project):
            n_tokens += len(tokens)
            vectors.add_to_fold(file_hash, fold_no)

        # Push the fold back.
        push(tokens_in_fold + n_tokens, fold_no)

        if tokens_in_smallest_fold() >= min_tokens:
            break

    # Ensure we have enough tokens!
    if min_tokens is not math.inf:
        assert all(n_tokens > min_tokens for n_tokens, _ in heap), (
            '{}-folds did not acheive minimum token length: '
            '{}'.format(folds, min_tokens)
        )


if __name__ == '__main__':
    main()
