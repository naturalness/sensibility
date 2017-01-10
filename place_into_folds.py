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
import sys

from pathlib import Path
from functools import partial
from itertools import islice

from tqdm import tqdm

from condensed_corpus import CondensedCorpus

error = partial(print, file=sys.stderr)

parser = argparse.ArgumentParser('Divides the corpus into folds.')
parser.add_argument('filename', type=Path, metavar='corpus')
parser.add_argument('-k', '--folds', type=int, default=10, help='default: 10')
parser.add_argument('-n', '--min-tokens', type=int, default=None)
parser.add_argument('-f', '--overwrite', action='store_true')


def main():
    # Creates variables: folds, min_tokens, filename
    globals().update(vars(parser.parse_args()))
    assert filename.exists()
    assert folds >= 1

    corpus = CondensedCorpus.connect_to(str(filename))

    if corpus.has_fold_assignments:
        if overwrite:
            corpus.destroy_fold_assignments()
        else:
            error('Will not overwrite existing fold assignments!')
            exit(-1)

    # We maintain a priority queue of folds. At the top of the heap is the
    # fold with the fewest tokens.
    heap = [(0, fold_no) for fold_no in range(folds)]
    heapq.heapify(heap)

    # This is kinda dumb, but:
    # Iterate through a shuffled list of ALL rowids...
    error('Shuffling...')
    shuffled_ids = list(range(corpus.min_index, corpus.max_index + 1))
    random.shuffle(shuffled_ids)


    # A series of helper functions.
    def generate_files():
        for random_id in shuffled_ids:
            fewest_tokens, _ = heap[0]
            if fewest_tokens >= min_tokens and normalish():
                break

            try:
                yield corpus[random_id]
            except TypeError:
                error('File ID not found:', random_id, 'Skipping...')
                continue

    def normalish(getter=operator.itemgetter(0)):
        """
        Rule of thumb calculation for normality.
        """
        sample_mean = statistics.mean(map(getter, heap))
        stddev = statisitcs.stddev(map(getter, heap))

        progress.set_description("mean: {}, stddev: {}".format(
            sample_mean, stddev
        ))

        def t_statisitic(observation):
            return (observation - sample_mean) / stddev

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
        globals().update(min_tokens=math.inf)
        progress = tqdm(generate_files())
    else:
        progress = tqdm(generate_files(), total=min_tokens)

    progress.set_description('Assigning until minimum met')
    for file_hash, tokens in progress:
        n_tokens = len(tokens)

        # Assign it to the smallest fold
        tokens_in_fold, fold_no = pop()
        corpus.add_to_fold(file_hash, fold_no)
        push(tokens_in_fold + len(tokens), fold_no)

        new_smallest, _ = heap[0]
        progress.update(new_smallest - tokens_in_fold)


if __name__ == '__main__':
    main()
