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
"""

import sys
import heapq
import random
from pathlib import Path
from functools import partial
from itertools import islice

from tqdm import tqdm

from condensed_corpus import CondensedCorpus

error = partial(print, file=sys.stderr)

FOLDS = 10


def main():
    _, filename, max_iters = sys.argv
    filename = Path(filename)
    assert filename.exists()
    max_iters = int(max_iters)

    corpus = CondensedCorpus.connect_to(str(filename))

    # We maintain a priority queue of heaps. At the top of the heap is the
    # one with the fewest tokens.
    heap = [(0, fold_no) for fold_no in range(FOLDS)]
    heapq.heapify(heap)

    # This is kinda dumb, but:
    # Iterate through a shuffled list of rowids...
    error('Shuffling...')
    shuffled_ids = list(range(corpus.min_index, corpus.max_index + 1))
    random.shuffle(shuffled_ids)

    if max_iters < 1:
        print("Using full corpus")
        max_iters = len(shuffled_ids)
    iterations = min(max_iters, len(shuffled_ids))

    def pop():
        return heapq.heappop(heap)

    def push(n_tokens, fold_no):
        assert isinstance(file_hash, str)
        assert isinstance(fold_no, int)
        return heapq.heappush(heap, (n_tokens, fold_no))


    progress = tqdm(islice(shuffled_ids, iterations), total=iterations)
    for file_id in progress:
        try:
            file_hash, tokens = corpus[file_id]
        except:
            error("file not found in corpus:", file_id)
            continue
        else:
            n_tokens = len(tokens)

        # Assign to the min fold
        tokens_in_fold, fold_no = pop()

        corpus.add_to_fold(file_hash, fold_no)
        push(tokens_in_fold + n_tokens, fold_no)
        progress.set_description(str(file_id))


if __name__ == '__main__':
    main()
