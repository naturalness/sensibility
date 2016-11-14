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
Creates token_lengths.npy from a full corpus stored as an SQLite database.

The file contains the token length counts as a numpy array.
"""

import sys
import numpy as np
from tqdm import tqdm

from corpus import Corpus


def main(len=len):
    _, filename = sys.argv
    corpus = Corpus.connect_to(filename)

    total = len(corpus)
    array = np.empty(total, dtype=np.uint32)

    MAX = 2**32 - 1

    for i, tokens in enumerate(tqdm(corpus, total=total)):
        n_tokens = len(tokens)
        assert n_tokens <= MAX
        array[i] = n_tokens
        del tokens

    np.save('token_lengths', array)

if __name__ == '__main__':
    main()
