#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

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
