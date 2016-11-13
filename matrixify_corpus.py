#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Creates corpus.npz from a full corpus stored as an SQLite database.

The file contains one-hot encoded matrices.
"""

import gc

import numpy as np
from tqdm import tqdm

from corpus import Corpus
from vectorize_tokens import matrixify_tokens

if __name__ == '__main__':
    import sys
    _, filename = sys.argv
    corpus = Corpus.connect_to(filename)
    array = np.empty(len(corpus), dtype=np.object_)

    for i, tokens in enumerate(tqdm(corpus, total=len(corpus))):
        array[i] = matrixify_tokens(tokens)
        if i % 64 == 0:
            gc.collect(0)

    np.save('corpus', array)
