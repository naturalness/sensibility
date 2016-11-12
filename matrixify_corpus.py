#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Creates corpus.npz from a full corpus stored as an SQLite database. 

The file contains one-hot encoded matrices.
"""

import numpy as np
from tqdm import tqdm

from corpus import Corpus
from vectorize_tokens import matrixify_tokens   

if __name__ == '__main__':
    import sys
    _, filename = sys.argv
    corpus = Corpus.connect_to(filename)
    iterator = tqdm((matrixify_tokens(tokens) for tokens in corpus),
                    total=len(corpus))
    np.savez_compressed('corpus', *iterator)
