#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from vocabulary import vocabulary, START_TOKEN, END_TOKEN
from stringify_token import stringify_token

import numpy as np

def vectorize_tokens(tokens):
    """
    Turns a file into a vector of indices (not a one-hot vector per token!).
    Automatically inserts start and end tokens.

    >>> from corpus import Token
    >>> vectorize_tokens([Token(value='var', type='Keyword', loc=None)])
    (0, 86, 99)
    """
    def generate():
        yield vocabulary.to_index(START_TOKEN)
        for token in tokens:
            yield vocabulary.to_index(stringify_token(token))
        yield vocabulary.to_index(END_TOKEN)

    return tuple(generate())

def create_one_hot_vector(index):
    """
    >>> v = create_one_hot_vector(1)
    >>> len(v) == len(vocabulary)
    True
    >>> v[1]
    1
    >>> all(v[i] == 0 for i in v if i != 1)
    True
    """
    return tuple(1 if i == index else 0 for i in range(len(vocabulary)))

def matrixify_tokens(tokens):
    """
    File's raw tokens -> one-hot encoded matrix that represents the file.

    >>> from corpus import Token
    >>> matrix = matrixify_tokens([Token(value='var', type='Keyword', loc=None)])
    >>> matrix.shape == (3, len(vocabulary))
    True
    >>> matrix.sum()
    3
    >>> matrix[0, 0]
    1
    >>> matrix[2, len(vocabulary) -1]
    1
    """

    # x dimension is stream; y is length of vocabulary
    dimensions = (2 + len(tokens), len(vocabulary))
    matrix = np.zeros(dimensions, dtype=np.bool)

    for t, y in enumerate(vectorize_tokens(tokens)):
        matrix[t, y] = 1

    return matrix
