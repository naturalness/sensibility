#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Turns a file into a vector of indices (not a one-hot vector per token!).
Automatically inserts start and end tokens.
"""

from vocabulary import vocabulary, START_TOKEN, END_TOKEN
from stringify_token import stringify_token


def vectorize_tokens(tokens):
    """
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
