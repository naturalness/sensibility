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

import array
import warnings
from typing import NewType, Sequence, Iterator

from . import Token
from .vocabulary import vocabulary, START_TOKEN, END_TOKEN
from .stringify_token import stringify_token


# A SourceVector, which can be treated as a Sequence of vocabulary indices.
SourceVector = NewType('SourceVector', array.array)


def vectorize_tokens(tokens):
    """
    Turns a file into a vector of indices (not a one-hot vector per token!).
    Automatically inserts start and end tokens.

    >>> vectorize_tokens([Token(value='var', type='Keyword', loc=None)])
    (0, 86, 99)
    """
    warnings.warn('Use serialize_token() instead', DeprecationWarning)
    def generate():
        yield vocabulary.to_index(START_TOKEN)
        yield from generated_vector(tokens)
        yield vocabulary.to_index(END_TOKEN)

    return tuple(generate())


def generated_vector(tokens):
    for token in tokens:
        yield vocabulary.to_index(stringify_token(token))


def serialize_tokens(tokens: Sequence[Token]) -> SourceVector:
    """
    Return an (unsigned) byte array of tokens, useful for storing.

    >>> toks = [Token(value='var', type='Keyword', loc=None)]
    >>> serialize_tokens(toks)
    array('B', [86])
    >>> serialize_tokens(toks).tobytes()
    b'V'
    """
    return SourceVector(array.array('B', generated_vector(tokens)))


def deserialize_tokens(byte_string: bytes) -> SourceVector:
    """
    Return an array of vocabulary entries given a byte string produced by
    serialize_token().tobytes()

    >>> deserialize_tokens(b'VZD')
    array('B', [86, 90, 68])
    """
    return SourceVector(array.array('B', byte_string))
