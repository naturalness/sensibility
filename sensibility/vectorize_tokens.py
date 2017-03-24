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
from typing import Sized, Sequence, Iterator, cast

from .token_utils import Token
from .vocabulary import vocabulary, Vind, START_TOKEN, END_TOKEN
from .stringify_token import stringify_token


# TODO: Combine this with .source_vector...
class SourceVector(Sequence[Vind]):
    """
    A sequence of vocabulary indices with MAXIMUM STORAGE EFFICENCY.
    """
    __slots__ = ('_array',)

    def __init__(self, token_sequence: Sequence[int]) -> None:
        self._array = array.array('B', token_sequence)

    def __iter__(self) -> Iterator[Vind]:
        return iter(cast(Iterator[Vind], self._array))

    def __getitem__(self, key):
        return self._array[key]

    def __len__(self) -> int:
        return len(cast(Sized, self._array))

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}([{', '.join(str(x) for x in self)}])"

    def tobytes(self) -> bytes:
        return self._array.tobytes()


def generated_vector(tokens):
    for token in tokens:
        yield vocabulary.to_index(stringify_token(token))


def serialize_tokens(tokens: Sequence[Token]) -> SourceVector:
    """
    Return an (unsigned) byte array of tokens, useful for storage.

    >>> toks = [Token(value='var', type='Keyword', loc=None)]
    >>> serialize_tokens(toks)
    SourceVector([86])
    >>> serialize_tokens(toks).tobytes()
    b'V'
    """
    return SourceVector(generated_vector(tokens))


def deserialize_tokens(byte_string: bytes) -> SourceVector:
    """
    Return an array of vocabulary entries given a byte string produced by
    serialize_token().tobytes()

    >>> deserialize_tokens(b'VZD')
    SourceVector([86, 90, 68])
    """
    return SourceVector(byte_string)
