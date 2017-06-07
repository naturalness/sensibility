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
import warnings
from typing import Sequence

from .lexical_analysis import Lexeme
from .source_vector import SourceVector
from .vocabulary import vocabulary
from .stringify_token import stringify_token


def generated_vector(tokens):
    for token in tokens:
        # XXX: HACK!
        try:
            yield vocabulary.to_index(stringify_token(token))
        except KeyError:
            warnings.warn(f'Casting unknown {token} to <UNK>')
            yield 0  # Unk


def serialize_tokens(tokens: Sequence[Lexeme]) -> SourceVector:
    """
    Return an (unsigned) byte array of tokens, useful for storage.

    >>> toks = [Lexeme(value='var', name='Keyword')]
    >>> serialize_tokens(toks)
    SourceVector([86])
    >>> serialize_tokens(toks).to_bytes()
    b'V'
    """
    return SourceVector(tuple(generated_vector(tokens)))
