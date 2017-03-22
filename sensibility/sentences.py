#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2017 Eddie Antonio Santos <easantos@ualberta.ca>
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
Yields contexts in both forwards and backwards directions.
"""

from itertools import chain, repeat
from typing import Sequence, TypeVar, Iterable, Tuple

from .vocabulary import vocabulary


# Types
T = TypeVar('T')
Sentence = Tuple[Sequence[T], T]


def forward_sentences(
        vector: Sequence[T],
        context: int=None,
        adjacent: int=1,
        sentence: int=20
) -> Iterable[Sentence]:
    """
    Yield "sentences" which consist of a context, and the token immediately to
    the RIGHT of the context (c.f., backward_sentences()).
    """
    from .abram import at_least
    if context is None:
        context = sentence - adjacent

    padding_token = vocabulary.start_token_index

    # Generate a sentence for each element in the vector.
    for i, element in enumerate(vector):
        # Ensure the beginning of the slice is AT LEAST 0 (or else the slice
        # will start from THE END of the vector!)
        beginning = at_least(0, i - context)
        real_context = vector[beginning:i]
        # Need to add padding when i is less than the context size.
        if i < context:
            padding = repeat(padding_token, context - i)
            yield tuple(chain(padding, real_context)), element
        else:
            # All tokens come from the vector
            yield tuple(real_context), element


def backward_sentences(
        vector: Sequence[T],
        context: int=None,
        adjacent: int=1,
        sentence: int=20
) -> Iterable[Sentence]:
    """
    Yield "sentences" which consist of a context, and the token immediately to
    the LEFT of the context (c.f., forward_sentences()).
    """
    if context is None:
        context = sentence - adjacent

    padding_token = vocabulary.end_token_index
    vector_length = len(vector)

    # Generate a sentence for each element in the vector.
    for c_start, element in enumerate(vector, start=1):
        c_end = c_start + context
        real_context = vector[c_start:c_end]
        # Must add padding when the context goes over the size of the vector.
        if c_end >= vector_length:
            padding = repeat(padding_token, c_end - vector_length)
            yield tuple(chain(real_context, padding)), element
        else:
            # All tokens come from the vector
            yield tuple(real_context), element
