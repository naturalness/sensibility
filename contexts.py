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

from itertools import chain, repeat, islice

from typing import Sequence, TypeVar, Iterable, Tuple

from vocabulary import vocabulary


# Types
T = TypeVar('T')
Sentence = Tuple[Sequence[T], T]


def forward_contexts(
        vector: Sequence[T],
        context: int=None,
        adjacent: int=1,
        sentence: int=20
) -> Iterable[Sentence]:
    """
    Yield "sentences" which consist of a context, and the token immediately to
    the RIGHT of the context.
    """
    if context is None:
        context = sentence - adjacent

    padding_token = vocabulary.start_token_index

    # Generate a sentence for each element in the vector.
    for i, element in enumerate(vector):
        real_context = islice(vector, max(0, i - context), i)
        # Need to add padding when i is less than the context size.
        if i < context:
            padding = repeat(padding_token, context - i)
            yield tuple(chain(padding, real_context)), element
        else:
            # All tokens come from the vector
            yield tuple(real_context), element


def backward_contexts():
    ...
