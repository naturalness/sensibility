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
from typing import Iterable, Sequence, Tuple, TypeVar, Union, overload

from sensibility import current_language
from sensibility.abram import at_least

# Types
T = TypeVar('T')
Sentence = Tuple[Sequence[T], T]


class Sentences(Sequence[Sentence]):
    """
    Turn a sequence of tokens into a sequence of sentences.
    DO NOT INSTANTIATE THIS CLASS DIRECTLY: instead use the .forwards_from()
    or .backwards_from() static methods to instantiate a subclass.
    """

    def __init__(self, seq: Sequence[T], context_length: int) -> None:
        self.seq = seq
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.seq)

    @overload
    def __getitem__(self, index: int) -> Sentence:
        pass

    @overload
    def __getitem__(self, index: slice) -> Sequence[Sentence]:
        pass

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            raise NotImplementedError
        # Conver a negative index into a positive one.
        if index < 0:
            index = len(self) + index
        if 0 <= index < len(self):
            return self.make_sentence(index)
        else:
            raise IndexError

    def make_sentence(self, index: int) -> Sentence:
        """
        Returns a single sentence at the given index.
        """
        raise NotImplementedError

    @staticmethod
    def forwards_from(seq: Sequence[T], context_length: int) -> 'ForwardSentences':
        return ForwardSentences(seq, context_length)

    @staticmethod
    def backwards_from(seq: Sequence[T], context_length: int) -> 'BackwardSentences':
        return BackwardSentences(seq, context_length)


class ForwardSentences(Sentences):
    """
    Addresses the prefix and the adjacent token from a token stream.
    """

    def make_sentence(self, index: int) -> Sentence:
        vector = self.seq
        assert 0 <= index < len(vector)
        context = self.context_length
        padding_token = current_language.vocabulary.start_token_index
        element = vector[index]
        # Ensure the beginning of the slice is AT LEAST 0 (or else the slice
        # will start from THE END of the vector!)
        beginning = at_least(0, index - context)
        real_context = vector[beginning:index]
        # Need to add padding when index is less than the context size.
        if index < context:
            padding = repeat(padding_token, context - index)
            return tuple(chain(padding, real_context)), element
        else:
            # All tokens come from the vector
            return tuple(real_context), element


class BackwardSentences(Sentences):
    """
    Addresses the suffix and the adjacent token from a token stream.
    """

    def make_sentence(self, index: int) -> Sentence:
        vector = self.seq
        assert 0 <= index < len(vector)
        padding_token = current_language.vocabulary.end_token_index
        context = self.context_length

        element = vector[index]
        c_start = index + 1
        c_end = c_start + context

        real_context = vector[c_start:c_end]
        # Must add padding when the context goes over the size of the vector.
        if c_end > len(vector):
            padding = repeat(padding_token, c_end - len(vector))
            return tuple(chain(real_context, padding)), element
        else:
            # All tokens come from the vector
            return tuple(real_context), element


def forward_sentences(vector: Sequence[T], context: int) -> Iterable[Sentence]:
    """
    Yield "sentences" which consist of a prefix, and the token immediately to
    the RIGHT of the context (c.f., backward_sentences()).
    """
    # Delegate from class-based implementation
    return iter(Sentences.forwards_from(vector, context))


def backward_sentences(vector: Sequence[T], context: int) -> Iterable[Sentence]:
    """
    Yield "sentences" which consist of a suffix, and the token immediately to
    the LEFT of the context (c.f., forward_sentences()).
    """
    # Delegate from class-based implementation
    return iter(Sentences.backwards_from(vector, context))
