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
A SourceVector is a sequence of Vind (vocabulary indices) that all allows for
mutations.
"""

import array
import random
import sys
from itertools import zip_longest
from typing import IO, Any, Iterable, Iterator, List, Sequence, cast

from .vocabulary import Vind


class SourceVector(Sequence[Vind]):
    """
    A sequence of vocabulary indices.
    """
    __slots__ = ('tokens',)

    def __init__(self, tokens: Iterable[Vind]) -> None:
        self.tokens = tuple(tokens)

    def __eq__(self, other: Any) -> bool:
        """
        True when both programs are token for token equivalent.

        >>> a = SourceVector([23, 48, 70])
        >>> b = SourceVector([23, 48, 70])
        >>> a == b
        True
        >>> c = SourceVector([23, 48])
        >>> a == c
        False
        """
        if isinstance(other, SourceVector):
            return all(a == b for a, b in zip_longest(self, other))
        else:
            return False

    def __iter__(self) -> Iterator[Vind]:
        return iter(self.tokens)

    # XXX: intentionally leave __getitem__ untyped, because it's annoying.
    def __getitem__(self, index):
        return self.tokens[index]

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}([{', '.join(str(x) for x in self)}])"

    def print(self, file: IO[str]=sys.stdout) -> None:
        """
        Prints the tokens to a file, using real tokens.
        """
        from sensibility.language import language
        for token in self:
            print(language.vocabulary.to_text(token), file=file, end=' ')
        # Print a final newline.
        print(file=file)

    def to_source_code(self) -> bytes:
        """
        Returns the source vector as bytes.
        """
        from sensibility.language import language
        to_text = language.vocabulary.to_source_text
        return ' '.join(to_text(token) for token in self).encode('UTF-8')

    def random_token_index(self) -> int:
        """
        Return the index of a random token in the file.
        """
        return random.randrange(0, len(self))

    def random_insertion_point(self) -> int:
        """
        Return a random insertion point in the program.  That is, an index in
        the program to insert BEFORE. This the imaginary token after the last
        token in the file.
        """
        return random.randint(0, len(self))

    def with_substitution(self, index: int, token: Vind) -> 'SourceVector':
        """
        Return a new program, swapping out the token at index with the given
        token.
        """
        # TODO: O(1) applying edits
        sequence: List[Vind] = []
        sequence.extend(self.tokens[:index])
        sequence.append(token)
        sequence.extend(self.tokens[index + 1:])
        return SourceVector(sequence)

    def with_token_removed(self, index: int) -> 'SourceVector':
        """
        Return a new program with the token at the given index removed.
        """
        assert len(self.tokens) > 0
        # TODO: O(1) applying edits
        assert 0 <= index < len(self)
        sequence: List[Vind] = []
        sequence.extend(self.tokens[:index])
        sequence.extend(self.tokens[index + 1:])
        return SourceVector(sequence)

    def with_token_inserted(self, index: int, token: Vind) -> 'SourceVector':
        """
        Return a new program with the token at the given index removed.
        """
        # TODO: O(1) applying edits
        assert 0 <= index <= len(self)
        sequence: List[Vind] = []
        sequence.extend(self.tokens[:index])
        sequence.append(token)
        sequence.extend(self.tokens[index:])
        return SourceVector(sequence)

    def to_array(self) -> array.array:
        """
        Convert to a dense array.array, suitable for compact serialization.
        """
        return array.array('B', self.tokens)

    def to_bytes(self) -> bytes:
        """
        Convert to bytes, for serialization.
        """
        return self.to_array().tobytes()

    @classmethod
    def from_bytes(self, byte_string: bytes):
        """
        Return an array of vocabulary entries given a byte string produced by
        serialize_token().tobytes()

        >>> SourceVector.from_bytes(b'VZD')
        SourceVector([86, 90, 68])
        """
        as_array = array.array('B', byte_string)
        return SourceVector(tuple(cast(Sequence[Vind], as_array)))


def to_source_vector(source: bytes, oov_to_unk: bool=False) -> SourceVector:
    from sensibility.language import current_language as language
    to_index = language.to_index_or_unk if oov_to_unk else language.to_index
    entries = language.vocabularize(source)
    return SourceVector(to_index(x) for x in entries)
