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


import sys
import random
from itertools import zip_longest
from typing import IO, Iterable, Iterator, Sequence, Sized, TypeVar, Any, List

from .vocabulary import vocabulary, Vind
from .vectorize_tokens import SourceVector

"""
TODO:
class SourceFile(source):
    sources: Corpus
    vectors: Vectors

    @property
    def vectors(self) -> Program: ...

    @property
    def source(self) -> Sequence[SourceToken]: ...

    def line_of_token(self, index: int) -> int: ...
"""

# TODO: make this a Sequence[Vind]? Maybe not...
class Program(Sized, Iterable[Vind]):
    """
    A source code program, with a file hash, and a token stream.
    """

    def __init__(self, filehash: str, tokens: Sequence[Vind]) -> None:
        assert len(tokens) > 0
        self.tokens = tokens
        # TODO: factor out filehash?
        self.filehash = filehash

    def __eq__(self, other: Any) -> bool:
        """
        True when both programs are token for token equivalent.

        >>> a = Program('<a>', [23, 48, 70])
        >>> b = Program('<b>', [23, 48, 70])
        >>> a == b
        True
        >>> c = Program('<a>', [23, 48])
        >>> a == c
        False
        """
        if isinstance(other, Program):
            return all(a == b for a, b in zip_longest(self, other))
        else:
            return False

    def __iter__(self) -> Iterator[Vind]:
        return iter(self.tokens)

    def __getitem__(self, index: int) -> Vind:
        return self.tokens[index]

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        tokens = ', '.join(repr(token) for token in self)
        return f"{clsname}({self.filehash!r}, [{tokens}])"

    def print(self, file: IO[str]=sys.stdout) -> None:
        """
        Prints the tokens to a file, using real tokens.
        """
        for token in self:
            print(vocabulary.to_text(token), file=file, end=' ')
        # Print a final newline.
        print(file=file)

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

    def with_substitution(self, index: int, token: Vind) -> 'Program':
        """
        Return a new program, swapping out the token at index with the given
        token.
        """
        sequence: List[Vind] = []
        sequence.extend(self.tokens[:index])
        sequence.append(token)
        sequence.extend(self.tokens[index + 1:])
        return Program(self.filehash, sequence)

    def with_token_removed(self, index: int) -> 'Program':
        """
        Return a new program with the token at the given index removed.
        """
        sequence: List[Vind] = []
        sequence.extend(self.tokens[:index])
        sequence.extend(self.tokens[index + 1:])
        return Program(self.filehash, sequence)

    def with_token_inserted(self, index: int, token: Vind) -> 'Program':
        """
        Return a new program with the token at the given index removed.
        """
        assert 0 <= index <= len(self)
        sequence: List[Vind] = []
        sequence.extend(self.tokens[:index])
        sequence.append(token)
        sequence.extend(self.tokens[index:])
        return Program(self.filehash, sequence)


# TODO: O(1) applying edits

T = TypeVar('T')


class ReadOnlySlice(Sequence[T]):
    """
    A read-only slice from a sequence (say a list) without copying the entire
    list.

    Derived From: http://stackoverflow.com/a/3485490/6626414
    Originally written by: Alex Martelli
        <https://stackoverflow.com/users/95810/alex-martelli>
    """

    def __init__(self, alist: Sequence[T], start: int, alen: int) -> None:
        self.alist = alist
        self.start = start
        self.alen = alen

    def __len__(self) -> int:
        return self.alen

    def adj(self, i: int) -> int:
        if i < 0:
            i += self.alen
        return i + self.start

    def __getitem__(self, i):
        return self.alist[self.adj(i)]
