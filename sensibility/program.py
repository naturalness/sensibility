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


import random
from typing import IO, Iterable, Iterator, Sequence, Sized, TypeVar

from .vocabulary import vocabulary, Vind
from .vectorize_tokens import SourceVector


T = TypeVar('T')

# reveal_type(Sequence[T].__getitem__)
# Revealed type is Overload(
#   def [_T_co] (typing.Sequence[_T_co`1], builtins.int) -> _T_co`1,
#   def [_T_co] (typing.Sequence[_T_co`1], builtins.slice) ->
#       typing.Sequence[_T_co`1]
# )


class Program(Sized, Iterable[Vind]):
    """
    A source code program, with a file hash, and a token stream.
    """

    def __init__(self, filehash: str, tokens: Sequence[Vind]) -> None:
        assert len(tokens) > 0
        self.tokens = tokens
        self.filehash = filehash

    def __iter__(self) -> Iterator[Vind]:
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def print(self, file: IO[str]) -> None:
        """
        Prints the tokens to a file, using real tokens.
        """
        raise NotImplementedError
        # TODO: O(n) printing source code.

    def random_token_index(self) -> int:
        """
        Produces a random insertion point in the program. Does not include
        start and end tokens.
        """
        return random.randrange(0, len(self))

    def random_insertion_point(self) -> int:
        """
        Produces a random insertion point in the program. Does not include
        start and end tokens.
        """
        return random.randint(0, len(self))

# TODO: O(1) applying edits


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
