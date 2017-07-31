#!/usr/bin/env python
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
Named `token_utils` because `token` is a Python standard library import, and
it breaks things if you shadow standard library stuff...
"""

from typing import Iterator

__all__ = [
    'Lexeme',
    'Token',
    'Location',
    'Position'
]


class Lexeme:
    """
    A lexeme is an abstract token.

    Lexeme pg. 111

    From "Compilers Principles, Techniques, & Tools, 2nd Ed." (WorldCat)
    by Aho, Lam, Sethi and Ullman:

    > A lexeme is a sequence of characters in the source program that matches
    > the pattern for a token and is identified by the lexical analyzer as an
    > instance of that token.
    """
    __slots__ = 'name', 'value'

    def __init__(self, *, name: str, value: str) -> None:
        self.name = name
        self.value = value
        # TODO: avoid storing value if name and value are equal?

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Lexeme(name={self.name!r}, value={self.value!r})"


class Position:
    """
    A line/column position in a text file.
    """
    __slots__ = 'line', 'column'

    def __init__(self, *, line: int, column: int) -> None:
        self.line = line
        self.column = column

    def __eq__(self, other) -> bool:
        return (isinstance(other, Position)
                and self.line == other.line
                and self.column == other.column)

    def __repr__(self) -> str:
        return f"Position(line={self.line!r}, column={self.column!r})"


class Location:
    """
    Represents the exact location of a token in a file.
    """
    __slots__ = 'start', 'end'

    def __init__(self, *, start: Position, end: Position) -> None:
        self.start = start
        self.end = end

    def __eq__(self, other) -> bool:
        return (isinstance(other, Location)
                and self.start == other.start
                and self.end == other.end)

    @property
    def spans_single_line(self) -> bool:
        """
        True if the token spans multiple lines.
        """
        return self.start.line == self.end.line

    def __repr__(self) -> str:
        return f"Location(start={self.start!r}, end={self.end!r})"

    @property
    def line(self) -> int:
        return self.start.line

    @classmethod
    def from_string(self, text: str, *, line: int, column: int) -> 'Location':
        r"""
        Determine the location from the size of the string.

        >>> Location.from_string("from", line=2, column=13)
        Location(start=Position(line=2, column=13), end=Position(line=2, column=16))
        >>> code = "'''hello,\nworld\n'''"
        >>> Location.from_string(code, line=14, column=6)
        Location(start=Position(line=14, column=6), end=Position(line=16, column=2))
        """
        start = Position(line=line, column=column)
        # How many lines are in the token?
        # NOTE: this may be different than what the tokenizer thinks is the
        # last line, due to handling of CR, LF, and CRLF, but I won't worry
        # too much about it.
        lines = text.split('\n')
        end_line = start.line + len(lines) - 1
        end_col = len(lines[-1]) - 1
        if len(lines) == 1:
            end_col += start.column
        end = Position(line=end_line, column=end_col)
        return Location(start=start, end=end)


class Token(Lexeme):
    """
    A lexeme with a location.  Includes location information.

    From "Compilers Principles, Techniques, & Tools, 2nd Ed." (WorldCat) by
    Aho, Lam, Sethi and Ullman:

    > A token is a pair consisting of a token name and an optional attribute
    > value.  The token name is an abstract symbol representing a kind of
    > lexical unit, e.g., a particular keyword, or sequence of input
    > characters denoting an identifier. The token names are the input symbols
    > that the parser processes.
    """
    __slots__ = 'start', 'end'

    def __init__(self, *, name: str, value: str, start: Position, end: Position) -> None:
        super().__init__(name=name, value=value)
        self.start = start
        self.end = end

    @property
    def column(self) -> int:
        """
        Column number of the beginning of the token.
        """
        return self.start.column

    @property
    def line(self) -> int:
        """
        Line number of the beginning of the token.
        """
        return self.start.line

    @property
    def lines(self) -> Iterator[int]:
        """
        An order list of all the lines in this file
        """
        yield from range(self.start.line, self.end.line + 1)

    @property
    def location(self) -> Location:
        """
        Location of the token.
        """
        return Location(start=self.start, end=self.end)

    @property
    def loc(self) -> Location:
        """
        Deprecated. Location of the token.
        """
        return self.location

    @property
    def spans_single_line(self) -> bool:
        """
        True if the token spans multiple lines.
        """
        return self.location.spans_single_line

    def __repr__(self) -> str:
        return (f"Token("
                f"name={self.name!r}, value={self.value!r}, "
                f"start={self.start!r}, end={self.end!r})")
