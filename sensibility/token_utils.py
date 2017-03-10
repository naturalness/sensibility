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

from collections import namedtuple

__all__ = [
    'Token',
    'Location',
    'Position'
]


class Token(namedtuple('BaseToken', 'value type loc')):
    """
    Represents one (language agnostic!) token from a file.
    """

    @classmethod
    def from_json(cls, obj):
        """
        Converts the Esprima JSON token into a token WITHOUT location
        information! Caches tokens to keep memoy usage down.
        """
        return Token(value=obj['value'],
                     type=obj['type'],
                     loc=Location.from_json(obj['loc']))

    @property
    def line(self):
        return self.loc.start.line

    @property
    def column(self):
        return self.loc.start.column

    def __str__(self):
        return self.value

    @property
    def is_on_single_line(self):
        """
        True if the token spans multiple lines.
        """
        return self.location.start.line == self.location.end.line


class Location(namedtuple('BaseLocation', 'start end')):
    """
    Represents the exact location of a token in a file.
    """
    def __new__(cls, start=None, end=None):
        assert isinstance(start, Position)
        assert isinstance(end, Position)
        return super().__new__(cls, start, end)

    @classmethod
    def from_json(cls, obj):
        return cls(start=Position.from_json(obj['start']),
                   end=Position.from_json(obj['end']))


class Position(namedtuple('BasePosition', 'line column')):
    """
    A line/column position in a text file.
    """
    def __new__(cls, line=None, column=None):
        assert isinstance(line, int) and line >= 1
        assert isinstance(column, int) and column >= 0
        return super().__new__(cls, line, column)

    @classmethod
    def from_json(cls, obj):
        return cls(line=obj['line'], column=obj['column'])
