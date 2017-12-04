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

from blessings import Terminal  # type: ignore
from pathlib import Path

from sensibility.language import current_language
from sensibility.edit import Edit, Insertion, Deletion, Substitution


class Suggestion:
    """
    Wraps an edit as a suggestion to a fix.
    """

    line: int
    # This is always 1-indexed!
    column: int

    @staticmethod
    def enclose(filename: Path, fix: Edit) -> 'Suggestion':
        tokens = tuple(current_language.tokenize(filename.read_bytes()))
        return {
            Insertion: not_implemented,
            Deletion: lambda: Remove(fix.index, tokens),
            Substitution: not_implemented,
        }[type(fix)]()

    def __str__(self) -> str:
        raise NotImplementedError('The subclass MUST implement this')


class Remove(Suggestion):
    def __init__(self, pos, tokens):
        self.pos = pos
        self.tokens = tokens

    @property
    def token(self):
        return self.tokens[self.pos]

    @property
    def line(self):
        return self.token.line

    @property
    def column(self):
        return self.token.column

    def __str__(self):
        t = Terminal()
        text = self.token.value

        msg = ("try removing '{t.bold}{text}{t.normal}' "
               "".format_map(locals()))
        line_tokens = get_token_line(self.pos, self.tokens)
        line = format_line(line_tokens)
        padding = ' ' * (1 + self.token.column)
        arrow = padding + t.bold_red('^')
        suggestion = padding + t.red(text)

        return '\n'.join((msg, line, arrow, suggestion))


class Insert(Suggestion):
    def __init__(self, token, pos, tokens):
        self.token = token
        assert 1 < pos < len(tokens)
        self.tokens = tokens

        # Determine if it should be an insert after or an insert before.
        # This depends on whether the token straddles a line.
        if tokens[pos - 1].line < tokens[pos].line:
            self.insert_after = True
            self.pos = pos - 1
        else:
            self.insert_after = False
            self.pos = pos

    @property
    def line(self):
        return self.tokens[self.pos].line

    @property
    def column(self):
        return self.tokens[self.pos].column

    @property
    def insert_before(self):
        return not self.insert_after

    def __str__(self):
        t = Terminal()

        pos = self.pos
        text = self.token.value

        # TODO: lack of bounds check...
        next_token = self.tokens[pos + 1]
        msg = ("try inserting '{t.bold}{text}{t.normal}' "
               "".format_map(locals()))

        line_tokens = get_token_line(self.pos, self.tokens)

        if self.insert_after:
            line = format_line(line_tokens)
            # Add an extra space BEFORE the insertion point:
            padding = ' ' * (2 + 1 + self.column)
        else:
            # Add an extra space AFTER insertion point;
            line = format_line(line_tokens,
                               insert_space_before=self.tokens[self.pos])
            padding = ' ' * (1 + self.column)

        arrow = padding + t.bold_green('^')
        suggestion = padding + t.green(text)

        return '\n'.join((msg, line, arrow, suggestion))


def format_fix(filename: Path, fix: Edit) -> None:
    """
    Prints a fix for the given filename.
    """
    suggestion = Suggestion.enclose(filename, fix)
    line = suggestion.line
    column = suggestion.column
    t = Terminal()
    header = t.bold(f"{filename}:{line}:{column}:")
    print(header, suggestion)


def get_token_line(pos, tokens):
    line_no = tokens[pos].line

    left_extent = pos
    while left_extent > 0:
        if tokens[left_extent - 1].line != line_no:
            break
        left_extent -= 1

    right_extent = pos + 1
    while right_extent < len(tokens):
        if tokens[right_extent].line != line_no:
            break
        right_extent += 1

    return tokens[left_extent:right_extent]


def format_line(tokens, insert_space_before=None):
    result = ''
    extra_padding = 0
    for token in tokens:
        if token is insert_space_before:
            extra_padding = 2

        padding = ' ' * (extra_padding + token.column + 1 - len(result))
        result += padding
        result += token.value
    return result


def not_implemented():
    raise NotImplementedError()
