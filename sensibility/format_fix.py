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

from pathlib import Path

from blessings import Terminal  # type: ignore

from sensibility.edit import Deletion, Edit, Insertion, Substitution
from sensibility.language import current_language
from sensibility.vocabulary import Vind


class Suggestion:
    """
    Wraps an edit as a suggestion to a fix.
    """

    # Both line and column MUST be one-indexed
    # Note, that for some dumb reason, the convention is that column numbers
    # are stored zero-indexed, but line numbers are stored one-indexed, so
    # account for that...
    line: int
    column: int

    @staticmethod
    def enclose(filename: Path, fix: Edit) -> 'Suggestion':
        tokens = tuple(current_language.tokenize(filename.read_bytes()))
        if isinstance(fix, Insertion):
            return Insert(fix.token, fix.index, tokens)
        elif isinstance(fix, Deletion):
            return Remove(fix.index, tokens)
        elif isinstance(fix, Substitution):
            return Replace(fix, tokens)
        else:
            raise ValueError(f"Unknown edit subclass: {fix}")

    def __str__(self) -> str:
        raise NotImplementedError('The subclass MUST implement this')


# TODO: these classes are ancient; I could fix them.

class Insert(Suggestion):
    def __init__(self, token: Vind, pos, tokens) -> None:
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
        """
        ONE-INDEXED COLUMN
        """
        return 1 + self.tokens[self.pos].column

    @property
    def insert_before(self):
        return not self.insert_after

    def __str__(self):
        t = Terminal()

        pos = self.pos
        text = current_language.to_source_text(self.token)

        # TODO: lack of bounds check...
        next_token = self.tokens[pos + 1]
        msg = ("try inserting '{t.bold}{text}{t.normal}' "
               "".format_map(locals()))

        line_tokens = get_token_line(self.pos, self.tokens)

        if self.insert_after:
            line = format_line(line_tokens)
            # Add an extra space BEFORE the insertion point:
            padding = ' ' * (2 + self.column)
        else:
            # Add an extra space AFTER insertion point;
            line = format_line(line_tokens,
                               insert_space_before=self.tokens[self.pos])
            padding = ' ' * (self.column)

        arrow = padding + t.bold_green('^')
        suggestion = padding + t.green(text)

        return '\n'.join((msg, line, arrow, suggestion))


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
        """
        ONE-INDEXED COLUMN
        """
        return 1 + self.token.column

    def __str__(self):
        t = Terminal()
        text = self.token.value

        msg = ("try removing '{t.bold}{text}{t.normal}' "
               "".format_map(locals()))
        line_tokens = get_token_line(self.pos, self.tokens)
        line = format_line(line_tokens)
        padding = ' ' * (self.token.column)
        arrow = padding + t.bold_red('^')
        suggestion = padding + t.red(text)

        return '\n'.join((msg, line, arrow, suggestion))


class Replace(Suggestion):
    def __init__(self, fix: Substitution, tokens) -> None:
        self.fix = fix
        self.tokens = tokens

    @property
    def pos(self) -> int:
        return self.fix.index

    @property
    def token(self):
        return self.tokens[self.pos]

    @property
    def line(self):
        return self.token.line

    @property
    def column(self):
        """
        ONE-INDEXED COLUMN
        """
        return 1 + self.token.column

    def __str__(self) -> str:
        t = Terminal()
        original = self.token.value
        replacement = current_language.to_source_text(self.fix.token)

        msg = (
            f"try replacing {t.bold_red}{original}{t.normal}"
            f" with {t.bold_green}{replacement}{t.normal}"
        )

        line_tokens = get_token_line(self.pos, self.tokens)
        # TODO: add strikethrough to the token!
        line = format_line(line_tokens)
        padding = ' ' * (self.token.column)
        arrow = padding + t.bold_red('^')
        suggestion = padding + t.red(replacement)

        return '\n'.join((msg, line, arrow, suggestion))


def format_fix(filename: Path, fix: Edit) -> None:
    """
    Prints a fix for the given filename.
    """
    suggestion = Suggestion.enclose(filename, fix)
    line = suggestion.line
    column = suggestion.column
    t = Terminal()
    # Use a format similar to Clang's.
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

        padding = ' ' * (extra_padding + token.column - len(result))
        result += padding
        result += token._raw
    return result


def not_implemented():
    raise NotImplementedError()
