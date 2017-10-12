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
Tests mistakes and edit distance.
"""

from functools import lru_cache

import pytest

from sensibility.evaluation.mistakes import Mistakes
from sensibility.evaluation.distance import (
    tokenwise_distance, determine_edit, determine_fix_event
)
from sensibility.language import language
from sensibility.vocabulary import Vind
from sensibility import Insertion, Substitution, Deletion


def setup_module():
    language.set('java')


def test_general() -> None:
    assert 1 == tokenwise_distance(b'class Hello {',    b'class Hello {}')
    assert 1 == tokenwise_distance(b'class Hello {}',   b'class Hello }')
    assert 1 == tokenwise_distance(b'enum Hello {}',    b'class Hello {}')
    assert 0 == tokenwise_distance(b'class Hello {}',   b'class Hello {}')

    assert 2 >= tokenwise_distance(b'enum Hello {}',    b'class Hello {')


def test_extra() -> None:
    # Regression: Lexer should be able to handle const and goto keywords,
    # even though Java does not use them.
    # https://docs.oracle.com/javase/tutorial/java/nutsandbolts/_keywords.html
    assert 1 == tokenwise_distance(b'const int hello;', b'final int hello;')
    assert 1 == tokenwise_distance(b'goto label;', b'int label;')

    # Regression: Distances should still be calculated if items are OOV
    # ERROR and UNDERSCORE are out-of-vocabulary as well.
    # In hindsight, const and goto should be OOV as well... :/
    assert 1 == tokenwise_distance(b'int #label;', b'int label;')
    assert 1 == tokenwise_distance(b'int _;', b'int label;')
    edit = determine_edit(b'int #label;', b'int label;')
    if isinstance(edit, Deletion):
        assert edit.original_token == language.vocabulary.unk_token_index
    else:
        pytest.fail(f'Wrong edit: {edit!r}')


@pytest.mark.skip  # Does an unnecessary database access.
def test_get_source() -> None:
    import sqlite3
    from sensibility._paths import MISTAKE_FILE
    m = Mistakes(sqlite3.connect(str(MISTAKE_FILE)))
    mistake = next(iter(m))
    assert 0 < tokenwise_distance(mistake.before, mistake.after)


def test_get_edit(c) -> None:
    ins = determine_edit(b'class Hello {',    b'class Hello {}')
    if isinstance(ins, Insertion):
        assert ins.token == index_of(c('}'))
        assert ins.index == 3
    else:
        pytest.fail(f'Wrong edit: {ins!r}')

    delt = determine_edit(b'class Hello {{}',   b'class Hello {}')
    if isinstance(delt, Deletion):
        assert delt.original_token == index_of(c('{'))
        assert delt.index in {2, 3}  # Can be either curly brace
    else:
        pytest.fail(f'Wrong edit: {delt!r}')

    sub = determine_edit(b'goto label;', b'int label;')
    if isinstance(sub, Substitution):
        assert sub.token == index_of(c('int'))
        assert sub.original_token == index_of(c('goto'))
        assert sub.index == 0
    else:
        pytest.fail(f'Wrong edit: {sub!r}')


def test_edit_line(c) -> None:
    head = [
        'class Hello {',
        'public static void main(String args[]) {'
    ]
    tail = [
        'System.error.println("Not enough args!");',
        'System.exit(1);',
        '}',
        'System.out.println("Hello, World!");',
        '}',
        '}'
    ]

    # Glue together a source file from head, tail, and the provided line.
    def to_source(line: str) -> bytes:
        return '\n'.join(head + [line] + tail).encode('UTF-8')

    before = to_source('if (args.length < 3)')
    after = to_source('if (args.length < 3) {')

    # The line the error happens on the line AFTER head (one-indexed).
    error_line = len(head) + 1

    # Sanity check: before should be invalid; after should be valid.
    assert not language.check_syntax(before)
    assert language.check_syntax(after)

    fix_event = determine_fix_event(before, after)
    assert fix_event.fix == Insertion(22, index_of(c('{')))
    assert fix_event.line_no == error_line
    assert fix_event.new_token == c('{')
    assert fix_event.old_token is None


def index_of(token: str) -> Vind:
    """
    Given a token in the vocabulary, returns its vocabulary index.
    """
    return language.vocabulary.to_index(token)
