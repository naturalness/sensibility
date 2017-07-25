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

import sqlite3

import pytest

from sensibility.evaluation.mistakes import Mistakes
from sensibility.evaluation.distance import tokenwise_distance, determine_edit
from sensibility.language import language
from sensibility.vocabulary import Vind
from sensibility import Insertion, Substitution, Deletion


def setup_module():
    language.set_language('java')


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


@pytest.mark.skip  # Does an unnecessary database access.
def test_get_source() -> None:
    m = Mistakes(sqlite3.connect('java-mistakes.sqlite3'))
    mistake = next(iter(m))
    assert 0 < tokenwise_distance(mistake.before, mistake.after)


def test_get_edit() -> None:
    ins = determine_edit(b'class Hello {',    b'class Hello {}')
    if isinstance(ins, Insertion):
        assert ins.token == index_of('}')
        assert ins.index == 3
    else:
        pytest.fail(f'Wrong edit: {ins!r}')

    delt = determine_edit(b'class Hello {{}',   b'class Hello {}')
    if isinstance(delt, Deletion):
        assert delt.original_token == index_of('{')
        assert delt.index in {2, 3}  # Can be either curly brace
    else:
        pytest.fail(f'Wrong edit: {delt!r}')

    sub = determine_edit(b'goto label;', b'int label;')
    if isinstance(sub, Substitution):
        assert sub.token == index_of('int')
        assert sub.original_token == index_of('goto')
        assert sub.index == 0
    else:
        pytest.fail(f'Wrong edit: {sub!r}')


def index_of(token: str) -> Vind:
    """
    Given a token in the vocabulary, returns its vocabulary index.
    """
    return language.vocabulary.to_index(token)
