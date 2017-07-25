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
    # assert ins.type is Insertion
    assert ins.new_token == index_of('}')
    assert ins.position == 3

    delt = determine_edit(b'class Hello {{}',   b'class Hello {}')
    # assert delt.type is Deletion
    assert delt.new_token is None
    assert delt.position in {2, 3}  # Can be either curly brace

    sub = determine_edit(b'goto label;', b'int label;')
    # assert sub.type is Substitution
    assert sub.position == 0
    assert sub.new_token == index_of('int')


def index_of(token: str) -> Vind:
    """
    Given a token in the vocabulary, returns its vocabulary index.
    """
    return language.vocabulary.to_index(token)
