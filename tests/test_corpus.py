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
Tests the Corpus class using a test.sql.
"""

import sqlite3
from pathlib import Path

from sensibility.corpus import Corpus

_DIRECTORY = Path(__file__).parent


def basic_test():
    """
    Tests opening and iterating the sample corpus.
    """
    corpus = Corpus(new_connection_for_testing())
    assert len(corpus) == len(list(corpus))
    hashes = next(iter(corpus))
    assert all(isinstance(x, str) for x in hashes)


def test_iterate():
    corpus = Corpus(new_connection_for_testing())
    files = tuple(corpus)
    assert len(files) == 2
    test_file_hash = '86cc829b0a086a9f655b942278f6be5c9e5057c34459dafafa312dfdfa3a27d0'
    assert next((corpus.iterate())) == test_file_hash


def test_get_source():
    corpus = Corpus(new_connection_for_testing())
    file_hash = next(iter(corpus))
    source = corpus.get_source(file_hash)
    assert isinstance(source, bytes)
    assert source == b'(name) => console.log(`Hello, ${name}!`);'


def new_connection_for_testing():
    """
    Return an SQLite3 connection suitable for testing.
    """
    conn = sqlite3.connect(':memory:')
    with open(str(_DIRECTORY / 'test.sql')) as sqlfile:
        conn.executescript(sqlfile.read())
    return conn
