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
Miscellaneous filehashes
"""

import sqlite3
import sys
from typing import Iterator


def filehashes(file=sys.stdin) -> Iterator[str]:
    """
    Yields valid filehashes from stdin.
    """
    # TODO: throw on invalid input or warn on invalid input
    # TODO: work on sys.argv
    for line in file:
        filehash = line.strip()
        if filehash:
            yield filehash


def create_query_table(conn: sqlite3.Connection, hashes: Iterator[str]=None) -> None:
    """
    Create a temporary table called `query_hash` that constists of a list of
    file hashes. One can then use a NATURAL JOIN to fetch rows matching the
    file hashes provided.
    """

    conn.execute('''
        CREATE TEMPORARY TABLE query_hash(hash PRIMARY KEY)
    ''')
    if hashes is None:
        hashes = filehashes()
    conn.executemany('''
        INSERT INTO query_hash(hash) VALUES (?)
    ''', ((fh,) for fh in hashes))
