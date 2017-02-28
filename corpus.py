#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2016 Eddie Antonio Santos <easantos@ualberta.ca>
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

>>> corpus = Corpus(new_connection_for_testing())
>>> len(corpus) == len(list(corpus))
True
>>> hashes = next(iter(corpus))
>>> all(isinstance(x, str) for x in hashes)
True
"""

import sqlite3
import json
import logging
from pathlib import Path

from typing import Iterable, Tuple, Sized

from collections import namedtuple, OrderedDict


_DIRECTORY = Path(__file__).parent

# CREATE VIEW IF NOT EXISTS usable_source AS
USABLE_SOURCE_QUERY = r'''
SELECT hash
FROM source_file JOIN parsed_source USING (hash)
WHERE path NOT GLOB '*.min.js';
'''


class Corpus(Iterable[str], Sized):
    """
    Controls access to the BIG corpus of source code.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    def __iter__(self):
        return self.iterate()

    def iterate(self) -> Iterable[str]:
        """
        >>> corpus = Corpus(new_connection_for_testing())
        >>> files = tuple(corpus.iterate())
        >>> len(files)
        2

        >>> next((corpus.iterate()))
        '86cc829b0a086a9f655b942278f6be5c9e5057c34459dafafa312dfdfa3a27d0'
        """

        yield from (t[0] for t in self.conn.execute('''
            SELECT hash
              FROM source_file JOIN parsed_source USING (hash)
             WHERE path NOT GLOB '*.min.js';
        '''))

    @property
    def projects(self) -> Iterable[Tuple[str, str]]:
        """
        An iterator of projects with each of their file hashes.
        """
        cur = self.conn.cursor()
        cur.execute('''
            SELECT owner, repo FROM repository
        ''')
        yield from cur

    def filenames_from_project(self, project):
        """
        Yields (hash, filename) tuples that belong to the given project.
        """
        owner, repo = project
        cur = self.conn.cursor()
        cur.execute('''
            SELECT hash, path
              FROM source_file
             WHERE owner = :owner AND repo = :repo
        ''', dict(owner=owner, repo=repo))
        yield from cur

    def file_info(self, file_hash):
        results = self.conn.execute('''
            SELECT repo, owner, path
              FROM source_file
             WHERE hash = :hash
        ''', dict(hash=file_hash))
        ((repo, owner, path),) = results
        return repo, owner, path

    def get_tokens(self, file_hash):
        raise NotImplementedError

    def get_source(self, file_hash: str) -> bytes:
        """
        >>> corpus = Corpus(new_connection_for_testing())
        >>> file_hash = next(iter(corpus))
        >>> s = corpus.get_source(file_hash)
        >>> isinstance(s, bytes)
        True
        >>> s.decode('UTF-8')
        '(name) => console.log(`Hello, ${name}!`);'
        """
        source, = self.conn.execute('''
            SELECT source
              FROM source_file
             WHERE hash = :hash
        ''', dict(hash=file_hash)).fetchone()
        return source

    def __len__(self):
        """
        Return the amount of usable sources.
        """
        cur = self.conn.cursor()
        cur.execute(r'''
          SELECT COUNT(*)
            FROM source_file JOIN parsed_source USING (hash)
           WHERE path NOT GLOB '*.min.js'
        ''')
        count, = cur.fetchone()
        return int(count)

    @classmethod
    def connect_to(cls, filename: str) -> 'Corpus':
        """
        Connect to the database (read-only) with the given filename.
        """
        path = Path(filename).resolve()
        assert path.exists(), '%r does not exist' % (filename,)
        uri = 'file:{}?mode=ro'.format(filename)
        conn = sqlite3.connect(uri, uri=True)  # type: ignore
        return Corpus(conn)


def new_connection_for_testing():
    """
    Return an SQLite3 connection suitable for testing.
    """
    conn = sqlite3.connect(':memory:')
    with open(str(_DIRECTORY / 'test.sql')) as sqlfile:
        conn.executescript(sqlfile.read())
    return conn
