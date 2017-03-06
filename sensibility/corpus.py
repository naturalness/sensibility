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
Represents a read-only corpus of sources, repositories, and all kinds of
goodness.
"""

import sqlite3
from pathlib import Path
from typing import Iterable, Tuple, Sized, Union


__all__ = ['Corpus']


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
        Iterate through all usable sources in the file.
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

    def filenames_from_project(self, project: str):
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
        Return the source code of the given file.
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
    def connect_to(cls, filename: Union[str, Path]) -> 'Corpus':
        """
        Connect to the database (read-only) with the given filename.
        """
        path = Path(filename).resolve()
        assert path.exists(), '%r does not exist' % (filename,)
        uri = 'file:{}?mode=ro'.format(filename)
        conn = sqlite3.connect(uri, uri=True)  # type: ignore
        return Corpus(conn)
