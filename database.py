#!/usr/bin/env python
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
>>> db = Database()
>>> rev = '1b8f23c763d08130ec2081c35e7f9fe0d392d700'
>>> repo = Repository.create('github', 'example', 'mit', rev)
>>> ret = db.add_repository(repo)
>>> ret == repo
True
>>> source_a = SourceFile.create(repo, b'void 0;', 'index.js')
>>> ret = db.add_source_file(source_a)
>>> ret == source_a
True
>>> source_b = SourceFile.create(repo, b'void 0;', 'undefined.js')
>>> source_a != source_b
True
>>> ret = db.add_source_file(source_b)
Traceback (most recent call last):
...
database.DuplicateFileError: duplicate file contents
>>> parsed = ParsedSource(source_a.hash, [], {})
>>> ret = db.add_parsed_source(parsed)
>>> parsed == ret
True
>>> db.set_failure(source_b.hash)

>>> db.get_source(source_a.hash)
b'void 0;'
"""

import logging
import sqlite3
from contextlib import closing

from path import Path

from datatypes import Repository, SourceFile, ParsedSource
from utils import is_hash


logger = logging.getLogger(__name__)

SCHEMA_FILENAME = Path(__file__).parent / 'schema.sql'
with open(SCHEMA_FILENAME, encoding='UTF-8') as schema_file:
    SCHEMA = schema_file.read()
    del schema_file


class DuplicateFileError(Exception):
    def __init__(self, hash_):
        assert is_hash(hash_)
        self.hash = hash_
        super(DuplicateFileError, self).__init__("duplicate file contents")


class SourceNotFoundError(Exception):
    def __init__(self, hash_):
        assert is_hash(hash_)
        self.hash = hash_
        super(SourceNotFoundError,
              self).__init__("could not find source", hash_)


class Database:
    def __init__(self, connection=None):
        if connection is None:
            logger.warn("Using in memory database!")
            self.conn = sqlite3.connect(':memory:')
        else:
            self.conn = connection

        self._set_wal()
        self._initialize_db()

    def _initialize_db(self):
        conn = self.conn
        if self._is_database_empty():
            with conn:
                conn.executescript(SCHEMA)

    def _set_wal(self):
        with closing(self.conn.cursor()) as cur:
            cur.execute('PRAGMA journal_mode=WAL')
            status, = cur.fetchone()
        assert status in ('wal', 'memory')

    def _is_database_empty(self):
        with closing(self.conn.cursor()) as cur:
            cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            answer, = cur.fetchone()

        return int(answer) == 0

    def add_repository(self, repo):
        assert isinstance(repo, Repository)

        with closing(self.conn.cursor()) as cur, self.conn:
            cur.execute(r"""
                INSERT INTO repository (owner, repo, license, revision)
                VALUES (?, ?, ?, ?);
            """, (repo.owner, repo.name, repo.license, repo.revision))

        return repo

    def add_source_file(self, source_file):
        assert isinstance(source_file, SourceFile)

        try:
            with closing(self.conn.cursor()) as cur, self.conn:
                cur.execute(r"""
                    INSERT INTO source_file (hash, owner, repo, path, source)
                    VALUES (?, ?, ?, ?, ?);
                """, (source_file.hash, source_file.owner, source_file.name,
                      source_file.path, source_file.source))
        except sqlite3.IntegrityError:
            raise DuplicateFileError(source_file.hash)

        return source_file

    def get_source(self, hash_):
        assert is_hash(hash_)

        with closing(self.conn.cursor()) as cur:
            cur.execute('SELECT source FROM source_file WHERE hash = ?', (hash_,))
            result = cur.fetchone()

        if result is None:
            raise SourceNotFoundError(hash_)

        source, = result
        return source.encode('utf-8') if isinstance(source, str) else source

    def add_parsed_source(self, parsed_source):
        assert isinstance(parsed_source, ParsedSource)
        with self.conn:
            self.conn.execute(r"""
                INSERT INTO parsed_source (hash, ast, tokens)
                VALUES (?, ?, ?)
            """, (parsed_source.hash, parsed_source.ast_as_json,
                  parsed_source.tokens_as_json))
        return parsed_source

    def set_failure(self, source_hash):
        assert is_hash(source_hash)
        with self.conn:
            self.conn.execute(r"""INSERT INTO failure (hash) VALUES (?)""",
                              (source_hash,))
