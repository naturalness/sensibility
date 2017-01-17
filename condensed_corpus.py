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


import io
import sqlite3

import numpy as np

from vectorize_tokens import vectorize_tokens
from vocabulary import vocabulary


assert len(vocabulary) < 256

SCHEMA = """
CREATE TABLE IF NOT EXISTS vectorized_source(
    hash TEXT PRIMARY KEY,
    np_array BLOB NOT NULL,     -- the numpy array, as a blob.
    n_tokens INTEGER NOT NULL   -- the amount of tokens, (excluding start/end)
);

CREATE TABLE IF NOT EXISTS fold_assignment(
    hash TEXT PRIMARY KEY,
    fold INTEGER NOT NULL   -- the fold assignment
);
"""


class CondensedCorpus:
    """
    Represents a corpus with condensed tokens according to a vocabulary.

    Can get results by rowid (ONE-INDEXED!) or by file SHA 256 hash.

    >>> from corpus import Token
    >>> c = CondensedCorpus.connect_to(':memory:')
    >>> tokens = (Token(value='var', type='Keyword', loc=None),)
    >>> c.insert('123abc', tokens)
    >>> file_hash, (x, y, z) = c['123abc']
    >>> x, y, z
    (0, 86, 99)
    >>> file_hash
    '123abc'
    >>> file_hash, rtokens = c[1]
    >>> file_hash
    '123abc'
    >>> x, y, z = rtokens
    >>> x, y, z
    (0, 86, 99)
    >>> c.min_index
    1
    >>> c.insert('foobar', tokens)
    >>> c.max_index
    2
    >>> list(c.hashes_in_fold(0))
    []
    >>> c.add_to_fold('foobar', 0)
    >>> list(c.hashes_in_fold(0))
    ['foobar']
    >>> c.add_to_fold('foobar', 1)
    Traceback (most recent call last):
      ...
    sqlite3.IntegrityError: UNIQUE constraint failed: fold_assignment.hash
    >>> c.add_to_fold('123abc', 0)
    >>> list(c.hashes_in_fold(0))
    ['foobar', '123abc']
    """

    def __init__(self, conn):
        with conn:
            conn.executescript(SCHEMA)
        self.conn = conn

    @classmethod
    def connect_to(cls, filename):
        conn = sqlite3.connect(filename)
        return cls(conn)

    def disconnect(self):
        self.conn.close()

    def get_tokens_by_hash(self, file_hash):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT np_array FROM vectorized_source
            WHERE hash = ?
        """, (file_hash,))
        blob, = cur.fetchone()
        return file_hash, unblob(blob)

    def get_result_by_rowid(self, rowid):
        assert isinstance(rowid, int)
        cur = self.conn.cursor()
        cur.execute("""
            SELECT hash, np_array FROM vectorized_source
            WHERE rowid = ?
        """, (rowid,))
        file_hash, blob = cur.fetchone()
        return file_hash, unblob(blob)

    @property
    def min_index(self):
        result, = self.conn.execute("""
            SELECT MIN(rowid) FROM vectorized_source
        """).fetchone()
        return result

    @property
    def max_index(self):
        result, = self.conn.execute("""
            SELECT MAX(rowid) FROM vectorized_source
        """).fetchone()
        return result

    def hashes_in_fold(self, fold_no):
        """
        Generate all hashes in the given fold number.
        """
        assert fold_no in self.fold_ids

        cur = self.conn.execute("""
            SELECT hash FROM fold_assignment WHERE fold = ?
        """, (fold_no,))
        yield from (result for result, in cur.fetchall())

    def files_in_fold(self, fold_no):
        """
        Generated all hash, token pairs from the corpus.
        """
        for file_hash in self.hashes_in_fold(fold_no):
            yield self.get_tokens_by_hash(file_hash)

    def __getitem__(self, key):
        if isinstance(key, (str, bytes)):
            return self.get_tokens_by_hash(key)
        else:
            return self.get_result_by_rowid(key)

    def insert(self, hash_, tokens):
        dimensions = 2 + len(tokens)
        array = np.empty(dimensions, dtype=np.uint8)

        for t, index in enumerate(vectorize_tokens(tokens)):
            array[t] = index

        filelike = io.BytesIO()
        np.save(filelike, array)

        with self.conn:
            self.conn.execute("""
                INSERT INTO vectorized_source(hash, np_array, n_tokens)
                     VALUES (?, ?, ?)
             """, (hash_, filelike.getbuffer(), len(tokens)))

    def add_to_fold(self, file_hash, fold_no):
        """
        Add a file hash to a fold.
        """
        assert isinstance(fold_no, int)
        with self.conn:
            self.conn.execute("""
                INSERT INTO fold_assignment(hash, fold) VALUES (?, ?)
             """, (file_hash, fold_no))

    @property
    def fold_ids(self):
        """
        A list of all current fold numbers.
        """
        cur = self.conn.execute("""
            SELECT DISTINCT fold
              FROM fold_assignment
        """)
        return [int(fold_id) for fold_id, in cur.fetchall()]

    @property
    def has_fold_assignments(self):
        """
        Does this corpus have fold assignments?
        """
        return len(self.fold_ids) > 0

    def destroy_fold_assignments(self):
        """
        Deletes any current fold assignments.
        """
        with self.conn:
            self.conn.execute("DELETE FROM fold_assignment")


def unblob(blob):
    assert isinstance(blob, bytes)
    with io.BytesIO(blob) as filelike:
        return np.load(filelike)
