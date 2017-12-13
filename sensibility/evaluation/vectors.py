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
Goals: instantiate this automatically via language.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, MutableMapping, Optional, Union

from .._paths import get_vectors_path
from ..lexical_analysis import Lexeme
from ..source_vector import SourceVector


SCHEMA = """
CREATE TABLE IF NOT EXISTS vector (
    filehash    TEXT PRIMARY KEY,
    array       BLOB NOT NULL       -- the array, as a blob.
);
"""


class Vectors(MutableMapping[str, SourceVector]):
    """
    Stores vectors on disk, with the posibility of mmapping it all in memory.
    """

    def __init__(self, conn: Optional[sqlite3.Connection]=None) -> None:
        from ..language import language
        assert len(language.vocabulary) < 256
        if conn is None:
            self.conn = determine_from_language()
        else:
            self.conn = conn
        self._instantiate_schema()
        self._mmap()

    def _instantiate_schema(self) -> None:
        self.conn.executescript(SCHEMA)

    def _mmap(self) -> None:
        # XXX: Hardcoded amount to mmap.
        size = 2 * 1024 ** 3  # 2 GiB
        self.conn.execute(f'PRAGMA mmap_size={size:d}')
        # Turn off Durability
        self.conn.execute(f'PRAGMA synchronous = OFF')
        # Optimize for read-only access.
        self.conn.execute(f'PRAGMA journal_mode = OFF')

    def length_of_vectors(self, hashes: Iterable[str]) -> int:
        """
        Determines the total number of tokens in the given hashes.
        """
        with query_table(self.conn, hashes), self.conn:
            n_tokens, = self.conn.execute('''
                SELECT SUM(LENGTH(array))
                  FROM vector NATURAL JOIN query
            ''').fetchone()
        return n_tokens or 0

    def disconnect(self) -> None:
        self.conn.close()

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __getitem__(self, filehash: str) -> SourceVector:
        cur = self.conn.execute("""
            SELECT array FROM vector WHERE filehash = ?
         """, (filehash,))
        item = cur.fetchone()
        if item is None:
            raise KeyError(filehash)
        else:
            return SourceVector.from_bytes(item[0])

    def __setitem__(self, filehash: str, vector: SourceVector) -> None:
        """
        Insert tokens in the database of vectors.
        """
        byte_string: bytes = vector.to_bytes()
        with self.conn:
            self.conn.execute("""
                INSERT INTO vector(filehash, array)
                     VALUES (?, ?)
             """, (filehash, byte_string))

    def __delitem__(self):
        raise NotImplementedError

    @classmethod
    def from_filename(cls, path: Union[str, os.PathLike]) -> 'Vectors':
        return cls(sqlite3.connect(os.fspath(path)))


@contextmanager
def query_table(conn: sqlite3.Connection, hashes: Iterable[str]):
    """
    Context manager that creates a table called `query` that can be joined
    against to speed up queries.
    """
    with conn:
        conn.execute('CREATE TEMPORARY TABLE query(filehash PRIMARY KEY)')
        conn.executemany('''
            INSERT INTO query(filehash) VALUES (?)
        ''', ((fh,) for fh in hashes))
    yield
    with conn:
        conn.execute('DROP TABLE IF EXISTS query')


def determine_from_language() -> sqlite3.Connection:
    path = os.fspath(get_vectors_path())
    return sqlite3.connect(path)
