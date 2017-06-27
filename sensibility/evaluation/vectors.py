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
from pathlib import Path
from typing import Iterator, MutableMapping, Optional

from ..lexical_analysis import Lexeme
from ..source_vector import SourceVector
from ..vocabulary import vocabulary
from .._paths import EVALUATION_DIR


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
        assert len(vocabulary) < 256
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
        size = 2 * 1024 * 1024  # 2 GiB
        self.conn.execute(f'PRAGMA mmap_size={size:d}')

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


def determine_from_language() -> sqlite3.Connection:
    from ..language import language
    path = os.fspath(EVALUATION_DIR / language.id / f'vectors.sqlite3')
    return sqlite3.connect(path)
