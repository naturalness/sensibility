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

import sqlite3
from pathlib import Path
from typing import Tuple, Sized, Iterable, Optional, Iterator

from .edit import Edit
from ._paths import MUTATIONS_PATH
from .source_file import SourceFile


MutationInfo = Tuple[SourceFile, Edit]


class Mutations(Sized, Iterable[MutationInfo]):
    """
    Persist every mutation, and enough data to reconstruct every single
    prediction.
    """

    SCHEMA = r"""
    PRAGMA encoding = "UTF-8";

    CREATE TABLE IF NOT EXISTS mutant_with_status (
        hash            TEXT NOT NULL,      -- file hash
        type            TEXT NOT NULL,      -- 'i', 'x', or 's'
        location        INTEGER NOT NULL,   -- location in the file (0-indexed)
        new_token       INTEGER,
        original_token  INTEGER,

        correct         BOOLEAN NOT NULL,  -- Whether the result is valid.

        PRIMARY KEY (hash, type, location, new_token, original_token)
    );

    -- Only syntactically-incorrect mutants.
    CREATE VIEW IF NOT EXISTS mutant
    AS SELECT hash, type, location, original_token, new_token
         FROM mutant_with_status
        WHERE correct = 0;

    -- Only syntacticall-correct mutants.
    CREATE VIEW IF NOT EXISTS correct_mutant
    AS SELECT hash, type, location, original_token, new_token
         FROM mutant_with_status
        WHERE correct = 1;
    """

    def __init__(self, database: Path=MUTATIONS_PATH,
                 read_only: bool = False) -> None:
        self.program: Optional[SourceFile] = None
        self._dbname = database
        self._conn: Optional[sqlite3.Connection] = None
        self.read_only = read_only

    def __len__(self) -> int:
        assert self._conn
        return self._conn.execute(r'''
            SELECT COUNT(*) FROM mutant
        ''').fetchall()[0][0]

    def __iter__(self) -> Iterator[Tuple[SourceFile, Edit]]:
        assert self._conn
        cur = self._conn.execute(r'''
            SELECT hash, type, location, new_token, original_token
            FROM mutant
        ''')
        for file_hash, code, location, new_token, original_token in cur:
            mutation = Edit.deserialize(code, location, new_token,
                                        original_token)
            yield SourceFile(file_hash), mutation

    @property
    def current_source_hash(self) -> str:
        """
        hash of the current source file being mutated.
        """
        if self.program:
            return self.program.file_hash
        else:
            raise ValueError('program not set')

    def add_mutant(self, mutation: Edit) -> None:
        """
        Register a new complete mutation.
        """
        return self._add_mutant(mutation, correct=False)

    def add_correct_file(self, mutation: Edit) -> None:
        """
        Records that a mutation created a syntactically-correct file.
        """
        return self._add_mutant(mutation, correct=True)

    def _add_mutant(self, mutation: Edit, *, correct: bool) -> None:
        assert self._conn

        sql = r'''
            INSERT INTO mutant_with_status (
                hash, type, location, new_token, original_token, correct
            ) VALUES (:hash, :type, :location, :new, :original, :correct)
        '''

        code, location, new_token, original_token = mutation.serialize()

        args = dict(hash=self.current_source_hash,
                    type=code, location=location,
                    new=new_token, original=original_token,
                    correct=int(correct))

        with self._conn:
            self._conn.execute(sql, args)

    def __enter__(self) -> 'Mutations':
        # Connect to the database
        conn = self._conn = sqlite3.connect(str(self._dbname))
        # Initialize the database.
        if not self.read_only:
            with conn:
                conn.executescript(self.SCHEMA)
                # Some speed optimizations:
                # http://codificar.com.br/blog/sqlite-optimization-faq/
                conn.executescript(r'''
                    PRAGMA journal_mode = WAL;
                    PRAGMA synchronous = normal;
                ''')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.close()
        self._conn = self._program = None
