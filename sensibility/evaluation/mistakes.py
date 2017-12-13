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
from typing import Iterable, Iterator, NewType, Optional, Tuple

from sensibility import Edit
from sensibility.evaluation.distance import FixEvent
from sensibility.vocabulary import Vind

# Different types of IDs that are both just ints, but it's very important
# that the IDs don't get confused for one-another.
SFID = NewType('SFID', int)
MEID = NewType('MEID', int)
Revision = Tuple[SFID, MEID]

# This schema assumes a table `mistake` already exists.
SCHEMA = r"""
CREATE TABLE IF NOT EXISTS distance(
    source_file_id  INT,
    before_id       INT,
    levenshtein     INT,
    PRIMARY KEY (source_file_id, before_id)
);

CREATE TABLE IF NOT EXISTS edit(
    -- Link back to mistake table (no foreign keys :/)
    source_file_id  INT,
    before_id       INT,

    -- Line number of the error.
    line_no         INT NOT NULL,

    -- How to go from the good file to the bad file.
    mistake        TEXT NOT NULL,
    mistake_index  INT NOT NULL,

    -- How to go from the bad file to the good file.
    fix             TEXT NOT NULL,
    fix_index       TEXT NOT NULL,

    -- The token that used to be there (can be null).
    old_token       TEXT,
    -- The token that is there now (can be null).
    new_token       TEXT,

    PRIMARY KEY (source_file_id, before_id)
);
"""


class Mistake:
    """
    Represents a mistake in the database.
    """
    def __init__(self, sfid: SFID, meid: MEID,
                 before: bytes, after: bytes) -> None:
        self.sfid = sfid
        self.meid = meid
        self.before = before
        self.after = after

    def __repr__(self) -> str:
        return '<mistake sfid=%d, mfid=%d>' % (self.sfid, self.meid)


class Mistakes(Iterable[Mistake]):
    """
    Access to the mistake database.
    """
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.conn.executescript(SCHEMA)

    def __iter__(self) -> Iterator[Mistake]:
        query = 'SELECT source_file_id, before_id, before, after FROM mistake'
        for row in self.conn.execute(query):
            yield Mistake(*row)

    @property
    def eligible_mistakes(self) -> Iterator[Mistake]:
        query = '''
            SELECT source_file_id, before_id, before, after
            FROM mistake NATURAL JOIN distance
            WHERE levenshtein = 1
            '''
        for row in self.conn.execute(query):
            yield Mistake(*row)

    def insert_distance(self, m: Mistake, dist: int) -> None:
        with self.conn:
            self.conn.execute('''
                INSERT INTO distance(source_file_id, before_id, levenshtein)
                VALUES (?, ?, ?)
            ''', (m.sfid, m.meid, dist))

    def insert_fix_event(self, m: Mistake, event: FixEvent) -> None:
        mistake, mistake_index, _, _ = event.mistake.serialize()
        fix, fix_index, _, _ = event.fix.serialize()
        with self.conn:
            self.conn.execute('''
                INSERT INTO edit(
                    source_file_id, before_id, line_no,
                    mistake, mistake_index, fix, fix_index,
                    new_token, old_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (m.sfid, m.meid, event.line_no,
                  mistake, mistake_index, fix, fix_index,
                  event.new_token, event.old_token,))
