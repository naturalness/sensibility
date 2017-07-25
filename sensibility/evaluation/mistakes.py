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

from sensibility.vocabulary import Vind


# Different types of IDs that are both just ints, but it's very important
# that the IDs don't get confused for one-another.
SFID = NewType('SFID', int)
MEID = NewType('MEID', int)
Revision = Tuple[SFID, MEID]

# To work along side java-mistakes.sqlite3
SCHEMA = r"""
CREATE TABLE IF NOT EXISTS distance(
    source_file_id  INT,
    before_id       INT,
    levenshtein     INT,
    PRIMARY KEY (source_file_id, before_id)
);

CREATE TABLE IF NOT EXISTS edit(
    source_file_id  INT,
    before_id       INT,

    -- Line number of the error.
    line_no         INT NOT NULL,

    -- How to go from the good file to the bad file.
    edit            TEXT NOT NULL,
    position        INT NOT NULL,
    new_token       TEXT,

    -- How to go from the bad file to the good file.
    fix             TEXT NOT NULL,
    position        TEXT NOT NULL,
    new_tok         TEXT,

    PRIMARY KEY (source_file_id, before_id)
);
"""


class EditType:
    """
    Symbolic constants for Insertion, Deletion, and Substitution.
    """
    id: str

    def __repr__(self) -> str:
        return type(self).__name__


Insertion = type('Insertion', (EditType,), {'id': 'i'})()
Deletion = type('Deletion', (EditType,), {'id': 'x'})()
Substitution = type('Substitution', (EditType,), {'id': 's'})()


# TODO: use Edit classes from sensibility?
class Edit:
    """
    Represents an edit.
    """
    __slots__ = 'type', 'position', 'new_token'

    def __init__(self, type_name: EditType, position: int,
                 new_token: Optional[Vind]) -> None:
        self.type = type_name
        self.position = position
        self.new_token = new_token


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

    def insert_edit(self, m: Mistake, edit: Edit) -> None:
        with self.conn:
            self.conn.execute('''
                INSERT INTO edit(
                    source_file_id, before_id, edit, position, new_token
                )
                VALUES (?, ?, ?, ?, ?)
            ''', (m.sfid, m.meid, edit.type.id, edit.position, edit.new_token))
