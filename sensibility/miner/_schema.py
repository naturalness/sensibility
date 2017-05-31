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
Defines the schema for the database containing repositories and source code.

TODO: consider partially mirroring GHTorrent's database schema.

-------
SQLite3
-------

It is recommended to use WAL mode and normal synchronization when updating the
database:

    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;

It is recommend to use DELETE mode when accessing the database read-only:

    PRAGMA journal_mode = DELETE;


"""

from sqlalchemy import (  # type: ignore
    Table, Column,
    Integer, String, DateTime, LargeBinary,
    MetaData,
    ForeignKeyConstraint
)

from typing import Iterable, Any, Tuple


def _to(table_name, *columns):
    "Helper: yields arguments for creating foreign key relations."
    yield columns
    yield tuple(f"{table_name}.{col}" for col in columns)


metadata = MetaData()
cascade_all = dict(onupdate='CASCADE', ondelete='CASCADE')

repository = Table(
    'repository', metadata,
    Column('owner', String, primary_key=True),
    Column('name', String, primary_key=True),

    Column('revision', String, nullable=False),
    Column('commit_date', DateTime, nullable=False),
    Column('license', String),

    #comment="A source code repository from GitHub at a particular revision"
)

source_file = Table(
    'source_file', metadata,
    Column('hash', String, primary_key=True),

    Column('source', LargeBinary, nullable=False),

    #comment="A source file, divorced from any repo it may belong to"
)

repository_source = Table(
    'repository_source', metadata,
    Column('owner', String, primary_key=True),
    Column('name', String, primary_key=True),
    Column('hash', String, primary_key=True),

    Column('path', String, nullable=False),
    ForeignKeyConstraint(*_to('repository', 'owner', 'name'),
                         **cascade_all),
    ForeignKeyConstraint(*_to('source_file', 'hash'),
                         **cascade_all),

    #comment=(
    #    "Relates a source file to a repository. "
    #    "A many-to-many relationship."
    #)
)

source_summary = Table(
    'source_summary', metadata,
    Column('hash', String, primary_key=True),

    Column('sloc', Integer, nullable=False),
    Column('n_tokens', Integer, nullable=False),

    ForeignKeyConstraint(*_to('source_file', 'hash'),
                         **cascade_all),

    #comment="Source files that are syntactically valid."
)

failure = Table(
    'failure', metadata,
    Column('hash', String, primary_key=True),

    ForeignKeyConstraint(*_to('source_file', 'hash'),
                         **cascade_all),

    #comment="Files that are syntacticall invalid."
)
