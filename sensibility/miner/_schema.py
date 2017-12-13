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

from typing import Any, Iterable, Tuple

from sqlalchemy.ext import compiler  # type: ignore
from sqlalchemy.schema import DDLElement  # type: ignore
from sqlalchemy.sql import table  # type: ignore

from sqlalchemy import (Column, DateTime,  # type: ignore
                        ForeignKeyConstraint, Index, Integer, LargeBinary,
                        MetaData, String, Table, literal_column, select)


def _to(table_name, *columns):
    "Helper: yields arguments for creating foreign key relations."
    yield columns
    yield tuple(f"{table_name}.{col}" for col in columns)


metadata = MetaData()
cascade_all = dict(onupdate='CASCADE', ondelete='CASCADE')

# stores metadata for the database.
meta = Table(
    'meta', metadata,
    Column('key', String, primary_key=True),
    Column('value', String)
)

repository = Table(
    'repository', metadata,
    Column('owner', String, primary_key=True),
    Column('name', String, primary_key=True),

    Column('revision', String, nullable=False),
    Column('commit_date', DateTime, nullable=False),
    Column('license', String),

    # comment="A source code repository from GitHub at a particular revision"
)

source_file = Table(
    'source_file', metadata,
    Column('hash', String, primary_key=True),

    Column('source', LargeBinary, nullable=False),

    # comment="A source file, divorced from any repo it may belong to"
)

repository_source = Table(
    'repository_source', metadata,
    Column('owner', String, primary_key=True),
    Column('name', String, primary_key=True),
    Column('hash', String, primary_key=True),
    Column('path', String, primary_key=True),

    # Makes accessing a filehash's information take O(log n) time.
    Index('idx_filehash', 'hash'),

    ForeignKeyConstraint(*_to('repository', 'owner', 'name'),
                         **cascade_all),
    ForeignKeyConstraint(*_to('source_file', 'hash'),
                         **cascade_all),

    # comment=(
    #     "Relates a source file to a repository."
    #     " A many-to-many relationship."
    #     " Note: A file hash may be in the same repository twice!"
    # )
)

source_summary = Table(
    'source_summary', metadata,
    Column('hash', String, primary_key=True),

    Column('sloc', Integer, nullable=False),
    Column('n_tokens', Integer, nullable=False),

    ForeignKeyConstraint(*_to('source_file', 'hash'),
                         **cascade_all),

    # comment="Source files that are syntactically valid."
)

failure = Table(
    'failure', metadata,
    Column('hash', String, primary_key=True),
    # TODO: add a reason why
    Column('reason', String, default='Syntax error'),

    ForeignKeyConstraint(*_to('source_file', 'hash'),
                         **cascade_all),

    # comment="Files that are syntactically invalid."
)


# Create view:
# https://bitbucket.org/zzzeek/sqlalchemy/wiki/UsageRecipes/Views

class CreateView(DDLElement):
    def __init__(self, name, selectable):
        self.name = name
        self.selectable = selectable


class DropView(DDLElement):
    def __init__(self, name):
        self.name = name


@compiler.compiles(CreateView)
def compile_create(element, compiler, **kw):
    return "\nCREATE VIEW %s AS\n%s" % (element.name, compiler.sql_compiler.process(element.selectable))


@compiler.compiles(DropView)
def compile_drop(element, compiler, **kw):
    return "DROP VIEW %s" % (element.name)


def view(name, metadata, selectable):
    t = table(name)

    for c in selectable.c:
        c._make_proxy(t)

    CreateView(name, selectable).execute_at('after-create', metadata)
    DropView(name).execute_at('before-drop', metadata)
    return t


eligible_source = view(
    'eligible_source', metadata,
    select([source_summary.c.hash, source_summary.c.n_tokens,
            source_summary.c.sloc])
    .where(source_summary.c.n_tokens > literal_column("0"))
    .where(source_summary.c.hash.notin_(select([failure.c.hash])))
)
