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
It should really be called "corpus", shouldn't it?
"""

from pathlib import Path

from sqlalchemy import create_engine  # type: ignore
from sqlalchemy import (
    Table, Column,
    Integer, String, DateTime, LargeBinary,
    MetaData,
    ForeignKeyConstraint
)  # type: ignore
from sqlalchemy.sql import select  # type: ignore

from .connection import sqlite3_path
from .models import RepositoryMetadata, SourceFileInRepository

from ..language.python import WordCount


class Corpus:
    def __init__(self, engine=None):
        if engine is not None:
            self.engine = engine
        else:
            self.engine = create_engine(f"sqlite:///{sqlite3_path}")

        metadata = self._initialize_schema()
        if self.empty():
            metadata.create_all(self.engine)

        self.conn = self.engine.connect()

    def __getitem__(self, filehash: str) -> bytes:
        """
        Yields a file from the corpus.
        """
        return self.get_source(filehash)

    def empty(self):
        metadata = MetaData()
        metadata.reflect(self.engine)
        return 'repository' not in metadata

    def insert_repository(self, repo: RepositoryMetadata) -> None:
        self.conn.execute(self.repository.insert(),
                          owner=repo.owner, name=repo.name,
                          revision=repo.revision, license=repo.license,
                          commit_date=repo.commit_date)

    def insert_source_file_from_repo(self, entry: SourceFileInRepository) -> None:
        trans = self.conn.begin()
        try:
            self.conn.execute((self.source_file.insert()
                                .prefix_with('OR IGNORE', dialect='sqlite')),
                              source=entry.source_file.source,
                              hash=entry.filehash)
            self.conn.execute(self.repository_source.insert(),
                              owner=entry.owner, name=entry.name,
                              hash=entry.filehash, path=str(entry.path))
        except:
            trans.rollback()
            raise
        else:
            trans.commit()

    def insert_source_summary(self, filehash: str, summary: WordCount) -> None:
        """
        Insert the word count into the source summary.
        """
        self.conn.execute(self.source_summary.insert(),
                          hash=filehash,
                          sloc=summary.sloc, n_tokens=summary.n_tokens)

    def insert_failure(self, filehash: str) -> None:
        """
        Insert the word count into the source summary.
        """
        self.conn.execute(self.failure.insert(), hash=filehash)

    def get_source(self, filehash: str) -> bytes:
        """
        Returns the source code for one file.
        """
        query = select([self.source_file.c.source])\
            .where(self.source_file.c.hash == filehash)
        result, = self.conn.execute(query)
        return result[0]

    def _initialize_schema(self):
        """
        The schema for this database.

        TODO: adapt from GHTorrent's database.
        """
        metadata = MetaData()
        cascade_all = dict(onupdate='CASCADE', ondelete='CASCADE')

        self.repository = Table('repository', metadata,
            Column('owner', String, primary_key=True),
            Column('name', String, primary_key=True),

            Column('revision', String, nullable=False),
            Column('commit_date', DateTime, nullable=False),
            Column('license', String)
        )

        self.source_file = Table('source_file', metadata,
            Column('hash', String, primary_key=True),

            Column('source', LargeBinary, nullable=False)
        )

        self.repository_source = Table('repository_source', metadata,
            Column('owner', String, primary_key=True),
            Column('name', String, primary_key=True),
            Column('hash', String, primary_key=True),

            Column('path', String, nullable=False),
            ForeignKeyConstraint(*_to('repository', 'owner', 'name'),
                                 **cascade_all),
            ForeignKeyConstraint(*_to('source_file', 'hash'),
                                 **cascade_all)
        )

        self.source_summary = Table('source_summary', metadata,
            Column('hash', String, primary_key=True),

            Column('sloc', Integer, nullable=False),
            Column('n_tokens', Integer, nullable=False),

            ForeignKeyConstraint(*_to('source_file', 'hash'),
                                 **cascade_all)
        )

        self.failure = Table('failure', metadata,
            Column('hash', String, primary_key=True),

            ForeignKeyConstraint(*_to('source_file', 'hash'),
                                 **cascade_all)
        )
        return metadata


def _to(table_name, *columns):
    yield columns
    yield tuple(f"{table_name}.{col}" for col in columns)
