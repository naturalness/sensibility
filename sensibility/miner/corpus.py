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
Access to the corpus.
"""

from pathlib import Path

from sqlalchemy import create_engine, event, MetaData  # type: ignore
from sqlalchemy.engine import Engine  # type: ignore
from sqlalchemy.sql import select  # type: ignore

from .connection import sqlite3_path
from .models import RepositoryMetadata, SourceFileInRepository
from ._schema import (
    failure, repository, repository_source, source_file, source_summary,
    metadata
)

from ..language.python import WordCount


class Corpus:
    def __init__(self, engine=None, read_only=False) -> None:
        if engine is not None:
            self.engine = engine
        else:
            self.engine = create_engine(f"sqlite:///{sqlite3_path}")

        self.initialize_sqlite3(read_only)

        if self.empty():
            metadata.create_all(self.engine)

        self.conn = self.engine.connect()

    def initialize_sqlite3(self, read_only: bool) -> None:
        """
        Set some pragmas for initialy creating the SQLite3 database.
        """

        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_connection, _connection_record):
            cur = dbapi_connection.cursor()
            cur.execute('PRAGMA encoding = "UTF-8"')
            cur.execute('PRAGMA foreign_keys = ON')
            if not read_only:
                cur.execute('PRAGMA journal_mode = WAL')
                cur.execute('PRAGMA synchronous = NORMAL')
            cur.close()

    def __getitem__(self, filehash: str) -> bytes:
        """
        Returns a file from the corpus.
        """
        return self.get_source(filehash)

    def empty(self) -> bool:
        metadata = MetaData()
        metadata.reflect(self.engine)
        return 'repository' not in metadata

    def insert_repository(self, repo: RepositoryMetadata) -> None:
        self.conn.execute(repository.insert(),
                          owner=repo.owner, name=repo.name,
                          revision=repo.revision, license=repo.license,
                          commit_date=repo.commit_date)

    def insert_source_file_from_repo(self, entry: SourceFileInRepository) -> None:
        trans = self.conn.begin()
        try:
            self.conn.execute((source_file.insert()
                                .prefix_with('OR IGNORE', dialect='sqlite')),
                              source=entry.source_file.source,
                              hash=entry.filehash)
            self.conn.execute(repository_source.insert(),
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
        self.conn.execute(source_summary.insert(),
                          hash=filehash,
                          sloc=summary.sloc, n_tokens=summary.n_tokens)

    def insert_failure(self, filehash: str) -> None:
        """
        Insert the word count into the source summary.
        """
        self.conn.execute(failure.insert(), hash=filehash)

    def get_source(self, filehash: str) -> bytes:
        """
        Returns the source code for one file.
        """
        query = select([source_file.c.source])\
            .where(source_file.c.hash == filehash)
        result, = self.conn.execute(query)
        return result[source_file.c.source]
