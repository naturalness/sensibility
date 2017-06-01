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

from typing import Any, Dict

from sqlalchemy import create_engine, event, MetaData  # type: ignore
from sqlalchemy.engine import Engine  # type: ignore
from sqlalchemy.sql import select  # type: ignore

from .connection import get_sqlite3_path
from .models import RepositoryMetadata, SourceFileInRepository
from ._schema import (
    failure, meta, repository, repository_source, source_file, source_summary,
    metadata
)

from sensibility.language import SourceSummary


class Corpus:
    def __init__(self, engine=None, read_only=False) -> None:
        if engine is not None:
            self.engine = engine
        else:
            self.engine = create_engine(f"sqlite:///{get_sqlite3_path()}")

        self._initialize_sqlite3(read_only)

        if self.empty:
            metadata.create_all(self.engine)

        self.conn = self.engine.connect()

    @property
    def language(self) -> str:
        """
        The main laguage of the mined sources.
        """
        query = select([meta.c.value]).\
            where(meta.c.key == 'language')
        result, = self.conn.execute(query)
        return result[meta.c.value]

    @property
    def empty(self) -> bool:
        metadata = MetaData()
        metadata.reflect(self.engine)
        return 'repository' not in metadata

    def __getitem__(self, filehash: str) -> bytes:
        """
        Returns a file from the corpus.
        """
        return self.get_source(filehash)

    def set_metadata(self, **kwargs: Dict[str, Any]) -> None:
        """
        Sets the metadata table.
        """
        self.conn.execute(meta.insert(), [
            {'key': key, 'value': str(value)}
            for key, value in kwargs.items()
        ])

    def insert_repository(self, repo: RepositoryMetadata) -> None:
        """
        Insert a repository's metadata into the database.
        """
        self.conn.execute(repository.insert(),
                          owner=repo.owner, name=repo.name,
                          revision=repo.revision, license=repo.license,
                          commit_date=repo.commit_date)

    def insert_source_file_from_repo(self, entry: SourceFileInRepository) -> None:
        """
        Insert a source file from a repository into the database. If the
        source file already exists in the database, only the relation is
        inserted.
        """
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

    def insert_source_summary(self, filehash: str, summary: SourceSummary) -> None:
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

    def _initialize_sqlite3(self, read_only: bool) -> None:
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
