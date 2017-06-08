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

import os
from typing import Any, Dict, Set, Union
from pathlib import PurePosixPath

from sqlalchemy import create_engine, event, MetaData  # type: ignore
from sqlalchemy.engine import Engine  # type: ignore
from sqlalchemy.sql import select  # type: ignore

from .connection import get_sqlite3_path
from .models import RepositoryMetadata, SourceFileInRepository, MockSourceFile
from ._schema import (
    failure, meta, repository, repository_source, source_file, source_summary,
    metadata
)

from sensibility.language import SourceSummary


class NewCorpusError(Exception):
    pass


class FileInfo:
    def __init__(self, mappings: Set[SourceFileInRepository],
                 summary: SourceSummary) -> None:
        assert len(mappings) > 0
        self._mappings = mappings
        self.summary = summary

    @property
    def filehash(self) -> str:
        return self._any.filehash

    @property
    def owner(self) -> str:
        return self._any.owner

    @property
    def name(self) -> str:
        return self._any.name

    @property
    def href(self) -> str:
        return self._any.href

    @property
    def license(self) -> str:
        return self._any.license

    @property
    def path(self) -> PurePosixPath:
        return self._any.path

    @property
    def n_tokens(self) -> int:
        return self.summary.n_tokens

    @property
    def sloc(self) -> int:
        return self.summary.sloc

    @property
    def is_unique(self) -> bool:
        return len(self._mappings) == 1

    @property
    def _any(self) -> SourceFileInRepository:
        return next(iter(self._mappings))


class Corpus:
    def __init__(self, engine=None,
                 url: str=None, path: Union[os.PathLike, str]=None,
                 read_only=False) -> None:
        if engine is not None:
            self.engine = engine
        else:
            if url is None:
                if path is None:
                    path = get_sqlite3_path() if path is None else path
                url = f"sqlite:///{os.fspath(path)}"
            self.engine = create_engine(url)

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
        try:
            result, = self.conn.execute(query)
        except ValueError:
            raise NewCorpusError
        else:
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

    def set_metadata(self, **kwargs: Any) -> None:
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

    def get_info(self, filehash: str) -> FileInfo:
        # Do an intense query, combining multiple tables.
        query = select([source_summary, repository_source, repository]).select_from(
            source_file.join(source_summary)\
            .join(repository_source)\
            .join(repository)
        ).where(source_summary.c.hash == filehash)
        results = self.conn.execute(query).fetchall()

        # TODO: what if it's not found?

        mock_file = MockSourceFile(filehash)
        repos = {
            (row[repository.c.owner], row[repository.c.name]): RepositoryMetadata(
                owner=row[repository.c.owner],
                name=row[repository.c.name],
                license=row[repository.c.license],
                revision=row[repository.c.revision],
                commit_date=row[repository.c.commit_date]
            ) for row in results
        }
        mappings = set(
            SourceFileInRepository(
                repository=repos[row[repository.c.owner], row[repository.c.name]],
                source_file=mock_file,
                path=PurePosixPath(row[repository_source.c.path])
            ) for row in results
        )

        row, *_ = results
        summary = SourceSummary(sloc=row[source_summary.c.sloc],
                                n_tokens=row[source_summary.c.n_tokens])
        return FileInfo(mappings, summary)

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
                cur.executescript('''
                    PRAGMA journal_mode = WAL;
                    PRAGMA synchronous = NORMAL;
                ''')
            cur.close()
