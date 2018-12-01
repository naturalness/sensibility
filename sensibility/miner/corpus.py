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
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Set, Tuple, Union

from sqlalchemy import MetaData, create_engine, event  # type: ignore
from sqlalchemy.engine import Engine  # type: ignore
from sqlalchemy.sql import select, text  # type: ignore

from sensibility.language import SourceSummary

from ._schema import (eligible_source, failure, meta, metadata, repository,
                      repository_source, source_file, source_summary)
from .connection import get_sqlite3_path
from .models import (MockSourceFile, RepositoryID, RepositoryMetadata,
                     SourceFile, SourceFileInRepository)


class NewCorpusError(Exception):
    """
    Raised when querying an empty corpus.
    """
    pass


class FileInfo:
    """
    Contains metadata over a file, as referenced by a filehash.
    Note that one filehash maps uniquely to content, however the same content
    can be present in multiple files, within a repository, or between
    repositories. Use is_unique to determine whether this is the only copy of
    the file within the corpus.
    """
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
        """
        True when there is one and only one file with the contents of this
        file. False otherwise.
        """
        return len(self._mappings) == 1

    @property
    def _any(self) -> SourceFileInRepository:
        """
        Returns any owner/repo/path containing this file.
        """
        return next(iter(self._mappings))


class Corpus:
    """
    A corpus of source code, backed by an SQLite3 database.

    Uses SQLAlchemy (as an experiment), but it's honestly not really worth it.
    """
    def __init__(self, engine=None,
                 url: str = None,
                 path: Union[os.PathLike, str] = None,
                 writable=False) -> None:
        if engine is not None:
            self.engine = engine
        else:
            if url is None:
                if path is None:
                    path = get_sqlite3_path()

                # Ensure the containing directory exists.
                effective_path = Path(path)
                containing_dir = effective_path.parent
                if not containing_dir.is_dir():
                    containing_dir.mkdir(parents=True)

                url = f"sqlite:///{os.fspath(effective_path)}"

            self.engine = create_engine(url)

        self._initialize_sqlite3(writable)

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

    @property
    def eligible_sources(self) -> Iterator[SourceFile]:
        """
        Yields source files eligible for training, validation, and testing
        (source and filehash).
        """
        query = select([source_file.c.source])\
            .select_from(source_file.join(eligible_source))
        for row in self.conn.execute(query):
            yield SourceFile(row[source_file.c.source])

    @property
    def source_summaries(self) -> Iterator[Tuple[str, SourceSummary]]:
        """
        Yields sources with computed summaries.
        """
        query = select([source_summary.c.hash, source_summary.c.n_tokens,
                        source_summary.c.sloc])
        for row in self.conn.execute(query):
            yield (row[source_summary.c.hash],
                   SourceSummary(sloc=row[source_summary.c.sloc],
                                 n_tokens=row[source_summary.c.n_tokens]))

    @property
    def sources_with_repository(self) -> Iterator[Tuple[str, str,
                                                        PurePosixPath, bytes]]:
        """
        Returns ALL sources including their repository and their repository
        path.
        """
        query = select([repository_source.c.owner, repository_source.c.name,
                        repository_source.c.path, source_file.c.source])\
            .select_from(repository_source.join(source_file))
        for owner, name, pathstr, source in self.conn.execute(query):
            yield owner, name, PurePosixPath(pathstr), source

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
        except Exception:
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

    def insert_failure(self, filehash: str, reason: str = None,
                       ignore: bool = False) -> None:
        """
        Insert the word count into the source summary.

        Can add an optional reason, and choose to ignore
        """
        self.conn.execute((failure.insert().prefix_with('OR IGNORE')
                           if ignore else failure.insert()),
                          hash=filehash, reason=reason)

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
            source_file.join(repository_source)
            .join(repository)
            .join(source_summary, isouter=True)
        ).where(source_file.c.hash == filehash)
        results = self.conn.execute(query).fetchall()

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

        # TODO: what if it's not found?
        row, *_ = results
        summary = SourceSummary(sloc=row[source_summary.c.sloc],
                                n_tokens=row[source_summary.c.n_tokens])
        return FileInfo(mappings, summary)

    def get_repositories_with_n_tokens(self) -> Iterator[Tuple[RepositoryID, int]]:
        """
        An incredibly specific method to return repositories, along with how
        many tokens that repository contains. This number may include
        duplicates.
        """
        query = text("""
            SELECT owner, name, n_tokens
            FROM repository_source JOIN eligible_source USING (hash)
            GROUP BY owner, name
        """)
        cursor = self.conn.execute(query)
        for owner, name, n_tokens in cursor:
            yield RepositoryID(owner, name), n_tokens

    def get_eligible_hashes_in_repo(self, repo: RepositoryID, elligible=False) -> Iterator[str]:
        """
        Lists all file hashes in this repository.
        """
        query = select([repository_source.c.hash])\
            .select_from(repository_source.join(
                eligible_source,
                repository_source.c.hash == eligible_source.c.hash
            ))\
            .where(repository_source.c.owner == repo.owner)\
            .where(repository_source.c.name == repo.name)
        for row in self.conn.execute(query):
            yield row[repository_source.c.hash]

    def _initialize_sqlite3(self, writable: bool) -> None:
        """
        Set some pragmas for initially creating the SQLite3 database.
        """

        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_connection, _connection_record):
            cur = dbapi_connection.cursor()
            cur.execute('PRAGMA encoding = "UTF-8"')
            cur.execute('PRAGMA foreign_keys = ON')
            if writable:
                cur.executescript('''
                    PRAGMA journal_mode = WAL;
                    PRAGMA synchronous = NORMAL;
                ''')
            cur.close()
