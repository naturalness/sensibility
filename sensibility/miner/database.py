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

from sqlalchemy import create_engine, MetaData # type: ignore

from .connection import sqlite3_path
from .models import RepositoryMetadata, SourceFileInRepository


here = Path(__file__).parent


class Database:
    def __init__(self):
        self.engine = create_engine(f"sqlite:///{sqlite3_path}")
        self._connect()

    def _connect(self) -> None:
        self._insert_schema()
        meta = self.meta = MetaData()
        meta.reflect(bind=self.engine)
        assert all(table in meta.tables for table in {
            'repository', 'source_file', 'repository_source'
        })
        self.conn = self.engine.connect()

    def _insert_schema(self) -> None:
        if Path(str(sqlite3_path)).exists():
            return

        from sensibility.miner.connection import sqlite3_connection
        with open(here / 'schema.sql') as schema:
            sqlite3_connection.execute('PRAGMA journal_mode = WAL')
            sqlite3_connection.execute('PRAGMA synchronous = NORMAL')
            sqlite3_connection.executescript(schema.read())
        sqlite3_connection.commit()

    def __getattr__(self, name):
        if name.endswith('_table'):
            end = len('_table')
            return self.meta.tables[name[:-end]]
        return super().__getattr__(name)

    def insert_repository(self, repo: RepositoryMetadata) -> None:
        self.conn.execute(self.repository_table.insert(),
                          owner=repo.owner, name=repo.name,
                          revision=repo.revision, license=repo.license,
                          commit_date=repo.commit_date)

    def insert_source_file_from_repo(self, entry: SourceFileInRepository) -> None:
        trans = self.conn.begin()
        try:
            self.conn.execute((self.source_file_table.insert()
                                .prefix_with('OR IGNORE', dialect='sqlite')),
                              source=entry.source_file.source,
                              hash=entry.filehash)
            self.conn.execute(self.repository_source_table.insert(),
                              owner=entry.owner, name=entry.name,
                              hash=entry.filehash, path=str(entry.path))
        except:
            trans.rollback()
            raise
        else:
            trans.commit()
