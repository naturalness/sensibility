#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
>>> db = Database()
>>> rev = '1b8f23c763d08130ec2081c35e7f9fe0d392d700'
>>> repo = Repository.create('github', 'example', 'mit', rev)
>>> ret = db.add_repository(repo)
>>> ret == repo
True
"""

import logging
import sqlite3

from path import Path

from datatypes import Repository, SourceFile, ParsedSource


logger = logging.getLogger(__name__)

SCHEMA_FILENAME = Path(__file__).parent / 'schema.sql'
with open(SCHEMA_FILENAME, encoding='UTF-8') as schema_file:
    SCHEMA = schema_file.read()
    del schema_file


class Database:
    def __init__(self, connection=None):
        if connection is None:
            logger.warn("Using in memory database!")
            self.conn = sqlite3.connect(':memory:')
        else:
            self.conn = connection

        self._initialize_db()

    def _initialize_db(self):
        conn = self.conn
        if self._is_database_empty():
            conn.executescript(SCHEMA)
            conn.commit()

    def _is_database_empty(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        answer, = cur.fetchone()
        return int(answer) == 0

    def add_repository(self, repo):
        assert isinstance(repo, Repository)
        cur = self.conn.cursor()
        cur.execute(r"""
            INSERT INTO repository (owner, repo, license, revision)
            VALUES (?, ?, ?, ?);
        """, (repo.owner, repo.name, repo.license, repo.revision))
        self.conn.commit()
        return repo

    def add_source_file(self, source_file):
        assert isinstance(repo, SourceFile)
        raise NotImplementedError

    def set_failure(self, source_file):
        assert isinstance(repo, SourceFile)

    def add_parsed_source(self, parsed_source):
        assert isinstance(repo, ParsedSource)
        raise NotImplementedError

