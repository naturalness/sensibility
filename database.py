#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import sqlite3

from path import Path

from datatypes import Repository


logger = logging.getLogger(__name__)

SCHEMA_FILENAME = Path(__file__).parent / 'schema.sql'
with open(SCHEMA_FILENAME, encoding='UTF-8') as schema_file:
    SCHEMA = schema_file.read()
    del schema_file


class Database:
    def __init__(self, connection=None):
        if connection is None:
            self.conn = sqlite3.connect(':memory:')
        else:
            self.conn = connection

        self._initialize_db()

    def _initialize_db(self):
        raise NotImplementedError

    def add_repo(self, repo):
        assert isinstance(repo, Repository)
        raise NotImplementedError
