#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2016 Eddie Antonio Santos <easantos@ualberta.ca>
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

>>> corpus = Corpus(test_corpus())
>>> len(corpus)
1
>>> raw_tokens = next(iter(corpus))
>>> token = raw_tokens[0]
>>> isinstance(token, Token)
True
>>> token.type
'Keyword'
>>> token.value
'var'
"""

import sqlite3
import json
import logging

from collections import namedtuple

from path import Path

logger = logging.Logger(__name__)
_DIRECTORY = Path(__file__).parent


class Token(namedtuple('BaseToken', 'value type loc')):
    @classmethod
    def from_json(cls, obj):
        assert isinstance(obj, dict)
        return cls(value=obj.get('value'),
                   type=obj.get('type'),
                   loc=None)


class Corpus:
    def __init__(self, connection):
        self.conn = connection

    def __iter__(self):
        cur = self.conn.cursor()
        cur.execute('''SELECT hash, tokens FROM parsed_source''')
        row = cur.fetchone()
        while row is not None:
            hash_id, blob = row
            try:
                 tokens = json.loads(blob)
            except json.decoder.JSONDecodeError:
                logging.warn("Could not parse file: %s", hash_id)
            else:
                yield [Token.from_json(raw_token) for raw_token in tokens]
        cur.close()

    def __len__(self):
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM parsed_source')
        count, = cur.fetchone()
        return int(count)


    @classmethod
    def connect_to(cls, filename):
        """
        Connects to the database (read-only) with the given filename.
        """
        filename = Path(filename).abspath()
        assert filename.exists()
        conn = sqlite3.connect('file:{}?mode=ro'.format(filename),
                               uri=True)
        return Corpus(conn)


def test_corpus():
    conn = sqlite3.connect(':memory:')
    with open(_DIRECTORY/'test.sql') as sqlfile:
        conn.executescript(sqlfile.read())
    return conn
