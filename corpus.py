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
3
>>> len(list(corpus))
3
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

from collections import namedtuple, OrderedDict
import functools

from path import Path

logger = logging.Logger(__name__)
_DIRECTORY = Path(__file__).parent


@functools.lru_cache(maxsize=256)
def _make_token(value, type_):
    """
    Caches tokens in an LRU cache, completley ignoring the location (location
    is always set to None).
    """
    return Token(value=value, type=type_, loc=None)


class Token(namedtuple('BaseToken', 'value type loc')):
    @classmethod
    def from_json(cls, obj, factory=_make_token):
        """
        Converts the Esprima JSON token into a token WITHOUT location
        information! Caches tokens to keep memoy usage down.
        """
        return factory(obj['value'], obj['type'])


class Corpus:
    def __init__(self, connection):
        self.conn = connection

    def __iter__(self):
        return self.iterate(with_hash=False)

    @property
    def first_rowid(self):
        """
        >>> Corpus(test_corpus()).first_rowid
        1
        """
        cur = self.conn.cursor()
        cur.execute('''SELECT MIN(rowid) FROM parsed_source''')
        number, = cur.fetchone()
        return number

    @property
    def last_rowid(self):
        """
        >>> corpus = Corpus(test_corpus())
        >>> corpus.last_rowid == len(corpus)
        True
        """
        cur = self.conn.cursor()
        cur.execute('''SELECT MAX(rowid) FROM parsed_source''')
        number, = cur.fetchone()
        return number

    def iterate(self, with_hash=False,
                min_rowid=None, max_rowid=None):
        """
        >>> corpus = Corpus(test_corpus())
        >>> files = tuple(corpus.iterate())
        >>> len(files)
        3

        >>> hash_, raw_tokens = next((corpus.iterate(with_hash=True)))
        >>> hash_
        'c48ebd00b8a0f8ccc10eaaffd26bf474ae8076dc9b4077fc1ba6bc6aee15d851'
        >>> token = raw_tokens[0]
        >>> isinstance(token, Token)
        True
        >>> token.type
        'Keyword'
        >>> token.value
        'var'

        >>> results = list(corpus.iterate(min_rowid=2, with_hash=True))
        >>> len(results)
        2
        >>> file_hash, _ = results[1]
        >>> file_hash
        '6d2f748e01c7a813b876ce2f3a3140048885aa2a120903882dad0c5d22756e7e'

        >>> results = list(corpus.iterate(max_rowid=2, with_hash=True))
        >>> len(results)
        2
        >>> file_hash, _ = results[1]
        >>> file_hash
        'fd30ede6650fc5f42c1aeb13e261f10eab31e8a50e8f69d461c10ee36a307b84'

        >>> it = corpus.iterate(min_rowid=2, max_rowid=2, with_hash=True)
        >>> results = list(it)
        >>> len(results)
        1
        >>> file_hash, _ = results[0]
        >>> file_hash
        'fd30ede6650fc5f42c1aeb13e261f10eab31e8a50e8f69d461c10ee36a307b84'
        """

        if min_rowid is None:
            min_rowid = self.first_rowid
        if max_rowid is None:
            max_rowid = self.last_rowid

        cur = self.conn.cursor()
        cur.execute('''
            SELECT hash, tokens
              FROM parsed_source
             WHERE rowid >= :min AND rowid <= :max
        ''', {'min': min_rowid, 'max': max_rowid})
        row = cur.fetchone()

        # INCOMING: some pretty grody code...
        while row is not None:
            hash_id, blob = row
            try:
                tokens = json.loads(blob)
            except json.decoder.JSONDecodeError:
                logger.warn("Could not parse file: %s", hash_id)
            else:
                result = tuple(Token.from_json(raw_token)
                               for raw_token in tokens)
                if with_hash:
                    yield hash_id, result
                else:
                    yield result

            row = cur.fetchone()

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
        assert filename.exists(), '%r does not exist' %(filename,)
        conn = sqlite3.connect('file:{}?mode=ro'.format(filename),
                               uri=True)
        return Corpus(conn)


def test_corpus():
    conn = sqlite3.connect(':memory:')
    with open(_DIRECTORY/'test.sql') as sqlfile:
        conn.executescript(sqlfile.read())
    return conn
