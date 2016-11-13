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
>>> len(list(corpus))
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
        return self.iterate(skip_empty=False,
                            with_hash=False)

    def iterate(self, skip_empty=False, with_hash=False):
        """
        >>> corpus = Corpus(test_corpus())
        >>> files = tuple(corpus.iterate())
        >>> len(files)
        1
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

        """
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
                # Some pretty gross code...
                if skip_empty and len(tokens) < 1:
                    pass
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
        assert filename.exists()
        conn = sqlite3.connect('file:{}?mode=ro'.format(filename),
                               uri=True)
        return Corpus(conn)


def test_corpus():
    conn = sqlite3.connect(':memory:')
    with open(_DIRECTORY/'test.sql') as sqlfile:
        conn.executescript(sqlfile.read())
    return conn
