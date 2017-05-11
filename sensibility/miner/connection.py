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
Proxies to the common connections.
"""

import sqlite3
import sys
from types import ModuleType

import redis
import github3
from lazy_object_proxy import Proxy

from ..language import language


__all__ = [
    'github', 'github_token',
    'redis_client',
    'sqlite3_connection', 'sqlite3_path',
]


redis_client = Proxy(lambda: redis.StrictRedis(db=0))
"""
The default Redis client.
"""

sqlite3_path = Proxy(lambda: f'sources-{language}.sqlite3')
"""
The SQLite3 database for given language
"""

sqlite3_connection = Proxy(lambda: sqlite3.connect(str(sqlite3_path)))
"""
The default sqlite3 connection.
"""

github = Proxy(lambda: github3.login(token=str(github_token)))
"""
The default GitHub connection.
"""

@Proxy
def github_token() -> str:
    """
    The GitHub token.
    """
    # Open $PWD/.token as the file containing the GitHub auth token.
    with open('.token', 'r', encoding='UTF=8') as token_file:
        return token_file.read().strip()
