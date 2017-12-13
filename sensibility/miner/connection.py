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
import warnings
from functools import lru_cache

import github3
import redis

from .._paths import get_sources_path

__all__ = [
    'get_github_client', 'get_github_token',
    'get_redis_client',
    'get_sqlite3_connection', 'get_sqlite3_path',
]


@lru_cache(maxsize=1)
def get_redis_client() -> redis.StrictRedis:
    """
    The default Redis client.
    """
    return redis.StrictRedis(db=0)


def get_sqlite3_path() -> str:
    """
    Path to the SQLite3 database for the active language.
    """
    # Delegate to _paths.
    return str(get_sources_path())


@lru_cache(maxsize=1)
def get_sqlite3_connection() -> sqlite3.Connection:
    """
    The SQLite3 database for the active language.
    """
    warnings.warn("Connecting to raw SQLite3 database.")
    return sqlite3.connect(get_sqlite3_path())


@lru_cache(maxsize=1)
def get_github_client() -> github3.GitHub:
    """
    The default GitHub connection.
    """
    return github3.login(token=str(get_github_token()))


@lru_cache(maxsize=1)
def get_github_token() -> str:
    """
    The GitHub token.
    """
    # Open $PWD/.token as the file containing the GitHub auth token.
    with open('.token', 'r', encoding='UTF=8') as token_file:
        return token_file.read().strip()
