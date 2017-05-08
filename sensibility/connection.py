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

import sys
from types import ModuleType

import redis
import sqlite3
import github3


class ConnectionModule(ModuleType):
    """
    Returns the appropriate clients.
    """
    __all__ = ['redis_client', 'sqlite3_connection', 'github']

    @property
    def redis_client(mod) -> redis.StrictRedis:
        """
        The default Redis client.
        """
        import redis
        return redis.StrictRedis(db=0)

    @property
    def sqlite3_connection(mod) -> sqlite3.Connection:
        """
        The default sqlite3 connection.
        """
        from .language import language
        import sqlite3
        return sqlite3.connect(f'sources-{language}.sqlite3')

    @property
    def github(mod) -> github3.GitHub:
        """
        The default GitHub API connection.
        """
        from github3 import login
        # Open $PWD/.token as the file containing the GitHub auth token.
        with open('.token', 'r', encoding='UTF=8') as token_file:
            github_token = token_file.read().strip()
        return login(token=github_token)

sys.modules[__name__] = ConnectionModule(__name__)
