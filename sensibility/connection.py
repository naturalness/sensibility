#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
from types import ModuleType

import redis
import sqlite3
import github3


class ConnectionModule(ModuleType):
    __all__ = ['redis_client', 'sqlite3_connection', 'github']

    @property
    def redis_client(mod) -> redis.StrictRedis:
        """
        Import the default Redis client.
        """
        import redis
        return redis.StrictRedis(db=0)

    @property
    def sqlite3_connection(mod) -> sqlite3.Connection:
        """
        Import the default Redis client.
        """
        from .language import language
        import sqlite3
        return sqlite3.connect(f'sources-{language}.sqlite3')

    @property
    def github(mod) -> github3.GitHub:
        from github3 import login
        # Open $PWD/.token as the file containing the GitHub auth token.
        with open('.token', 'r', encoding='UTF=8') as token_file:
            github_token = token_file.read().strip()
        return login(token=github_token)

sys.modules[__name__] = ConnectionModule(__name__)
