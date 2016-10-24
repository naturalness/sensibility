#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Returns the appropriate clients
"""

import redis
import sqlite3
from github3 import login

__all__ = ['redis_client', 'sqlite3_connection', 'github']

# Default connection.
redis_client = redis.StrictRedis(db=0)

# Default database
sqlite3_connection = sqlite3.connect('parsed_sources.sqlite3')

# Default GitHub
with open('.token', 'r', encoding='UTF=8') as token_file:
    github = login(token=token_file.read().strip())
    del token_file
