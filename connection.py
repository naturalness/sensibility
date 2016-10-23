#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Returns the appropriate clients
"""

import redis
import sqlite3

__all__ = ['redis_client', 'sqlite3_connection']

# Default connection.
redis_client = redis.StrictRedis(db=0)

# Default database
sqlite3_connection = sqlite3.connect('parsed_sources.sqlite3')
