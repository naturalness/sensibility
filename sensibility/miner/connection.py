#!/usr/bin/env python
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
Returns the appropriate clients
"""

import redis
import sqlite3
from github3 import login

__all__ = ['redis_client', 'sqlite3_connection', 'github', 'github_token']

# Default Redis connection.
redis_client = redis.StrictRedis(db=0)

# Database hardcoded to this file path.
sqlite3_connection = sqlite3.connect('sources.sqlite3')

# Open $PWD/.token as the file containing the GitHub auth token.
with open('.token', 'r', encoding='UTF=8') as token_file:
    github_token = token_file.read().strip()

del token_file
github = login(token=github_token)
