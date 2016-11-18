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
Esprima 3.1.0 thought 0xg is a numeric literal. I beg to differ. Since the
corpus we collected included this information, this script scans the corpus
and prints out a list of invalid hashes that have been parsed from the given
database.

Usage:

./check_illegal.py | tee bad-hashes.txt
"""

from tqdm import tqdm

from parse_worker import parse_js, ParseError
from connection import sqlite3_connection



def parses_okay(source):
    try:
        parse_js(source)
    except ParseError:
        return False
    else:
        return True

def iterate(cursor):
    batch = cursor.fetchmany()
    while len(batch):
        yield from batch
        batch = cursor.fetchmany()

if __name__ == '__main__':
    cur = sqlite3_connection.execute('''
        SELECT hash, source
          FROM source_file JOIN parsed_source USING (hash)
    ''')

    progress = tqdm(iterate(cur))

    for file_hash, source in progress:
        progress.set_description(file_hash)
        if parses_okay(source):
            continue
        print(file_hash)
