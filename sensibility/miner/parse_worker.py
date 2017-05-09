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
Listens to updates on q:analyze, and inserts them into the database.
"""

import json
import logging
import tempfile
from collections import Counter

#import sh
#from sh import node

#from miner_db import Database
#from miner_db.datatypes import ParsedSource
#from rqueue import Queue, WorkQueue
#from connection import redis_client, sqlite3_connection

#from names import PARSE_QUEUE as QUEUE_NAME, CHAR_COUNT
#QUEUE_ERRORS = QUEUE_NAME.errors

logger = logging.getLogger('parse_worker')


def main():
    db = Database(sqlite3_connection)
    worker = WorkQueue(Queue(QUEUE_NAME, redis_client))
    aborted = Queue(QUEUE_ERRORS, redis_client)

    logger.info("Parser listening on %s", QUEUE_NAME)

    while True:
        try:
            file_hash = worker.get()
        except KeyboardInterrupt:
            logging.info('Interrupted while idle (no data loss)')
            break

        file_hash = file_hash.decode('utf-8')

        logger.debug('Pulled: %s', file_hash)

        try:
            source_code = db.get_source(file_hash)
            tokens, ast = parse_js(source_code)
            db.add_parsed_source(ParsedSource(file_hash, tokens, ast))
            insert_count(count_codepoints_in_literals(tokens))
        except KeyboardInterrupt:
            aborted << file_hash
            logger.warn("Interrupted: %s", file_hash)
            break
        except ParseError:
            db.set_failure(file_hash)
            worker.acknowledge(file_hash)
            logger.info("Syntax error in %s", file_hash)
        except:
            aborted << file_hash
            worker.acknowledge(file_hash)
            logger.exception("Failed: %s", file_hash)
        else:
            worker.acknowledge(file_hash)
            logger.info('Analyzed: %s', file_hash)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
