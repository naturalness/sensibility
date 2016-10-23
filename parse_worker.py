#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Listens to updates on q:analyze, and inserts them into the database.
"""

import json
import logging
import tempfile

import sh
from sh import node

import database
from datatypes import ParsedSource
from rqueue import Queue, WorkQueue
from connection import redis_client, sqlite3_connection

# TODO: count characters in string literals!
logger = logging.getLogger('parse_worker')

class ParseError(Exception):
    def __init__(self):
        super(ParseError, self).__init__('failed to parse file')


def parse_js(source):
    """
    >>> tokens, ast = parse_js("void 0;")
    >>> len(tokens)
    3
    >>> isinstance(ast, dict)
    True
    >>> parse_js("%$")
    Traceback (most recent call last):
    ...
    parse_worker.ParseError: failed to parse file
    """

    with tempfile.NamedTemporaryFile() as source_file:
        source_file.write(source.encode('utf-8'))
        source_file.flush()
        try:
            result_string = node('parse-js', source_file.name)
        except sh.ErrorReturnCode_1:
            raise ParseError
    result = json.loads(str(result_string))
    return result['tokens'], result['ast']


def main():
    db = database(sqlite3_connection)
    worker = WorkQueue(Queue('q:analyze'))
    aborted = Queue('q:analyze:aborted')

    while True:
        try:
            file_hash = worker.get()
        except KeyboardInterrupt:
            break

        try:
            source_code = db.get_source(file_hash)
            tokens, ast = parse_js(source_code)
            db.parsed_source(ParsedSource(file_hash, tokens, ast))
        except KeyboardInterrupt:
            aborted << file_hash
            logger.warn("Interrupted: %s", file_hash)
            break
        except ParseError:
            db.set_failure(file_hash)
            worker.acknowledge(file_hash)
            logger.info("Failed: %s", file_hash)
        else:
            worker.acknowledge(file_hash)
            logger.info('Done: %s', file_hash)


if __name__ == '__main__':
    main()
