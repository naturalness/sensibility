#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import tempfile
import json

from sh import node

import database
from rqueue import Queue, WorkQueue
from connection import redis_client, sqlite3_connection

def parse_js(source):
    """
    >>> tokens, ast = parse_js("void 0;")
    >>> len(tokens)
    3
    >>> isinstance(ast, dict)
    True
    """

    with tempfile.NamedTemporaryFile() as source_file:
        source_file.write(source.encode('utf-8'))
        source_file.flush()
        result_string = node('parse-js', source_file.name)
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
            db.get_source(file_hash)
            result_json = parse()
        except KeyboardInterrupt:
            aborted << file_hash
            break


if __name__ == '__main__':
    main()
