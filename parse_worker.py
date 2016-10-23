#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import tempfile
import json

import sh

import database
from rqueue import Queue, WorkQueue
from connection import redis_client, sqlite3_connection


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
