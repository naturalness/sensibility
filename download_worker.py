#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Downloads metadata from the GitHub API.
"""

import logging
import zipfile

import requests

import database
from datatypes import RepositoryID, Repository
from rqueue import Queue, WorkQueue
from connection import redis_client, sqlite3_connection

QUEUE_NAME = 'q:download'
QUEUE_ERRORS = 'q:download:aborted'

logger = logging.getLogger('download_worker')


def get_repo_info(repo_id):
    url = repo_id.api_url
    logging.debug('Accessing %s', url)

    resp = repo_id.get(url, headers={
        'User-Agent': 'eddieantonio-ad-hoc-miner',
        'Accept': 'application/vnd.github.drax-preview+json'
    })

    wait_for_rate_limit()

    assert resp.status_code == 200
    body = resp.json()

    license = body.get('license', {}).get('key', None)
    rev = body.get('default_branch', 'master')

    return Repository(repo_id, license, rev)


def download_source_files(repo):
    assert isinstance(repo, Repository)
    url = repo.id.archive_url(format="zipball", revision=repo.revision)
    resp = requests.get(url)
    # TODO: open zip file
    # TODO: yield SourceFile objects


def main():
    db = database(sqlite3_connection)
    worker = WorkQueue(Queue(QUEUE_NAME, redis_client))
    aborted = Queue(QUEUE_ERRORS, redis_client)

    while True:
        try:
            repo_name = worker.get()
        except KeyboardInterrupt:
            logging.info('Interrupted while idle (no data loss)')
            break

        repo_id = RepositoryID.parse(repo_name)

        logger.debug('Set to download: %s', repo_id)

        try:
            repo = get_repo_info(repo_id)
            db.add_repository(repo)
            for source_file in download_source_files(repo_id):
                db.add_source_file(source_file)
        except KeyboardInterrupt:
            aborted << file_hash
            logger.warn("Interrupted: %s", file_hash)
            break
        except Exception as ex:
            aborted << file_hash
            worker.acknowledge(file_hash)
            logger.exception("Failed: %s", file_hash)
        else:
            worker.acknowledge(file_hash)
            logger.debug('Done: %s', file_hash)


    default_branch = ...

