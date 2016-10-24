#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Downloads metadata from the GitHub API.
"""

import logging
import zipfile
import datetime
import io

import requests

import database
from datatypes import RepositoryID, Repository, SourceFile
from rqueue import Queue, WorkQueue
from connection import redis_client, sqlite3_connection, github

QUEUE_NAME = 'q:download'
QUEUE_ERRORS = 'q:download:aborted'

logger = logging.getLogger('download_worker')


def get_repo_info(repo_id):
    url = repo_id.api_url
    logging.debug('Accessing %s', url)

    wait_for_rate_limit()
    resp = requests.get(url, headers={
        'User-Agent': 'eddieantonio-ad-hoc-miner',
        'Accept': 'application/vnd.github.drax-preview+json'
    })

    assert resp.status_code == 200
    body = resp.json()

    license = body.get('license', {}).get('key', None)
    rev = body.get('default_branch', 'master')

    return Repository(repo_id, license, rev)


def extract_js(archive):
    for filename in archive.namelist():
        if not filename.endswith('.js'):
            continue

        with archive.open(filename) as js_file:
            yield filename, js_file.read()


def download_source_files(repo):
    assert isinstance(repo, Repository)
    url = repo.id.archive_url(format="zipball", revision=repo.revision)

    wait_for_rate_limit()
    resp = requests.get(url)
    assert resp.status_code == 200

    # Open zip file
    fake_file = io.BytesIO(resp.content)
    with zipfile.ZipFile(fake_file) as repo_zip:
        # Iterate through all javascript files
        for path, source in extract_js(repo_zip):
            logger.debug("Extracted %s", path)
            yield SourceFile.create(repo, source, path)


def wait_for_rate_limit():
    limit_info = github.rate_limit()
    core = limit_info['resources']['core']
    remaining = core['remaining']
    if remaining < 10:
        # Wait an hour
        reset = core['reset']
        time.sleep(seconds_until(reset) + 1)


def seconds_until(timestamp):
    now = datetime.datetime.now()
    future = datetime.datetime.fromtimestamp(timestamp)
    difference = future - now
    return difference.seconds


def main():
    db = database.Database(sqlite3_connection)
    worker = WorkQueue(Queue(QUEUE_NAME, redis_client))
    aborted = Queue(QUEUE_ERRORS, redis_client)

    logger.info("Downloader listening on %s", QUEUE_NAME)

    while True:
        try:
            repo_name = worker.get()
        except KeyboardInterrupt:
            logging.info('Interrupted while idle (no data loss)')
            break

        repo_id = RepositoryID.parse(repo_name.decode('utf-8'))

        logger.debug('Set to download: %s', repo_id)

        try:
            repo = get_repo_info(repo_id)
            db.add_repository(repo)
            for source_file in download_source_files(repo):
                db.add_source_file(source_file)
        except KeyboardInterrupt:
            aborted << repo_id
            logger.warn("Interrupted: %s", repo_id)
            break
        except:
            aborted << repo_id
            worker.acknowledge(repo_id)
            logger.exception("Failed: %s", repo_id)
        else:
            worker.acknowledge(repo_id)
            logger.info('Downloaded: %s', repo_id)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
