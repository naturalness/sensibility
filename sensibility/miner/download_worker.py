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
Downloads metadata from the GitHub API.
"""

import datetime
import io
import logging
import time
import zipfile

import requests

#from miner_db import Database, DuplicateFileError
#from miner_db.datatypes import RepositoryID, Repository, SourceFile
#from rqueue import Queue, WorkQueue
#from connection import redis_client, sqlite3_connection, github_token
#from rate_limit import wait_for_rate_limit

#from names import DOWNLOAD_QUEUE as QUEUE_NAME, PARSE_QUEUE
#QUEUE_ERRORS = QUEUE_NAME.errors

logger = logging.getLogger('download_worker')


def get_repo_info(repo_id):
    url = repo_id.api_url
    logger.debug('Accessing %s', url)

    wait_for_rate_limit()
    resp = requests.get(url, headers={
        'User-Agent': 'eddieantonio-ad-hoc-miner',
        'Accept': 'application/vnd.github.drax-preview+json',
        'Authorization': "token %s" % (github_token,)
    })

    assert resp.status_code == 200
    body = resp.json()
    logger.debug("Headers: %r", resp.headers)
    logger.debug("Body: %r", resp.content)

    # Get the license, else assume None
    try:
        license = body['license']['key']
    except (KeyError, TypeError):
        license = None
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
    resp = requests.get(url, headers={
        'User-Agent': 'eddieantonio-ad-hoc-miner',
        'Authorization': "token %s" % (github_token,)
    })

    assert resp.status_code == 200

    # Open zip file
    fake_file = io.BytesIO(resp.content)
    with zipfile.ZipFile(fake_file) as repo_zip:
        # Iterate through all javascript files
        for path, source in extract_js(repo_zip):
            logger.debug("Extracted %s", path)
            yield SourceFile.create(repo, source, path)


def seconds_until(timestamp):
    now = datetime.datetime.now()
    future = datetime.datetime.fromtimestamp(timestamp)
    difference = future - now
    return difference.seconds


def main():
    db = Database(sqlite3_connection)
    worker = WorkQueue(Queue(QUEUE_NAME, redis_client))
    aborted = Queue(QUEUE_ERRORS, redis_client)
    parser_worker = Queue(PARSE_QUEUE, redis_client)

    logger.info("Downloader listening on %s", worker.name)

    while True:
        try:
            repo_name = worker.get()
        except KeyboardInterrupt:
            logger.info('Interrupted while idle (no data loss)')
            break

        repo_id = RepositoryID.parse(repo_name.decode('utf-8'))

        logger.debug('Set to download: %s', repo_id)

        try:
            repo = get_repo_info(repo_id)
            db.add_repository(repo)
            to_analyze = set()
            for source_file in download_source_files(repo):
                try:
                    db.add_source_file(source_file)
                except DuplicateFileError:
                    logger.info("Duplicate file: %s", source_file.path)
                else:
                    to_analyze.add(source_file.hash)

            # XXX: For some reason, we need to wait a bit before adding to the
            # queue, or else the parser will not be able to read sources.
            time.sleep(1)
            for hash_ in to_analyze:
                parser_worker << hash_
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
    logging.basicConfig(level=logging.INFO)
    main()
