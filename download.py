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
from typing import Union, Any, Dict, NamedTuple

import requests
import dateutil.parser

from sensibility.miner.connection import redis_client, sqlite3_connection, github_token
from sensibility.miner.names import DOWNLOAD_QUEUE, PARSE_QUEUE
from sensibility.miner.rqueue import Queue, WorkQueue
from sensibility.miner.rate_limit import seconds_until
from sensibility.miner.repository import RepositoryID


QUEUE_ERRORS = DOWNLOAD_QUEUE.errors
logger = logging.getLogger('download_worker')


class RepositoryMetadata(NamedTuple):
    owner: str
    name: str
    revision: str
    license: str
    commit_date: datetime.datetime


class GitHubGraphQLClient:
    endpoint = "https://api.github.com/graphql"

    def __init__(self) -> None:
        ...
        self._requests_remaining = 11  # One more than min
        # todo: datetime?
        self._ratelimit_reset = 0.0

    def fetch_repository(self, owner: str, name: str) -> RepositoryMetadata:
        r"""
        Returns:
            {
              "repository": {
                "nameWithOwner": "orezpraw/MIT-Language-Modeling-Toolkit",
                "url": "https://github.com/orezpraw/MIT-Language-Modeling-Toolkit",
                "defaultBranchRef": {
                  "name": "master",
                  "target": {
                    "sha1": "267325017f60dee86caacd5b207eacdc50a3fc32",
                    "committedDate": "2017-04-05T01:56:08Z"
                  }
                },
                "license": "BSD 3-clause \"New\" or \"Revised\" License"
              }
            }
        """
        json_data = self.query(r"""
            query RepositoryInfo($owner: String!, $name: String!) {
              repository(owner: $owner, name: $name) {
                nameWithOwner
                url
                defaultBranchRef {
                  name
                  target {
                    sha1: oid
                    ... on Commit {
                      committedDate
                    }
                  }
                }
                license
              }
            }
        """, owner=owner, name=name)

        info = json_data['repository']
        owner, name = RepositoryID.parse(info['nameWithOwner'])
        latest_commit = info['defaultBranchRef']['target']

        return RepositoryMetadata(
            owner=owner,
            name=name,
            revision=latest_commit['sha1'],
            license=info['license'],
            commit_date=dateutil.parser.parse(latest_commit['committedDate'])
        )

    def query(self, query: str, **kwargs: Union[str, float, bool]) -> Dict[str, Any]:
        """
        Requests for all the relevant information.
        """

        logger.info("Performing query with vars: %r", kwargs)

        self.wait_for_rate_limit()
        resp = requests.post(self.endpoint, headers={
            'Authorization': f"bearer {github_token}",
            'Accept': 'application/json',
            'User-Agent': 'eddieantonio-ad-hoc-miner/0.2.0',
        }, json={
            "query": query,
            "variables": kwargs
        })
        resp.raise_for_status()
        self.update_rate_limit(resp)

        response = resp.json()
        # Check that there are no errors.
        if 'errors' in response and not response.get('data', False):
            raise ValueError(repr(response['errors']))

        return response['data']

    def update_rate_limit(self, resp: requests.Response) -> None:
        self._requests_remaining = int(resp.headers['X-RateLimit-Remaining'])
        self._ratelimit_reset = float(resp.headers['X-RateLimit-Reset'])
        logger.info('Updated ratelimit: %d left; reset at %s (in %s seconds)',
                    self._requests_remaining, self.ratelimit_reset,
                    self.seconds_until_reset)

    @property
    def ratelimit_reset(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self._ratelimit_reset)

    @property
    def seconds_until_reset(self) -> float:
        return seconds_until(self._ratelimit_reset)

    def wait_for_rate_limit(self) -> None:
        """
        Blocks until the rate limit is ready.
        """
        if self._requests_remaining > 10:
            return
        seconds_remaining = seconds_until(self._ratelimit_reset) + 2
        logger.info("Rate limit almost exceeded; waiting %d seconds",
                    seconds_remaining)
        time.sleep(seconds_remaining)


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


def main():
    db = Database(sqlite3_connection)
    worker = WorkQueue(Queue(DOWNLOAD_QUEUE, redis_client))
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
    client = GitHubGraphQLClient()
    data = client.fetch_repository('orezpraw', 'mit-language-modeling-toolkit')
    import pdb; pdb.set_trace()
    #main()
