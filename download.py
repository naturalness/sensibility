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
Downloads metadata and source files from GitHub.

Have the Redis server running, then

    python download.py

This script will do the rest :D.
"""

import datetime
import io
import logging
import time
import zipfile
from typing import Any, Dict, Iterator, NamedTuple, Tuple, Union
from pathlib import PurePosixPath

import requests
import dateutil.parser

from sensibility.miner.connection import redis_client, sqlite3_connection, github_token
from sensibility.miner.names import DOWNLOAD_QUEUE, PARSE_QUEUE
from sensibility.miner.rqueue import Queue, WorkQueue
from sensibility.miner.rate_limit import wait_for_rate_limit, seconds_until
from sensibility.miner.models import (
    RepositoryID, RepositoryMetadata, SourceFile, SourceFileInRepository
)


QUEUE_ERRORS = DOWNLOAD_QUEUE.errors
logger = logging.getLogger('download_worker')


class GitHubGraphQLClient:
    """
    Allows fetches from GitHub's GraphQL endpoint.
    As of this writing, the endpoint is in alpha status, and has a restrictive
    rate limit.
    """
    endpoint = "https://api.github.com/graphql"

    def __init__(self) -> None:
        ...
        self._requests_remaining = 11  # One more than min
        # todo: datetime?
        self._ratelimit_reset = 0.0

    def fetch_repository(self, repo: RepositoryID) -> RepositoryMetadata:
        """
        Return RepositoryMetadata for the given owner/name.
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
        """, owner=repo.owner, name=repo.name)

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
        Issues a GraphQL query.
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



class SourceFileExtractor:
    extension: str = '.py'

    def extract_sources(self, archive) -> Iterator[Tuple[PurePosixPath, bytes]]:
        for path in archive.namelist():
            if not path.endswith(self.extension):
                continue
            with archive.open(path, mode='r') as source_file:
                yield clean_path(path), coerce_to_bytes(source_file.read())

    @staticmethod
    def zip_url_for(repo: RepositoryMetadata) -> str:
        owner = repo.owner
        name = repo.name
        revision = repo.revision
        return f"https://api.github.com/repos/{owner}/{name}/zipball/{revision}"

    def download(self, repo: RepositoryMetadata) -> Iterator[SourceFileInRepository]:
        url = self.zip_url_for(repo)

        wait_for_rate_limit()
        resp = requests.get(url, headers={
            'User-Agent': 'eddieantonio-ad-hoc-miner/0.2.0',
            'Authorization': f"token {github_token}"
        })
        resp.raise_for_status()

        # Open zip file
        fake_file = io.BytesIO(resp.content)
        with zipfile.ZipFile(fake_file) as repo_zip:
            # Iterate through all javascript files
            for path, source in self.extract_sources(repo_zip):
                yield SourceFileInRepository(repo, SourceFile(source), path)


def main() -> None:
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


def coerce_to_bytes(thing: Union[str, bytes]) -> bytes:
    return thing.encode('UTF-8') if isinstance(thing, str) else thing


def clean_path(path: str) -> PurePosixPath:
    """
    >>> clean_path('eddieantonio-bop-9884ff9/bop/__init__.py')
    PurePosixPath('bop/__init__.py')
    """
    return PurePosixPath(*PurePosixPath(path).parts[1:])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    client = GitHubGraphQLClient()
    repo = client.fetch_repository(RepositoryID.parse('eddieantonio/training-grammar-guru'))
    extractor = SourceFileExtractor()
    contents = list(extractor.download(repo))
    #main()
