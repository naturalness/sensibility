#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2017 Eddie Antonio Santos <easantos@ualberta.ca>
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
Contains the Downloader class.

TODO: use GHTorrent instead of GitHub's alpha GraphQL API.
"""

import datetime
import io
import logging
import time
import zipfile
from pathlib import PurePosixPath
from typing import Any, Dict, Iterator, Tuple, Union, Optional

import dateutil.parser
import requests

from sensibility.language import language

from .connection import get_github_token, get_redis_client
from .corpus import Corpus, NewCorpusError
from .models import (RepositoryID, RepositoryMetadata, SourceFile,
                     SourceFileInRepository)
from .names import DOWNLOAD_QUEUE
from .rate_limit import seconds_until, wait_for_rate_limit
from .rqueue import Queue, WorkQueue

QUEUE_ERRORS = DOWNLOAD_QUEUE.errors
logger = logging.getLogger('download_worker')


class Downloader:
    def __init__(self) -> None:
        redis_client = get_redis_client()
        self.client = GitHubGraphQLClient()
        self.worker = WorkQueue(Queue(DOWNLOAD_QUEUE, redis_client))
        self.errors = Queue(QUEUE_ERRORS, redis_client)
        self.corpus = Corpus(writable=True)
        self._headers = {
            'User-Agent': 'eddieantonio-sensibility/0.3.0',
            'Authorization': f"token {get_github_token()}"
        }

        # Ensure the corpus is initialized.
        # If it's initialized with a different language, bail.
        try:
            lang_name = self.corpus.language
        except NewCorpusError:
            import datetime
            now = datetime.datetime.utcnow()
            self.corpus.set_metadata(language=language.name, mined=now)
        else:
            assert lang_name == language.name, (
                f"Refusing to overwrite {lang_name} corpus"
            )

    def loop_forever(self) -> None:
        logger.info("Downloader queue: %s", self.worker.name)
        while True:
            job = self.get_a_job()
            try:
                self.do_job(job)
            except Exception:
                # "You had ONE job!"
                self.log_error(job)
            finally:
                self.acknowledge(job)

    def get_a_job(self) -> str:
        """
        This will block until a job is available.
        (to place a job, use bin/enqueue-job)
        """
        # TODO: if there already is a job in the worker queue, then, fetch
        # this job.
        return self.worker.get().decode('UTF-8')

    def acknowledge(self, job: str) -> None:
        """
        This must be done before getting another job.
        """
        return self.worker.acknowledge(job)

    def do_job(self, job: str) -> None:
        repo_id = RepositoryID.parse(job)

        logger.info('Fetching %s', repo_id)
        repo = self.client.fetch_repository(repo_id)
        self.insert_repository(repo)

        for source_file in self.download(repo):
            self.insert_source_file(source_file)

    def download(self, repo: RepositoryMetadata) -> Iterator[SourceFileInRepository]:
        """
        Downloads files for the given repository revision.
        """
        url = self.zip_url_for(repo)

        logger.debug('Downloading %s', url)
        wait_for_rate_limit()
        resp = requests.get(url, headers=self._headers)
        resp.raise_for_status()

        # Open zip file
        fake_file = io.BytesIO(resp.content)
        with zipfile.ZipFile(fake_file) as repo_zip:
            # Iterate through all javascript files
            for path, source in self.extract_sources(repo_zip):
                yield SourceFileInRepository(repo, SourceFile(source), path)

    def log_error(self, job: str) -> None:
        logger.exception('Error downloading "%s"', job)
        self.errors << job

    def insert_repository(self, repo: RepositoryMetadata) -> None:
        logger.debug('Insering %s', repo)
        self.corpus.insert_repository(repo)

    def insert_source_file(self, entry: SourceFileInRepository) -> None:
        logger.debug('  > %s', entry.path)
        self.corpus.insert_source_file_from_repo(entry)

    def extract_sources(self, archive: zipfile.ZipFile) -> Iterator[Tuple[PurePosixPath, bytes]]:
        """
        Extracts sources (with the given extension) from a zip file.
        """
        for path in archive.namelist():
            if not language.matches_extension(path):
                continue
            with archive.open(path, mode='r') as source_file:
                yield clean_path(path), coerce_to_bytes(source_file.read())

    @staticmethod
    def zip_url_for(repo: RepositoryMetadata) -> str:
        return (f"https://api.github.com/repos/{repo.owner}/{repo.name}"
                f"/zipball/{repo.revision}")


class GitHubGraphQLClient:
    """
    Allows fetches from GitHub's GraphQL endpoint.
    As of this writing, the endpoint is in alpha status, and has a restrictive
    rate limit.
    """
    endpoint = "https://api.github.com/graphql"

    def __init__(self) -> None:
        self._requests_remaining = 11  # One more than min
        # todo: datetime?
        self._ratelimit_reset = 0.0
        self._headers = {
            'Authorization': f"bearer {get_github_token()}",
            'Accept': 'application/json',
            'User-Agent': 'eddieantonio-sensibility/0.3.0'
        }

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
                licenseInfo {
                  name
                }
              }
            }
        """, owner=repo.owner, name=repo.name)

        info = json_data['repository']
        if info is None:
            raise ValueError(f'Could not fetch info for {repo}')
        owner, name = RepositoryID.parse(info['nameWithOwner'])
        latest_commit = info['defaultBranchRef']['target']
        license_name: Optional[str] = None
        if info['licenseInfo'] is not None:
            # as per https://developer.github.com/v4/object/license/
            license_name = info['licenseInfo']['name']

        return RepositoryMetadata(
            owner=owner,
            name=name,
            revision=latest_commit['sha1'],
            license=license_name or '',
            commit_date=dateutil.parser.parse(latest_commit['committedDate'])
        )

    def query(self, query: str, **variables: Union[str, float, bool]) -> Dict[str, Any]:
        """
        Issues a GraphQL query.
        """

        logger.debug("Performing query with vars: %r", variables)

        self.wait_for_rate_limit()
        resp = requests.post(self.endpoint, headers=self._headers, json={
            "query": query,
            "variables": variables
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
        logger.debug('Updated ratelimit: %d left; reset at %s (in %s seconds)',
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


def clean_path(path: str) -> PurePosixPath:
    """
    Cleans paths from zip files.
    >>> clean_path('eddieantonio-bop-9884ff9/bop/__init__.py')
    PurePosixPath('bop/__init__.py')
    """
    return PurePosixPath(*PurePosixPath(path).parts[1:])


def coerce_to_bytes(thing: Union[str, bytes]) -> bytes:
    """
    Ensure whatever is passed in is a bytes object.
    """
    return thing.encode('UTF-8') if isinstance(thing, str) else thing
