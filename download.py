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
Obtains a list of repos from GitHub's API.
"""

import re
import logging
#import redis
import sqlite3
from github3 import login
from typing import Iterator

logger = logging.getLogger('search_worker')


class LanguageQuery:
    def __init__(self, language: str, min_stars: int=10, max_stars: int=None) -> None:
        self.language = language.lower()
        assert re.match('^\w+$', self.language)
        self.max_stars = max_stars
        self.min_stars = min_stars

    def _make_query(self) -> str:
        if self.max_stars is not None:
            return (f'language:{self.language} '
                    f'stars:{self.min_stars}..{self.max_stars}')
        else:
            return (f'language:{self.language} '
                    f'stars:>={self.min_stars}')

    def more(self) -> Iterator[str]:
        """
        Yields the next query.
        """
        query = self._make_query()
        logger.info('Issuing query: %r', query)
        result_set = list(github.search_repositories(query, sort='stars'))

        if not result_set:
            raise StopIteration

        yield from (repo.repository.full_name for repo in result_set)
        self.max_stars = result_set[-1].repository.stargazers - 1
        if self.min_stars > self.max_stars:
            raise StopIteration


# Open $PWD/.token as the file containing the GitHub auth token.
with open('.token', 'r', encoding='UTF=8') as token_file:
    github_token = token_file.read().strip()

del token_file
github = login(token=github_token)
logger.info('Logging in using %s', github)


def _unused():
    # Default Redis connection.
    redis_client = redis.StrictRedis(db=0)

    # Database hardcoded to this file path.
    sqlite3_connection = sqlite3.connect('sources.sqlite3')


def main() -> None:
    language = 'Python'
    query = LanguageQuery(language)

    #for num, result in enumerate(search_iter, start=1):
    #   queue << result.full_name
    #   logger.info('[%d] Adding %s to queue...', num, result.full_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
