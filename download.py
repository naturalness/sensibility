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
import sqlite3
from typing import Iterator, Iterable

from sensibility.connection import github  # type: ignore

logger = logging.getLogger('search_worker')


class LanguageQuery(Iterable[str]):
    """
    Searches for the most popular repositories for the given langauge.
    """

    def __init__(self, language: str, min_stars: int=10, max_stars: int=None) -> None:
        self.language = language.lower()
        assert re.match('^\w+$', self.language)
        self.max_stars = max_stars
        self.min_stars = min_stars
        self._done = False

    def _make_query(self) -> str:
        if self.max_stars is not None:
            return (f'language:{self.language} '
                    f'stars:{self.min_stars}..{self.max_stars}')
        else:
            return (f'language:{self.language} '
                    f'stars:>={self.min_stars}')

    def __iter__(self) -> Iterator[str]:
        while not self._done:
            yield from self.more()

    def more(self) -> Iterator[str]:
        """
        Yields the next query.
        """
        query = self._make_query()
        logger.info('Issuing query: %r', query)
        result_set = list(github.search_repositories(query, sort='stars'))

        if not result_set:
            self._done = True
            return
        # Update the new upper-bound
        self.max_stars = result_set[-1].repository.stargazers - 1
        logger.info('New upper-bound: %d;', self.max_stars)

        # Finally, yield all the results.
        yield from (repo.repository.full_name for repo in result_set)

        if self.min_stars >= self.max_stars:
            self._done = True
            return


def main() -> None:
    from itertools import islice
    # TODO: take arguments: language, max results
    language = 'Python'
    for repo_name in islice(LanguageQuery('Python'), 10_000):
        print(repo_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
