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

import logging

from rqueue import Queue
from connection import github, redis_client
from names import DOWNLOAD_QUEUE

logger = logging.getLogger(__name__)


def main():
    queue = Queue(DOWNLOAD_QUEUE, redis_client)

    search_iter = github.search_repositories(
        'language:javascript stars:>=10',
        sort='stars'
    )

    logger.info('Starting iteration...')

    for num, result in enumerate(search_iter, start=1):
        queue << result.full_name
        logger.info('[%d] Adding %s to queue...', num, result.full_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
