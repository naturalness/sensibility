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


import datetime
import logging
import time

from .connection import get_github_client

logger = logging.getLogger(__name__)


def wait_for_rate_limit(resource='core') -> None:
    """
    Checks the rate limit and waits if the rate limit is beyond a certain
    threshold.
    """
    assert resource in ('core', 'search')
    response = get_github_client().rate_limit()
    limit_info = response['resources'][resource]

    remaining = limit_info['remaining']
    logger.debug('%d requests remaining', remaining)

    if remaining <= 10:
        # Wait until reset
        reset = limit_info['reset']
        logger.info('Exceded rate limit; waiting until %r', reset)
        # Wait two extra seconds just to ensure.
        time.sleep(seconds_until(reset) + 2)
        wait_for_rate_limit(resource)


def seconds_until(timestamp: float) -> float:
    now = datetime.datetime.now()
    future = datetime.datetime.fromtimestamp(timestamp)
    difference = future - now
    return difference.seconds
