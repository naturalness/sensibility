#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
enqueues respos from stdin or from command line arguments
"""

import argparse
import re
import sys
from typing import Iterator

from sensibility.miner.connection import redis_client
from sensibility.miner.names import DOWNLOAD_QUEUE
from sensibility.miner.rqueue import Queue


def repository(item: str) -> str:
    match = re.match(r'''^
        [\w\-.]+
        /
        [\w\-.]+
    $''', item, re.VERBOSE)
    if match is None:
        raise ValueError(item)
    return item


def from_stdin() -> Iterator[str]:
    yield from sys.stdin.readlines()


parser = argparse.ArgumentParser()
parser.add_argument('repositories', nargs='*', type=repository,
                    metavar='owner/name')


if __name__ == '__main__':
    args = parser.parse_args()
    if len(args.repositories) > 0:
         repos = args.repositories
    else:
        repos = from_stdin()
    queue = Queue(DOWNLOAD_QUEUE, redis_client)
    for name in repos:
        queue << name
