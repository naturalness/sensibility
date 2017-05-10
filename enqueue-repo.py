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
from sensibility.miner.models import RepositoryID

parser = argparse.ArgumentParser()
parser.add_argument('repositories', nargs='*', type=RepositoryID.parse,
                    metavar='owner/name')


if __name__ == '__main__':
    args = parser.parse_args()
    if len(args.repositories) > 0:
         repos = args.repositories
    else:
        repos = sys.stdin.readlines()
    queue = Queue(DOWNLOAD_QUEUE, redis_client)
    for repo in repos:
        queue << str(repo)
