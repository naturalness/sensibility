#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

r"""

First, ensure the Redis server is running, and clear it.
"""

import re

import redis
import pytest  # type: ignore

from sensibility.miner.rqueue import Queue, WorkQueue


def redis_running():
    try:
        redis.StrictRedis().ping()
    except redis.exceptions.ConnectionError:
        return False
    return True


with_redis = pytest.mark.skipif(not redis_running(),
                                reason='redis is not running')

@pytest.fixture
def redis_client():
    client = redis.StrictRedis(db=1)
    assert client.flushdb()
    return client


@with_redis
def test_base_queue(redis_client):
    foo = Queue('foo', redis_client)
    foo << 'eddieantonio/bop'
    assert list(foo) == [b'eddieantonio/bop']

    bar = Queue('bar', redis_client)
    foo << "<sentinel>"
    foo >> bar
    assert list(foo) == [b'<sentinel>']
    assert list(bar) == [b'eddieantonio/bop']


@with_redis
def test_work_queue(redis_client):
    q = Queue('foo', redis_client)
    q << "hello"

    worker = WorkQueue(q)
    assert re.match(r'^q:worker:[0-9a-f\-]{20,}$', worker.name)
    assert worker.get() == b'hello'

    worker.acknowledge(b'hello')
    assert worker.get(timeout=1) is None
