#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""

First, ensure the Redis server is running, and clear it.

>>> client = redis.StrictRedis()
>>> client.flushdb()
True

Okay, now we can rock.

>>> q = Queue('foo', client)
>>> from datatypes import RepositoryID
>>> q << RepositoryID('eddieantonio', 'bop')
>>> list(q)
[b'eddieantonio/bop']

>>> q << "<sentinel>"
>>> line = Queue('bar', client)
>>> q >> line
>>> list(q)
[b'<sentinel>']
>>> list(line)
[b'eddieantonio/bop']

"""

import redis


class Queue:
    def __init__(self, name, client=None):
        self.name = name
        if client is None:
            self.client = redis.StrictRedis()
        else:
            assert isinstance(client, redis.StrictRedis)
            self.client = client

    def enqueue(self, thing):
        return self.client.lpush(self.name, str(thing))

    def pop(self):
        return self.client.rpop(self.name)

    def __lshift__(self, other):
        """Alias for self.enqueue(rhs); discards return."""
        self.enqueue(other)

    def __rshift__(self, other):
        """Alias for self.transfer(other)"""
        self.transfer(other)

    def __iter__(self):
        return iter(self.client.lrange(self.name, 0, -1))

    def transfer(self, other):
        """Transfer one element to the other"""
        assert isinstance(other, Queue)
        self.client.rpoplpush(self.name, other.name)
