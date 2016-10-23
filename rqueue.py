#!/usr/bin/env python
# -*- coding: UTF-8 -*-

r"""

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

>>> q.clear()
>>> q << "hello"
>>> worker = WorkQueue(q)
>>> name = worker.name
>>> import re; bool(re.match(r'^[0-9a-f\-]{20,}$', name))
True
>>> worker.get()
b'hello'
>>> worker.get(timeout=1) is None
True

"""

import uuid

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
        if isinstance(thing, (str, bytes)):
            serialized = thing
        else:
            serialized = str(thing)
        return self.client.lpush(self.name, thing)

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

    def clear(self):
        self.client.delete(self.name)

    def transfer(self, other, timeout=0):
        """Transfer one element to the other"""
        assert isinstance(other, Queue)
        return self.client.brpoplpush(self.name, other.name, timeout)


class WorkQueue:
    def __init__(self, queue):
        assert isinstance(queue, Queue)
        self.origin = queue
        self._processing = Queue(str(uuid.uuid4()),
                                client=queue.client)
    @property
    def name(self):
        return self._processing.name

    def get(self, timeout=0):
        return self.origin.transfer(self._processing, timeout)
