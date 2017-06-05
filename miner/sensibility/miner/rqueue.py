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

import uuid
from typing import AnyStr, Iterable

import redis

from .names import WORK_QUEUE


class Queue(Iterable[bytes]):
    def __init__(self, name: str, client: redis.StrictRedis=None) -> None:
        self.name = name
        if client is None:
            self.client = redis.StrictRedis()
        else:
            self.client = client

    def enqueue(self, thing: AnyStr) -> None:
        self.client.lpush(self.name, thing)

    def pop(self) -> None:
        return self.client.rpop(self.name)

    def __lshift__(self, other: AnyStr) -> None:
        """Alias for self.enqueue(rhs); discards return."""
        self.enqueue(other)

    def __rshift__(self, other: 'Queue') -> None:
        """Alias for self.transfer(other)"""
        self.transfer(other)

    def __iter__(self):
        return iter(self.client.lrange(self.name, 0, -1))

    def clear(self) -> None:
        self.client.delete(self.name)

    def remove(self, value: AnyStr, count: int=1) -> None:
        self.client.lrem(self.name, count, value)

    def transfer(self, other: 'Queue', timeout: int=0) -> bytes:
        """Transfer one element to the other"""
        return self.client.brpoplpush(self.name, other.name, timeout)


class WorkQueue:
    def __init__(self, queue: Queue) -> None:
        self.origin = queue
        self._id = uuid.uuid4()
        self._processing = Queue(self.name, client=queue.client)

    @property
    def name(self) -> str:
        return WORK_QUEUE[self._id]

    def get(self, timeout: int=0) -> bytes:
        return self.origin.transfer(self._processing, timeout)

    def acknowledge(self, value: AnyStr) -> None:
        self._processing.remove(value)
