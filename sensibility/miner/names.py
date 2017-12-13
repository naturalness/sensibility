#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Names for all of the Redis queues.

q:download: <repo/owner>
 ~ download, extract, and insert JavaScript files

q:analyze: <sha256 of file>
 ~ requires syntactic and lexical analysis

q:work:[uuid]: <[data]>
 ~ work queue for a process
"""

from uuid import UUID

__all__ = ['DOWNLOAD_QUEUE', 'PARSE_QUEUE', 'WORK_QUEUE']


class WithErrors(str):
    """
    Adds errors
    >>> s = WithErrors('some:name')
    >>> s.errors
    'some:name:errors'
    """
    @property
    def errors(self) -> str:
        return f"{self}:errors"


class WorkQueueName:
    """
    >>> uuid = UUID('{12345678-1234-5678-1234-567812345678}')
    >>> WORK_QUEUE[uuid]
    'q:work:12345678-1234-5678-1234-567812345678'
    """
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __getitem__(self, queue_id: UUID) -> str:
        return f"{self.prefix}:{queue_id!s}"


DOWNLOAD_QUEUE = WithErrors('q:download')
PARSE_QUEUE = WithErrors('q:analyze')
WORK_QUEUE = WorkQueueName('q:work')
