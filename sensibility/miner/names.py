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

char_count: <code_point>: <count>
 ~ hash of word frequencies

"""

__all__ = ['DOWNLOAD_QUEUE', 'PARSE_QUEUE', 'CHAR_COUNT']


class WithErrors(str):
    """
    >>> s = WithErrors('some:name')
    >>> s.errors
    'some:name:errors'
    """
    @property
    def errors(self):
        return "%s:errors" %(self,)


DOWNLOAD_QUEUE = WithErrors('q:download')
PARSE_QUEUE = WithErrors('q:analyze')
CHAR_COUNT = 'char_count'
