#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__all__ = ['DOWNLOAD_QUEUE', 'PARSE_QUEUE']


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
