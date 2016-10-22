#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import re
import hashlib

def sha256(contents):
    assert isinstance(contents, bytes)
    h = hashlib.new('sha256')
    h.update(contents)
    return h.hexdigest()


def is_hash(text):
    """
    >>> is_hash(sha256(b'blode'))
    True
    """
    return bool(re.match(r'^[0-9a-f]{40}$', text))
