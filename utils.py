#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import re
import hashlib

def sha256(contents):
    """
    >>> sha256(b'hello')
    '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    assert isinstance(contents, bytes)
    h = hashlib.new('sha256')
    h.update(contents)
    return h.hexdigest()


def is_hash(text):
    """
    >>> is_hash(sha256(b'blode'))
    True
    """
    return bool(re.match(r'^[0-9a-f]{64}$', text))
