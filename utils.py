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


def is_sha(text):
    """
    >>> is_sha('1b8f23c763d08130ec2081c35e7f9fe0d392d700')
    True
    """
    return bool(re.match(r'^[0-9a-f]{40}$', text))
