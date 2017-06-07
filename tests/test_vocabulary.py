#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2017 Eddie Antonio Santos <easantos@ualberta.ca>
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

"""
Tests the vocabulary classes.
"""

import pytest

from sensibility import vocabulary
from sensibility.stringify_token import stringify_token
from sensibility.tokenize_js import id_to_token, tokenize


def test_start_entry():
    assert vocabulary.start_token_index == 0
    entry_text = vocabulary.to_text(0)
    assert len(entry_text) >= 0
    assert 'start' in entry_text.lower()
    assert vocabulary.to_index(entry_text) == 0
    assert entry_text == vocabulary.start_token
    tokens = tokenize(entry_text)
    assert len(tokens) == 0, '<START> produced a token.'
    assert id_to_token(0) is None, 'Start token CANNOT have a token form!'


def test_end_entry():
    last_entry_id = len(vocabulary) -1
    assert vocabulary.end_token_index == last_entry_id
    entry_text = vocabulary.to_text(last_entry_id)
    assert len(entry_text) >= 0
    assert 'end' in entry_text.lower()
    assert vocabulary.to_index(entry_text) == last_entry_id
    assert entry_text == vocabulary.end_token
    tokens = tokenize(entry_text)
    assert len(tokens) == 0, '<END> produced a token.'
    assert id_to_token(0) is None, 'end token CANNOT have a token form!'
