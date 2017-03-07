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
Tests the round trip mapping from vocabulary string entry, to stringified
token, to
"""

import pytest

from sensibility import vocabulary
from sensibility.stringify_token import stringify_token
from sensibility.tokenize_js import id_to_token, tokenize


slow = pytest.mark.skipif(
        not pytest.config.getoption("--runslow"),
        reason="need --runslow option to run"
)


def test_javascript_vocabulary():
    """
    Tests random properties of the JavaScript vocabulary.
    """
    LENGTH = 100
    assert len(vocabulary) == LENGTH
    assert vocabulary.to_text(0) == vocabulary.start_token
    assert vocabulary.to_text(LENGTH - 1) == vocabulary.end_token


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


@slow
def test_round_trip():
    """
    This very slow test ensures that (nearly) all tokens can go from
    vocabulary entries, to their stringified text, and back.
    """

    # Iterate throught all entries EXCEPT special-cased start and end entries.
    for entry_id in range(vocabulary.start_token_index + 1, vocabulary.end_token_index):
        # Ensure that the text cooresponds to the ID and vice-versa.
        entry_text = vocabulary.to_text(entry_id)
        assert vocabulary.to_index(entry_text) == entry_id

        # HACK: This is a bug in Esprima?
        # https://github.com/jquery/esprima/issues/1772
        if entry_text in ('/', '/='):
            continue

        # These will never work being tokenized without context.
        if entry_text in ('`template-start${', '}template-middle${', '}template-tail`'):
            continue

        tokens = tokenize(entry_text)
        assert len(tokens) == 1, (
            'Unexpected number of tokens for entry {:d}: {!r}'.format(
                entry_id, entry_text
            )
        )
        # TODO: do not rely on id_to_token to make Token instances for you.
        entry_token = id_to_token(entry_id)
        assert stringify_token(entry_token) == entry_text
