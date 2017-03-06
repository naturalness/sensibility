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
Tests the generation of sentences.
"""

import pytest

from sensibility import Token, serialize_tokens
from sensibility.sentences import forward_sentences, backward_sentences
from sensibility.vocabulary import vocabulary


FILE = serialize_tokens([
    Token(value='(', type='Punctuator', loc=None),
    Token(value='name', type='Identifier', loc=None),
    Token(value=')', type='Punctuator', loc=None),
    Token(value='=>', type='Punctuator', loc=None),
    Token(value='console', type='Identifier', loc=None),
    Token(value='.', type='Punctuator', loc=None),
    Token(value='log', type='Identifier', loc=None),
    Token(value='(', type='Punctuator', loc=None),
    Token(value='`Hello, ${', type='Template', loc=None),
    Token(value='name', type='Identifier', loc=None),
    Token(value='}!`', type='Template', loc=None),
    Token(value=')', type='Punctuator', loc=None),
    Token(value=';', type='Punctuator', loc=None)
])

assert len(FILE) == 13


def test_forward_sentences():
    """
    Test creatign padded forward sentences.
    """
    n = 10  # sentence length.
    m = n - 1  # context length.

    sentences = list(forward_sentences(FILE, context=m, adjacent=1))

    # Even with padding, there should be the same number of sentences as there
    # are tokens in the original vector.
    assert len(sentences) == len(FILE)

    # Test each sentence generated.
    for i, (context, adjacent) in enumerate(sentences):
        assert len(context) == m
        assert adjacent == FILE[i]

    # The first context should be a context with all padding.
    context, adjacent = sentences[0]
    assert all(index == vocabulary.start_token_index for index in context)

    # Try using ONLY sentence length. Should get the same result.
    assert list(forward_sentences(FILE, sentence=n)) == sentences


def test_forward_sentences_too_big():
    """
    test for when sentence size is LARGER than file
    """
    n = 20
    sentences = list(forward_sentences(FILE, sentence=n))

    # There should be the same number of sentences as tokens.
    assert len(sentences) == len(FILE)

    # The first context should be a context with all padding.
    context, adjacent = sentences[0]
    assert adjacent == FILE[0]
    assert len(context) == n -1
    assert all(index == vocabulary.start_token_index for index in context)

    # Check the last sentence
    context, adjacent = sentences[-1]
    assert adjacent == FILE[-1]
    # It should still have padding!
    padding = context[:-len(FILE) - 1]
    assert len(padding) > 0
    assert all(index == vocabulary.start_token_index for index in padding)


def test_backward_sentences():
    """
    Test creatign padded backwards sentences.
    """
    n = 10  # sentence length.
    m = n - 1  # context length.

    sentences = list(backward_sentences(FILE, context=m, adjacent=1))

    # Even with padding, there should be the same number of sentences as there
    # are tokens in the original vector.
    assert len(sentences) == len(FILE)

    # Test each sentence generated.
    for i, (context, adjacent) in enumerate(sentences):
        assert adjacent == FILE[i]
        assert len(context) == m, str(i) + ': ' + vocabulary.to_text(adjacent)

    # The first context should be all NON padding!
    context, adjacent = sentences[0]
    assert all(index != vocabulary.end_token_index for index in context)

    # The last context should be a context with all padding.
    context, adjacent = sentences[-1]
    assert all(index == vocabulary.end_token_index for index in context)

    # Try using ONLY sentence length. Should get the same result.
    assert list(backward_sentences(FILE, sentence=n)) == sentences


def test_both_sentences():
    args = (FILE,)
    kwargs = dict(sentence=10)
    combined = zip(forward_sentences(*args, **kwargs),
                   backward_sentences(*args, **kwargs))

    # Check if both adjacents are THE SAME.
    for (_, t1), (_, t2) in combined:
        assert t1 == t2
