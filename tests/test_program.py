#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from hypothesis import given
from hypothesis.strategies import lists, integers

from sensibility import Program, vocabulary
#semicolon = vocabulary.to_index(';')


@given(lists(integers(min_value=vocabulary.start_token_index,
                      max_value=vocabulary.end_token_index),
             min_size=1))
def test_program_random(tokens):
    p = Program('<none>', tokens)
    assert 0 <= p.random_token_index() < len(p)
    assert 0 <= p.random_insertion_point() <= len(p)
