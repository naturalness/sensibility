#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import io

from hypothesis import given
from hypothesis.strategies import lists, integers

from sensibility import Program, vocabulary
#semicolon = vocabulary.to_index(';')


@given(lists(integers(min_value=vocabulary.start_token_index + 1,
                      max_value=vocabulary.end_token_index - 1),
             min_size=1))
def test_program_random(tokens):
    p = Program('<none>', tokens)
    assert 0 <= p.random_token_index() < len(p)
    assert 0 <= p.random_insertion_point() <= len(p)


@given(lists(integers(min_value=vocabulary.start_token_index + 1,
                      max_value=vocabulary.end_token_index - 1),
             min_size=1))
def test_program_print(tokens):
    program = Program('<none>', tokens)
    with io.StringIO() as output:
        program.print(output)
        output_text = output.getvalue()
    assert len(program) >= 1
    assert len(program) == len(output_text.split())
