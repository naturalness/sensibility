#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import io

from hypothesis import given
from hypothesis.strategies import builds, lists, integers, just

from sensibility import Program, vocabulary


tokens = integers(min_value=vocabulary.start_token_index + 1,
                  max_value=vocabulary.end_token_index - 1)
vectors = lists(tokens, min_size=1)
programs = builds(Program, just('<test>'), vectors)


@given(programs)
def test_program_random(program):
    assert 0 <= program.random_token_index() < len(program)
    assert 0 <= program.random_insertion_point() <= len(program)


@given(programs)
def test_program_print(program):
    with io.StringIO() as output:
        program.print(output)
        output_text = output.getvalue()
    assert len(program) == len(output_text.split())
