#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from sensibility import Program, vocabulary


def test_program_random():
    semicolon = vocabulary.to_index(';')
    p = Program('<none>', [semicolon])
    assert len(p) == 1

    assert 0 <= p.random_token_index() < len(p)
    assert 0 <= p.random_insertion_point() <= len(p)
