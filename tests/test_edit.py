#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from hypothesis import given
from hypothesis.strategies import random_module

from strategies import programs

from sensibility import Substitution


@given(programs(), random_module())
def test_substitution_creation(program, random):
    mutation = Substitution.create_random_mutation(program)
    assert isinstance(mutation, Substitution)
    index = mutation.index

    # Ensure the substitution doesn't generate an identity function...
    assert program[index] != mutation.token


@given(programs(), random_module())
def test_substitution_creation(program, random):
    mutation = Substitution.create_random_mutation(program)
    mutant = program + mutation
    assert len(mutant) == len(program)
    assert sum(t_p == t_m for t_p, t_m in zip(program, mutant)) == len(program) - 1
