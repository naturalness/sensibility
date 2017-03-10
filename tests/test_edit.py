#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from hypothesis import given
from hypothesis.strategies import random_module, sampled_from

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
def test_substitution_application(program, random):
    mutation = Substitution.create_random_mutation(program)
    mutant = program + mutation
    assert len(mutant) == len(program)

    # Ensure that all but ONE tokens are exactly the same.
    n_identical_tokens = sum(t_p == t_m for t_p, t_m in zip(program, mutant))
    assert n_identical_tokens == len(program) - 1

"""
@given(programs(), sampled_from([Addition, Deletion, Substitution]))
def test_additive_inverse(program, edit_cls):
    mutation = edit_cls.create_random_mutation(program)
    mutant = program + mutation
    assert mutant != program
    assert mutant + (-mutation) == program
"""
