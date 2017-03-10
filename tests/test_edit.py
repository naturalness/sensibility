#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from hypothesis import given, assume
from hypothesis.strategies import random_module, sampled_from

from strategies import programs

from sensibility import Insertion, Deletion, Substitution


@given(programs(), random_module())
def test_create_substitution(program, random):
    mutation = Substitution.create_random_mutation(program)

    # Ensure the substitution doesn't generate an identity function...
    index = mutation.index
    assert program[index] != mutation.token
    # Ensure that the substitution remembers the token it has replaced.
    assert program[index] == mutation.original_token


@given(programs(), random_module())
def test_apply_substitution(program, random):
    mutation = Substitution.create_random_mutation(program)
    mutant = program + mutation
    assert len(mutant) == len(program)

    # Ensure that all but ONE tokens are exactly the same.
    n_identical_tokens = sum(t_p == t_m for t_p, t_m in zip(program, mutant))
    assert n_identical_tokens == len(program) - 1


@given(programs(), random_module())
def test_create_deletion(program, random):
    mutation = Deletion.create_random_mutation(program)
    assert 0 <= mutation.index < len(program)
    assert program[mutation.index] == mutation.original_token


@given(programs(), random_module())
def test_apply_deletion(program, random):
    assume(len(program) > 1)
    mutation = Deletion.create_random_mutation(program)
    mutant = program + mutation
    assert len(mutant) == len(program) - 1
    # Ensure it's all the same tokens until the mutation point
    assert all(program[i] == mutant[i]
               for i in range(0, mutation.index))
    assert all(program[i + 1] == mutant[i]
               for i in range(mutation.index, len(mutant)))


@given(programs(), random_module())
def test_create_insertion(program, random):
    mutation = Insertion.create_random_mutation(program)
    assert 0 <= mutation.index <= len(program)
    # TODO: assert old token?


@given(programs(), random_module())
def test_apply_insertion(program, random):
    mutation = Insertion.create_random_mutation(program)
    mutant = program + mutation
    assert len(mutant) == len(program) + 1
    # Ensure it's all the same tokens until the mutation point
    assert all(program[i] == mutant[i]
               for i in range(0, mutation.index))
    # Ensure it's all the same program AFTER the mutation point
    assert all(program[i] == mutant[i + 1]
               for i in range(mutation.index, len(program)))


@given(programs(), sampled_from((Insertion, Deletion, Substitution)))
def test_additive_inverse(program, edit_cls):
    """
    For all edits $x$ there is an additive inverse $y such that for a program
    $p$, $p + x + y = p$.
    """
    # Deletions may only be applied on programs with more than one token.
    if edit_cls is Deletion:
        assume(len(program) > 1)

    mutation = edit_cls.create_random_mutation(program)
    mutant = program + mutation
    # The mutation ALWAYS produces a different program
    assert mutant != program
    # Applying the mutation, then the inverse returns the original program
    assert mutant + (-mutation) == program

    # Inverse of the inverse is the original mutation
    assert mutation == -(-mutation)
