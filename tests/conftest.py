#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Pytest configuration. Adds --runslow switch to enable slow tests.
Based on: http://doc.pytest.org/en/latest/example/simple.html
"""

from functools import lru_cache

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests")
    parser.addoption("--with-javascript", action="store_true",
                     help="run JavaScript tests")


@pytest.fixture
def c():
    """
    Returns a function that tokenizes input, and returns its canoncial
    representation.
    """
    from sensibility.language import language

    @lru_cache()
    def canonical_representation(token: str) -> str:
        """
        Returns the vocabulary entry for a token (can be any valid, single token).
        """
        tokens = tuple(language.vocabularize(token))
        assert len(tokens) == 1
        return tokens[0]
    return canonical_representation


@pytest.fixture
def i(c):
    """
    Returns a function that tokenizes its input and returns the vocabulary ID.
    """
    from sensibility import Vind, current_language

    def index_of_input(text: str) -> Vind:
        return current_language.to_index(c(text))
    return index_of_input
