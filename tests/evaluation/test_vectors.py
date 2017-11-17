#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import tempfile
from pathlib import Path

import pytest

from sensibility.evaluation.vectors import Vectors
from sensibility.source_vector import to_source_vector


def setup():
    from sensibility import current_language
    current_language.set('python')


def test_creates_file(new_vectors_path):
    """
    Create a new vector database, and test that reconnecting to it persists
    changes.
    """
    hello_vector = to_source_vector('print("hello, world!")')
    vectors = Vectors.from_filename(new_vectors_path)
    vectors['hello'] = hello_vector
    vectors.disconnect()

    vectors = Vectors.from_filename(new_vectors_path)
    assert hello_vector == vectors['hello']

    with pytest.raises(KeyError):
        vector['non-existent']


@pytest.fixture
def new_vectors_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / 'vectors.sqlite3'
