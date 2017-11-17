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


def test_creates_file(new_vectors_path: Path) -> None:
    """
    Create a new vector database, and test that reconnecting to it persists
    changes.
    """
    hello_vector = to_source_vector(b'print("hello, world!")')
    vectors = Vectors.from_filename(new_vectors_path)
    vectors['hello'] = hello_vector
    vectors.disconnect()

    vectors = Vectors.from_filename(new_vectors_path)
    assert hello_vector == vectors['hello']

    with pytest.raises(KeyError):
        vectors['non-existent']


def test_length_of_vectors(new_vectors_path: Path) -> None:
    """
    Create a new database, add a whole bunch of files.
    """

    examples = dict(
        file_a=to_source_vector(b'print("hello, world!")'),
        file_b=to_source_vector(b'import sys; sys.exit(0)'),
        file_c=to_source_vector(b'print(934 * 2 * 3442990 + 1)')
    )

    # Insert all the examples.
    vectors = Vectors.from_filename(new_vectors_path)
    for name, vector in examples.items():
        vectors[name] = vector
    vectors.disconnect()

    # Reopen it and test the length
    vectors = Vectors.from_filename(new_vectors_path)

    # Test fetching all of them.
    actual = sum(len(vec) for vec in examples.values())
    assert actual == vectors.length_of_vectors({'file_a', 'file_b', 'file_c'})

    # Check that we can query an empty set.
    assert 0 == vectors.length_of_vectors(())

    # Check that we can query a single item.
    assert len(examples['file_a']) == vectors.length_of_vectors({'file_a'})

    # Check that we can query a subset.
    actual = sum(len(examples[name]) for name in ('file_a', 'file_c'))
    assert actual == vectors.length_of_vectors({'file_a', 'file_c'})


@pytest.fixture
def new_vectors_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / 'vectors.sqlite3'
