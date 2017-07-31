#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from tempfile import NamedTemporaryFile
from pathlib import Path

import pytest

from hypothesis import given, assume  # type: ignore
from hypothesis.strategies import (  # type: ignore
    composite, integers, lists, random_module, sampled_from
)

from sensibility import SourceFile, Insertion, Deletion, Substitution
from sensibility import Vectors, Corpus
from sensibility.evaluation.evaluate import Model, Predictions
from sensibility.mutations import Mutations
from sensibility._paths import MODEL_DIR, VECTORS_PATH, SOURCES_PATH


forwards_model = MODEL_DIR / 'javascript-f0.hdf5'
backwards_model = MODEL_DIR / 'javascript-b0.hdf5'
data_does_not_exist = not (forwards_model.exists() and
                           backwards_model.exists() and
                           VECTORS_PATH.exists() and
                           SOURCES_PATH.exists())

edit_classes = Insertion, Deletion, Substitution


@composite
def source_files(draw):
    vectors = SourceFile.vectors
    index = draw(integers(min_value=vectors.min_index,
                          max_value=vectors.max_index))
    assume(exists(index))
    file_hash, vectors = vectors[index]
    assume(len(vectors) > 2)
    return SourceFile(file_hash)


def exists(index):
    "Return true when a vector with the given index exists"
    try:
        SourceFile.vectors[index]
        return True
    except:
        return False


@pytest.mark.skipif(data_does_not_exist, reason='Data does not exist')
@given(source_files(), sampled_from(edit_classes), random_module())
def test_mutations_and_predictions(source_file, edit_class, seed):
    """
    Test source code mutations, and predictions.
    """

    mutations = Mutations(Path(':memory:'))
    test_predictions = False

    # Create a random mutation and apply it.
    source_vector = source_file.vector
    mutation = edit_class.create_random_mutation(source_vector)

    # The in-memory database and all that's contained within
    # will be destoyed after this with-statement
    with mutations:
        assert len(mutations) == 0

        mutations.program = source_file
        mutations.add_mutant(mutation)

        if test_predictions:
          predictions = Predictions(0, filename=Path(':memory:'))
          # Predict on the mutated file, regardless of whether the file is
          # syntactically okay or not.
          with NamedTemporaryFile(mode='w+', encoding='UTF-8') as mutant_file:
              mutation.apply(source_vector).print(file=mutant_file)
              mutant_file.flush()
              predictions.predict(mutant_file.name)

        assert len(mutations) == 1
        stored_source_file, stored_mutation = next(iter(mutations))

    # Assert round-trip restores the original file, and the mutation
    # can be applied again and get the same result.
    assert stored_mutation == mutation
    assert source_file.file_hash == stored_source_file.file_hash
    assert (stored_source_file.vector + stored_mutation ==
            source_vector + mutation)

    # Ensure the vector has as many tokens as the original token stream.
    assert len(source_file.vector) == len(source_file.source_tokens)


def setup_module():
    SourceFile.corpus = Corpus.connect_to(SOURCES_PATH)
    SourceFile.vectors = Vectors.connect_to(VECTORS_PATH)


def teardown_module():
    SourceFile.vectors.disconnect()
