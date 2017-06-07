#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pytest

from sensibility.evaluation.vectors import Vectors
from sensibility.language import language

FILEHASH = 'hash'
VECTOR = b'hello, world'

@pytest.mark.skip
def test_creates_file():
    language.set_language('Python')
    # TODO: test correct file path

    vectors = Vectors()
    vectors[FILEHASH] = VECTOR
    vectors.disconnect()

    vector = Vectors()
    assert VECTOR == vector[FILEHASH]

    with pytest.raises(KeyError):
        vector['non-existent']
