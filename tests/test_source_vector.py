#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pytest

from sensibility.vocabulary import OutOfVocabularyError
from sensibility.language import current_language
from sensibility.source_vector import to_source_vector


def setup():
    current_language.set('java')


def test_source_vector_unk_conversion():
    problematic_source = b'class _ { # }'
    with pytest.raises(OutOfVocabularyError):
        to_source_vector(problematic_source)

    vector = to_source_vector(problematic_source, oov_to_unk=True)
    assert 5 == len(vector)
    assert current_language.vocabulary.unk_token_index == vector[1] == vector[3]
