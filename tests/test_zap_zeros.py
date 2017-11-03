#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from hypothesis import given
from hypothesis.strategies import lists, floats, integers

from sensibility.fix import zap_zeros_inplace


@given(lists(floats(min_value=0.0, max_value=1.0,
                    allow_nan=False, allow_infinity=False)),
       integers(min_value=1, max_value=100))
def test_zap_zeros(sample_list, amount):
    original = np.array(sample_list + [0.0] * amount, dtype=np.float32)
    duplicate = original[:]

    assert 0.0 in original
    zap_zeros_inplace(duplicate)
    assert 0.0 not in duplicate
    # Thanks, floating point numbers
    assert original.sum() == duplicate.sum()
