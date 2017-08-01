#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pytest
from hypothesis import given  # type: ignore
from hypothesis.strategies import floats  # type: ignore

from sensibility.utils import clamp


def test_clamp_nan() -> None:
    """
    Clamp should raise when given NaN.
    """
    from math import nan
    with pytest.raises(FloatingPointError):
        clamp(nan)


@given(floats(allow_nan=False))
def test_clamp(x: float) -> None:
    assert 0. <= clamp(x) <= 1.
