#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import pytest  # type: ignore

from sensibility import Language
from sensibility.language.java import java
from sensibility import Position

from location_factory import LocationFactory


test_file_good = r"""package ca.ualberta.cs.emplab.example;

class Example {
}
"""

test_file_bad = r"""package ca.ualberta.cs.emplab.example;

class Example {
    static final int NO_MORE_DOCS = -1;

    static {
        for (int i = 0; i < scorers.length; i++) {
            if (scorers[i].nextDoc() == NO_MORE_DOCS)
                lastDoc = NO_MORE_DOCS;
                return;
            }
        }
    }
}
"""


def test_check_syntax():
    assert java.check_syntax(test_file_good)
    assert not java.check_syntax(test_file_bad)
