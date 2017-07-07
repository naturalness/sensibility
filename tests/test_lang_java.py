#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import pytest  # type: ignore

from sensibility import Language
from sensibility.language.java import java
from sensibility import Position

from location_factory import LocationFactory


test_file_good = r"""package ca.ualberta.cs.emplab.example;

cl\u0061ss Example // This will still compile:\u000A{
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


def test_sanity_check() -> None:
    assert java.id == 'java'


def test_check_syntax():
    assert java.check_syntax(test_file_good)
    assert not java.check_syntax(test_file_bad)


def test_summarize() -> None:
    summary = java.summarize(test_file_good)
    assert summary.n_tokens == 15
    # Return the PHYSICAL number of lines of code.
    # The tokenizer may split more logical lines on \u000a escapes, but those
    # are dumb.
    assert summary.sloc == 4


@pytest.mark.skip(reason="Column numbers are wonky.")
def test_vocabularize() -> None:
    loc = LocationFactory(Position(line=1, column=0))
    result = list(java.vocabularize_with_locations(test_file_bad))
    expected = [
        (loc.across(len("package")),            'package'),
        (loc.space().across(len('ca')),         '<IDENTIFIER>'),
        (loc.space().across(1),                 '.'),
        (loc.space().across(len('ualberta')),    '<IDENTIFIER>'),
        (loc.space().across(1),                 '.'),
        (loc.space().across(len('cs')),         '<IDENTIFIER>'),
        (loc.space().across(1),                 '.'),
        (loc.space().across(len('emplab')),     '<IDENTIFIER>'),
        (loc.space().across(1),                 '.'),
        (loc.space().across(len('example')),    '<IDENTIFIER>'),
        (loc.space().across(1),                 ';'),
        (loc.next_line().next_line().across(5), 'class'),
    ]
    assert result[:len(expected)] == expected
