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

# c269bfeb157c6dce747f1c786e6136de31cc6700eb73e38e81eef47e2dfc00a4
test_file_really_bad = r"""class RenameTest {
    static void fo<caret>o1(Number n) {
        System.out.println("1");
    }
    static void foo2(Long i) {
        System.out.println("2");
    }
    public static void main(String[] args) {
        long n = 0;
        foo1(n);
    }
}"""


def test_sanity_check() -> None:
    assert java.id == 'java'


def test_check_syntax():
    assert java.check_syntax(test_file_good)
    assert not java.check_syntax(test_file_bad)

    # Invalid token
    assert java.check_syntax('#') is False
    assert java.check_syntax(test_file_really_bad) is False


def test_summarize() -> None:
    summary = java.summarize(test_file_good)
    assert summary.n_tokens == 15
    # Return the PHYSICAL number of lines of code.
    # There are 4 logical lines in this example, caused by the \u000A escape.
    assert summary.sloc == 3


def test_vocabularize() -> None:
    loc = LocationFactory(Position(line=1, column=0))
    result = list(java.vocabularize_with_locations(test_file_bad))
    expected = [
        (loc.across(len("package")),            'package'),
        (loc.space().across(len('ca')),         '<IDENTIFIER>'),
        (loc.across(1),                         '.'),
        (loc.across(len('ualberta')),           '<IDENTIFIER>'),
        (loc.across(1),                         '.'),
        (loc.across(len('cs')),                 '<IDENTIFIER>'),
        (loc.across(1),                         '.'),
        (loc.across(len('emplab')),             '<IDENTIFIER>'),
        (loc.across(1),                         '.'),
        (loc.across(len('example')),            '<IDENTIFIER>'),
        (loc.across(1),                         ';'),
        (loc.next_line().next_line().across(5), 'class'),
    ]
    assert result[:len(expected)] == expected


def test_tokenize_invalid():
    tokens = list(java.tokenize('#'))
    assert len(tokens) == 1
    assert tokens[0].name == 'ERROR'
