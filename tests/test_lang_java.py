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


def test_vocabularize(c) -> None:
    loc = LocationFactory(Position(line=1, column=0))
    result = list(java.vocabularize_with_locations(test_file_bad))
    expected = [
        (loc.across(len("package")),            c('package')),
        (loc.space().across(len('ca')),         'IDENTIFIER'),
        (loc.across(1),                         c('.')),
        (loc.across(len('ualberta')),           'IDENTIFIER'),
        (loc.across(1),                         c('.')),
        (loc.across(len('cs')),                 'IDENTIFIER'),
        (loc.across(1),                         c('.')),
        (loc.across(len('emplab')),             'IDENTIFIER'),
        (loc.across(1),                         c('.')),
        (loc.across(len('example')),            'IDENTIFIER'),
        (loc.across(1),                         c(';')),
        (loc.next_line().next_line().across(5), c('class')),
    ]
    assert result[:len(expected)] == expected


def test_vocabulary() -> None:
    """
    Test whether every entry is  unique and source-representable.
    """
    unique_entries = set(java.vocabulary.representable_indicies())
    entries_seen = 0

    for idx in java.vocabulary.representable_indicies():
        text = java.vocabulary.to_source_text(idx)
        # What happens when we reparse the single token?
        tokens = tuple(java.vocabularize(text))
        assert len(tokens) == 1

        actual_idx = java.vocabulary.to_index(tokens[0])
        assert idx == actual_idx
        entries_seen += 1

    assert len(unique_entries) == entries_seen


def test_tokenize_invalid():
    tokens = list(java.tokenize('#'))
    assert len(tokens) == 1
    assert tokens[0].name == 'ERROR'


def test_tokenize_evil():
    # I'm learning awfull things about Java today
    # For a good time, read The Java SE Specification Section §3.3
    # https://docs.oracle.com/javase/specs/jls/se7/html/jls-3.html#jls-3.3
    # Then follow that up with The Java SE Specification Section §3.5
    # https://docs.oracle.com/javase/specs/jls/se7/html/jls-3.html#jls-3.5
    # tokens = list(java.tokenize('p. \\u0042 \\uuu003B\u001a'))
    tokens = list(java.tokenize('p. \\u0042 \\uuu003B'))
    assert 4 == len(tokens)
    assert tokens[0].value == 'p'
    assert tokens[1].value == '.'
    assert tokens[2].value == 'B'
    assert tokens[3].value == ';'
