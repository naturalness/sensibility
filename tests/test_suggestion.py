#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import NamedTuple, List
import io
import tempfile

import pytest

from sensibility.edit import Edit, Insertion, Deletion, Substitution
from sensibility.format_fix import format_fix


inner_missing_paren_source = ("""package ca.ualberta.cs.example;

public class Foo {
    public boolean isBar(Object obj) {
        if ((obj instanceof Bar) {
            return true;
        }
        return false;
    }
}
""").encode('UTF-8')

inner_bad_keyword_source = ("""package ca.ualberta.cs.example;

public class Foo {
    public boolean isBar(Object obj) {
        If (obj instanceof Bar) {
            return true;
        }
        return false;
    }
}
""").encode('UTF-8')

# TODO: make tests for regressions here.
"""public class A {
    public static void main(String args[]) {
        if (args.length 2) {
            System.out.println("Not enough args!");
            System.exit(1);
        }
        System.out.println("Hello, world!");
    }
}
"""


def setup():
    from sensibility import current_language
    current_language.set('java')


def test_format_deletion(inner_missing_paren: 'File', i):
    broken_file = inner_missing_paren
    fix = Deletion(23, i('('))
    with slurp_stdout() as lines:
        format_fix(broken_file.filename, fix)

    # The error message format has exactly four lines
    assert len(lines) == 4

    # Check the formatting of the first line:
    filename, line, column, message = lines[0].split(':', 3)
    assert filename.endswith(broken_file.filename.name)
    assert line == '5'
    assert column == '13'
    assert message.lstrip().startswith('try removing')

    # Check that the second line came from the file:
    assert broken_file.lines[4] in lines[1]
    # Check that the fourth line has the deletion token
    assert '(' in lines[-1]

    # TODO: more robust tests
    # TODO: check that the caret is in the right place.


def test_format_insertion(inner_missing_paren: 'File', i):
    broken_file = inner_missing_paren
    fix = Insertion(27, i(')'))
    with slurp_stdout() as lines:
        format_fix(broken_file.filename, fix)

    # The error message format has exactly four lines
    assert len(lines) == 4

    # Check the formatting of the first line:
    filename, line, column, message = lines[0].split(':', 3)
    assert filename.endswith(broken_file.filename.name)
    assert line == '5'
    assert column == '32'
    assert message.lstrip().startswith('try inserting')

    # Check that the second line came from the file:
    assert ''.join(broken_file.lines[4].split()) == ''.join(lines[1].split())
    # Check that the fourth line has the insertion token
    assert ')' in lines[-1]

    # TODO: more robust tests
    # TODO: check that the caret is in the right place.


def test_format_replacement(inner_bad_keyword: 'File', i):
    broken_file = inner_bad_keyword
    fix = Substitution(21, original_token=i('If'), replacement=i('if'))
    with slurp_stdout() as lines:
        format_fix(broken_file.filename, fix)

    # The error message format has exactly four lines
    assert len(lines) == 4

    # Check the formatting of the first line:
    filename, line, column, message = lines[0].split(':', 3)
    assert filename.endswith(broken_file.filename.name)
    assert line == '5'
    assert column == '9'
    assert message.lstrip().startswith('try replacing')

    # Check that the second line came from the file:
    assert ''.join(broken_file.lines[4].split()) == ''.join(lines[1].split())
    # Check that the fourth line has the insertion token
    assert 'if' in lines[-1]

    # TODO: more robust tests
    # TODO: check that the caret is in the right place.


class File(NamedTuple):
    filename: Path
    source: bytes

    @property
    def lines(self) -> List[str]:
        return self.filename.read_text().split('\n')


@pytest.fixture
def inner_missing_paren():
    with tempfile.TemporaryDirectory() as dirname:
        dirpath = Path(dirname)
        filename = dirpath / 'Foo.java'
        filename.write_bytes(inner_missing_paren_source)
        yield File(filename, inner_missing_paren_source)


@pytest.fixture
def inner_bad_keyword():
    with tempfile.TemporaryDirectory() as dirname:
        dirpath = Path(dirname)
        filename = dirpath / 'Foo.java'
        filename.write_bytes(inner_bad_keyword_source)
        yield File(filename, inner_bad_keyword_source)


@contextmanager
def slurp_stdout():
    f = io.StringIO()
    lines = []
    with redirect_stdout(f):
        # This is how the result will be "returned".
        yield lines
    # Add the lines captured.
    lines.extend(f.getvalue().split('\n'))

    # The last line is usually empty, so remove it.
    if f.getvalue()[-1] == '\n' and lines[-1] == '':
        lines.pop()
