#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import NamedTuple, List
import io
import tempfile

import pytest

from sensibility.edit import Edit, Insertion, Deletion
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


def setup():
    from sensibility import current_language
    current_language.set('java')


def test_format_deletion(inner_missing_paren: 'File', i):
    broken_file = inner_missing_paren
    fix = Deletion(23, i('('))
    with slurp_stdout() as lines:
        format_fix(broken_file.filename, fix)

    # Check that the first line has the filename
    assert broken_file.filename.stem in lines[0]
    # TODO: assert other basic things, like correct line number


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


@contextmanager
def slurp_stdout():
    f = io.StringIO()
    lines = []
    with redirect_stdout(f):
        # This is how the result will be "returned".
        yield lines
    # Add the lines captured.
    lines.extend(f.getvalue().split('\n'))
