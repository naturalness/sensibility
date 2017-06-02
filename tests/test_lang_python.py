#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pytest  # type: ignore

from sensibility.language import Language
from sensibility.language.python import python
from sensibility.token_utils import Location
from sensibility.token_utils import Position

from location_factory import LocationFactory

test_file = r"""#!/usr/bin/env python
# coding: us-ascii

'''
This is an example file.
'''

from __future__ import print_function


if __name__ == '__main__':
    print(f"This is the documentation: {__doc__}")
"""

def test_sanity_check() -> None:
    assert isinstance(python, Language)


def test_tokenize():
    tokens = python.tokenize(test_file)
    # TODO: more robust tests for this.
    assert len(tokens) == 30


def test_summarize() -> None:
    with pytest.raises(SyntaxError):
        python.summarize('import $')

    summary = python.summarize(test_file)
    assert summary.sloc == 6
    assert summary.n_tokens == 20


def test_pipeline() -> None:
    loc = LocationFactory(Position(line=4, column=0))
    result = list(python.pipeline.execute_with_locations(test_file))
    assert result[:7] == [
        (loc.until(Position(line=6, column=3)),     'STRING'),
        (loc.newline(),                             'NEWLINE'),
        (loc.next_line().across(len("from")),       'from'),
        (loc.space().across(len("__future__")),     'IDENTIFIER'),
        (loc.space().across(len("import")),         'import'),
        (loc.space().across(len("print_function")), 'IDENTIFIER'),
        (loc.newline(),                             'NEWLINE'),
    ]
    # TODO: Test more locations?
