#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pytest
import shutil

from sensibility import Language
from sensibility.language.javascript import javascript
from sensibility import Position

from location_factory import LocationFactory


# Only run tests in this module if Node.js is installed.
pytestmark = pytest.mark.skipif(
    not (shutil.which('node') or shutil.which('nodejs')),
    reason="Requires Node.JS"
)

# Use this as as decorator to mark slow tests.
slow = pytest.mark.skipif(
        not pytest.config.getoption("--runslow"),
        reason="need --runslow option to run"
)

test_file = r"""#!/usr/bin/env node
/*!
 * This is an example file.
 */

import {ಠ_ಠ} from "-_-";

/* TODO: crazy ES2017 features. */
"""


def test_sanity_check() -> None:
    assert isinstance(javascript, Language)


def test_tokenize() -> None:
    tokens = javascript.tokenize(test_file)
    # TODO: more robust tests for this.
    assert len(tokens) == 7
    assert tokens[2].value == 'ಠ_ಠ'


def test_check_syntax() -> None:
    assert not javascript.check_syntax('import #')
    assert javascript.check_syntax(test_file)


def test_summarize() -> None:
    summary = javascript.summarize(test_file)
    assert summary.sloc == 1
    assert summary.n_tokens == 7


def test_vocabularize() -> None:
    loc = LocationFactory(Position(line=6, column=0))
    result = list(javascript.vocabularize_with_locations(test_file))
    expected = [
        (loc.across(len("import")),         'import'),
        (loc.space().across(1),             '{'),
        (loc.across(len("ಠ_ಠ")),            '<IDENTIFIER>'),
        (loc.across(1),                     '}'),

        # XXX: Esprima reports that `from` is an identifier, even though in my
        # opinion, it's a keyword, but the fix is far too difficult to
        # implement right now.
        #
        # "from" is a **contextual keyword**, so scanners usually treat them as
        # identifiers.
        # See: Parser#matchContextualKeyword() in
        # https://github.com/jquery/esprima/blob/master/src/parser.ts
        (loc.space().across(len("from")),   '<IDENTIFIER>'),
        # (loc.space().across(len("from")),   'from'),

        (loc.space().across(len("'-_-'")),  '<STRING>'),
        (loc.across(1),                    ';'),
    ]
    assert result[:len(expected)] == expected


def test_javascript_vocabulary():
    """
    Tests random properties of the JavaScript vocabulary.
    """
    vocabulary = javascript.vocabulary
    LENGTH = 101  # includes <UNK>, <s>, </s>
    assert len(vocabulary) == LENGTH
    assert vocabulary.to_text(0) == vocabulary.unk_token
    assert vocabulary.to_text(1) == vocabulary.start_token
    assert vocabulary.to_text(2) == vocabulary.end_token


@slow
def test_round_trip():
    """
    This very slow test ensures that (nearly) all tokens can go from
    vocabulary entries, to their stringified text, and back.
    """
    # TODO: rewrite this for new vocabulary or remove

    # Iterate throught all entries EXCEPT special-cased start and end entries.
    for entry_id in range(vocabulary.start_token_index + 1, vocabulary.end_token_index):
        # Ensure that the text cooresponds to the ID and vice-versa.
        entry_text = vocabulary.to_text(entry_id)
        assert vocabulary.to_index(entry_text) == entry_id

        # HACK: This is a bug in Esprima?
        # https://github.com/jquery/esprima/issues/1772
        if entry_text in ('/', '/='):
            continue

        # These will never work being tokenized without context.
        if entry_text in ('`template-start${', '}template-middle${', '}template-tail`'):
            continue

        tokens = tokenize(entry_text)
        assert len(tokens) == 1, (
            'Unexpected number of tokens for entry {:d}: {!r}'.format(
                entry_id, entry_text
            )
        )
        # TODO: do not rely on id_to_token to make Token instances for you.
        entry_token = id_to_token(entry_id)
        assert stringify_token(entry_token) == entry_text
