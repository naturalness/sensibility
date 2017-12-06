#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2017 Eddie Antonio Santos <easantos@ualberta.ca>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import os
import sys
import token
from io import BytesIO
from keyword import iskeyword
from pathlib import Path
from typing import (
    Any, AnyStr, Callable, IO, Iterable, Optional, Tuple, Union,
    overload, cast
)

import javac_parser

from .. import Language, SourceSummary
from ...lexical_analysis import Lexeme, Location, Position, Token
from ...vocabulary import NoSourceRepresentationError, Vocabulary, Vind


here = Path(__file__).parent


class JavaVocabulary(Vocabulary):
    """
    The vocabulary, except it returns from
    """
    first_entry_num = len(Vocabulary.SPECIAL_ENTRIES)

    def __init__(self, entries: Iterable[str], reprs: Iterable[str]) -> None:
        super().__init__(entries)
        # Create a look-up table for source representations.
        # The special tokens <unk>, <s>, </s> have NO reprs, thus are not
        # stored.
        self._index2repr = tuple(reprs)
        assert len(self._index2text) == self.first_entry_num + len(self._index2repr)

    def to_source_text(self, idx: Vind) -> str:
        if idx < self.first_entry_num:
            raise NoSourceRepresentationError(idx)
        return self._index2repr[idx - self.first_entry_num]

    @staticmethod
    def load() -> 'JavaVocabulary':
        entries = []
        reprs = []

        # Load from a tab-separated-values file
        with open(here / 'vocabulary.tsv') as vocab_file:
            first_entry = JavaVocabulary.first_entry_num
            for expected_num, line in enumerate(vocab_file, start=first_entry):
                # src_repr -- source representation
                num, entry, src_repr = line.rstrip().split()
                assert expected_num == int(num)
                entries.append(entry)
                reprs.append(src_repr)

        return JavaVocabulary(entries, reprs)


def to_str(source: Union[str, bytes, IO[bytes]]) -> str:
    """
    Coerce an input format to a Unicode string.
    """
    if isinstance(source, str):
        return source
    elif isinstance(source, bytes):
        # XXX: Assume it's UTF-8 encoded!
        return source.decode('UTF-8')
    else:
        raise NotImplementedError


class LazyVocabulary:
    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, value):
        if not hasattr(self, 'value'):
            self.value = self.fn()
        return self.value


class JavaToken(Token):
    """
    HACK: javac_parser has some... interesting ideas about normalization.
    so add a `_raw` field to the token.
    """
    # TODO: fix with upstream (javac_parser) to return a sensible value for the normalized value
    __slots__ = ('_raw',)

    def __init__(self, *, _raw: str, name: str, value: str, start: Position, end: Position) -> None:
        super().__init__(name=name, value=value, start=start, end=end)
        self._raw = _raw

    def __repr__(self) -> str:
        cls = type(self).__name__
        return (f"{cls}(_raw={self._raw!r}"
                f"name={self.name!r}, value={self.value!r}, "
                f"start={self.start!r}, end={self.end!r})")


class Java(Language):
    """
    Defines the Java 8 programming language.
    """

    extensions = {'.java'}
    vocabulary = cast(Vocabulary, LazyVocabulary(JavaVocabulary.load))

    @property
    def java(self):
        """
        Lazily start up the Java server. This decreases the chances of things
        going horribly wrong when two seperate process initialize
        the Java language instance around the same time.
        """
        if not hasattr(self, '_java_server'):
            self._java_server = javac_parser.Java()

            # Py4j usually crashes as Python is cleaning up after exit() so
            # decrement the servers' reference count to lessen the chance of
            # that happening.
            @atexit.register
            def remove_reference():
                del self._java_server

        return self._java_server

    def tokenize(self, source: Union[str, bytes, IO[bytes]]) -> Iterable[Token]:
        tokens = self.java.lex(to_str(source))
        # Each token is a tuple with the following structure
        # (reproduced from javac_parser.py):
        #   1. Lexeme type
        #   2. Value (as it appears in the source file)
        #   3. A 2-tuple of start line, start column
        #   4. A 2-tuple of end line, end column
        #   5. A whitespace-free representation of the value
        for name, raw_value, start, end, normalized in tokens:
            # Omit the EOF token, as it's only useful for the parser.
            if name == 'EOF':
                continue
            # Take the NORMALIZED value, as Java allows unicode escapes in
            # ARBITRARY tokens and then things get hairy here.
            yield JavaToken(_raw=raw_value,
                            name=name, value=normalized,
                            start=Position(line=start[0], column=start[1]),
                            end=Position(line=end[0], column=end[1]))

    def check_syntax(self, source: Union[str, bytes]) -> bool:
        return self.java.get_num_parse_errors(to_str(source)) == 0

    def summarize_tokens(self, source: Iterable[Token]) -> SourceSummary:
        toks = [tok for tok in source if tok.name != 'EOF']
        slines = set(line for tok in toks for line in tok.lines)
        return SourceSummary(n_tokens=len(toks), sloc=len(slines))

    def vocabularize_tokens(self, source: Iterable[Token]) -> Iterable[Tuple[Location, str]]:
        for token in source:
            yield token.location, token.name


java: Language = Java()
