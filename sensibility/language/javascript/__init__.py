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

"""
Language definition for JavaScript.
"""

import re
import tempfile
from io import IOBase
from pathlib import Path
from typing import Any, Callable, IO, Iterable, Optional, Sequence, Tuple, Union
from typing import cast

from .. import Language, SourceSummary
from ...lexical_analysis import Token, Lexeme, Location, Position
from ...vocabulary import Vocabulary
from .esprima_interface import Server, get_server, tokenize, check_syntax


here = Path(__file__).parent.absolute()


class JavaScript(Language):
    """
    Defines the JavaScript language.
    """

    extensions = {'.js'}
    vocabulary = Vocabulary.from_json_file(here / 'vocabulary.json')

    def tokenize(self, source: Union[str, bytes, IO[bytes]]) -> Sequence[Token]:
        """
        Tokenizes the given JavaScript file.
        """

        with SafeSourceFile(source) as source_file:
            return esprima_to_tokens(tokenize(source_file))

    def check_syntax(self, source: Union[str, bytes]) -> bool:
        with SafeSourceFile(source) as source_file:
            return check_syntax(source_file)

    def summarize_tokens(self, source: Iterable[Token]) -> SourceSummary:
        tokens = list(source)
        unique_lines = set(lineno for token in tokens
                           for lineno in token.lines)

        return SourceSummary(sloc=len(unique_lines), n_tokens=len(tokens))

    def vocabularize_tokens(self, tokens: Iterable[Token]) -> Iterable[Tuple[Location, str]]:
        for token in tokens:
            yield token.location, stringify_lexeme(token)


class JavaScriptWithServer(JavaScript):
    fullname = 'JavaScript'

    def tokenize(self, source: Union[str, bytes, IO[bytes]]) -> Sequence[Token]:
        tokens = get_server().tokenize(ensure_bytes(source))
        return esprima_to_tokens(tokens)

    def check_syntax(self, source: Union[str, bytes]) -> bool:
        return get_server().check_syntax(ensure_bytes(source))


def ensure_bytes(source: Union[str, bytes, IO[bytes]]) -> bytes:
    if isinstance(source, str):
        return source.encode('UTF-8')
    elif isinstance(source, bytes):
        return source
    else:
        # Hack: we don't trust the IO[bytes] signature, so ensure bytes once
        # more, just in case.
        return ensure_bytes(source.read())


def esprima_to_tokens(raw_tokens: Iterable[Any]) -> Sequence[Token]:
    return [from_esprima_format(tok) for tok in raw_tokens]


class SafeSourceFile:
    """
    Context manager that always yields a IO[bytes] object, and safely closes
    it if it was created here.
    """

    def __init__(self, source: Union[str, bytes, IO[bytes]]) -> None:
        self.source = source
        self._owned: Optional[IO[bytes]] = None
        self._foreign: Optional[IO[bytes]] = None

    def __enter__(self) -> IO[bytes]:
        if isinstance(self.source, (str, bytes)):
            self._owned = synthetic_file(self.source)
            return self._owned
        elif isinstance(self.source, IOBase):
            self._foreign = self.source
            return self.source
        else:
            raise ValueError(self.source)

    def __exit__(self, *exc_info: Any) -> None:
        if self._owned is not None:
            self._owned.close()
        else:
            assert self._foreign is not None


def synthetic_file(source: Union[str, bytes]) -> IO[bytes]:
    """
    Creates a true file, readable-file with the given contents.
    """
    file_obj = tempfile.TemporaryFile(mode='w+b')
    if isinstance(source, str):
        file_obj.write(source.encode('UTF-8'))
    else:
        file_obj.write(source)
    file_obj.flush()
    file_obj.seek(0)
    return file_obj


def from_esprima_format(token) -> Token:
    """
    Parses the Esprima's token format
    """
    loc = token['loc']
    return Token(name=token['type'],
                 value=token['value'],
                 start=Position(line=loc['start']['line'],
                                column=loc['start']['column']),
                 end=Position(line=loc['end']['line'],
                              column=loc['end']['column']))


class StringifyLexeme:
    r"""
    Converts a Lexeme to its vocabularized form.

    >>> stringify_lexeme(Lexeme(value='**=', name='Punctuator'))
    '**='
    >>> stringify_lexeme(Lexeme(value='\\u002e', name='Punctuator'))
    '.'
    >>> stringify_lexeme(Lexeme(value='var', name='Keyword'))
    'var'
    >>> stringify_lexeme(Lexeme(value='\\u0069f', name='Keyword'))
    'if'
    >>> stringify_lexeme(Lexeme(value='yie\\u{006C}d', name='Keyword'))
    'yield'
    >>> stringify_lexeme(Lexeme(value='false', name='Boolean'))
    'false'
    >>> stringify_lexeme(Lexeme(value='\\u0066alse', name='Boolean'))
    'false'
    >>> stringify_lexeme(Lexeme(value='null', name='Null'))
    'null'
    >>> stringify_lexeme(Lexeme(value='``', name='Template'))
    '<STANDALONE-TEMPLATE>'
    >>> stringify_lexeme(Lexeme(value='`${', name='Template'))
    '<TEMPLATE-HEAD>'
    >>> stringify_lexeme(Lexeme(value='}`', name='Template'))
    '<TEMPLATE-TAIL>'
    >>> stringify_lexeme(Lexeme(value='}  ${', name='Template'))
    '<TEMPLATE-MIDDLE>'
    >>> stringify_lexeme(Lexeme(value='"hello world"', name='String'))
    '<STRING>'
    >>> stringify_lexeme(Lexeme(value='💩', name='Identifier'))
    '<IDENTIFIER>'
    >>> stringify_lexeme(Lexeme(value='/g/i', name='RegularExpression'))
    '<REGEXP>'
    >>> stringify_lexeme(Lexeme(value='3.14', name='Numeric'))
    '<NUMBER>'
    """

    def __call__(self, token) -> str:
        # This is essentially my attempt to hack-in pattern matching in
        # Python. There's a fixed number of Token#name that we match on, and
        # decide what string to output.
        try:
            fn = getattr(self, token.name)
        except AttributeError:
            raise TypeError(f'Unhandled type: {token.name}')
        return fn(token.value)

    def Boolean(self, text):
        return unescape_unicode(text)

    def Identifier(self, text):
        return '<IDENTIFIER>'

    def Keyword(self, text):
        # Note: this also handles keywords in identifier position.
        # e.g.,
        #   blah.\u0069f = 1;
        #
        # See: https://git.io/vQCh6
        return unescape_unicode(text)

    def Null(self, text):
        return 'null'

    def Numeric(self, text):
        return '<NUMBER>'

    def Punctuator(self, text):
        return unescape_unicode(text)

    def String(self, text):
        return '<STRING>'

    def RegularExpression(self, text):
        return '<REGEXP>'

    def Template(self, text):
        assert len(text) >= 2
        if text.startswith('`'):
            if text.endswith('`'):
                return '<STANDALONE-TEMPLATE>'
            elif text.endswith('${'):
                return '<TEMPLATE-HEAD>'
        elif text.startswith('}'):
            if text.endswith('`'):
                return '<TEMPLATE-TAIL>'
            elif text.endswith('${'):
                return '<TEMPLATE-MIDDLE>'
        raise TypeError('Unhandled template literal: ' + text)


def unescape_unicode(text: str) -> str:
    r"""
    Unescapes \uhhhh sequences in the string.

    Needed because according to the ECMAScript standard:

        In string literals, regular expression literals, template literals and
        identifiers, any Unicode code point may also be expressed using
        Unicode escape sequences that explicitly express a code point's
        numeric value

    https://www.ecma-international.org/ecma-262/#sec-source-text

    >>> unescape_unicode(r'\u0069\u0066')
    'if'
    >>> unescape_unicode(r'\u{0066}or')
    'for'
    >>> unescape_unicode(r'\u{110000}')
    Traceback (most recent call last):
        ...
    ValueError: chr() arg not in range(0x110000)
    >>> unescape_unicode(r'\u{01f4A9}')
    '💩'
    """
    # Match:
    # https://www.ecma-international.org/ecma-262/#prod-UnicodeEscapeSequence
    return re.sub(r'\\u([0-9a-fA-F]{4}|[{][0-9a-fA-F]+[}])',
                  lambda m: chr(int(m.group(1).strip('{}'), 16)),
                  text)


# The main exports.
javascript = JavaScriptWithServer()
stringify_lexeme = cast(Callable[[Lexeme], str], StringifyLexeme())
