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

import os
import token
import tokenize
from io import BytesIO
from keyword import iskeyword
from typing import AnyStr, IO, Iterable, Optional, Sequence, Union

from sensibility.pipeline import Pipeline, PipelineStage

from . import Language, SourceSummary
from ..lexical_analysis import Lexeme, Position, Token


class PythonPipeline(Pipeline):
    """
    Converts Python tokens to a format suitable for training and evaluating.
    """

    @property
    def stages(self) -> Sequence[PipelineStage]:
        return self.vocabularize, self.prune

    def prune(self, token: str) -> Optional[str]:
        EXTRANEOUS_TOKENS = {
            # Always occurs as the first token: internally indicates the file
            # ecoding, but is irrelelvant once the stream is already tokenized
            'ENCODING',

            # Always occurs as the last token.
            'ENDMARKER',

            # Insignificant newline; not to be confused with NEWLINE
            'NL',

            # Discard comments
            'COMMENT',

            # Represents a tokenization error. This should never appear for
            # syntatically correct files.
            'ERRORTOKEN',
        }
        if token in EXTRANEOUS_TOKENS:
            return None
        else:
            return token

    def vocabularize(self, token: Lexeme) -> Optional[str]:
        return open_closed_tokens(token)

    def tokenize(self, source: AnyStr) -> Sequence[Token]:
        return python.tokenize(source)


class Python(Language):
    """
    Defines the Python 3.6 language.
    """

    extensions = {'.py'}
    pipeline = PythonPipeline()

    def tokenize(self, source: Union[str, bytes, IO[bytes]]) -> Sequence[Token]:
        """
        Tokenizes Python sources.

        NOTE:
        This may include extra, unwanted tokens, including
        COMMENT, ENCODING, ENDMARKER, ERRORTOKEN

        NOTE:
        There are TWO newline tokens:
        NEWLINE and NL

        NEWLINES are actually used in the grammar;
        whereas NL are "filler" newlines, for formatting.

        BUGS: does not accept the same language as check_syntax()!

        To use CPython's internal tokenizer, one must create a brand new
        Python-accessible interface for it.
        """

        def open_as_file() -> IO[bytes]:
            if isinstance(source, str):
                # TODO: technically incorrect -- have to check coding line,
                # but I ain't doing that...
                return BytesIO(source.encode('UTF-8'))
            elif isinstance(source, bytes):
                return BytesIO(source)
            else:
                return source

        with open_as_file() as source_file:
            token_stream = tokenize.tokenize(source_file.readline)
            return [Token(name=token.tok_name[tok.type],
                          value=tok.string,
                          start=Position(line=tok.start[0], column=tok.start[1]),
                          end=Position(line=tok.end[0], column=tok.end[1]))
                    for tok in token_stream]

    def check_syntax(self, source: Union[str, bytes]) -> bool:
        r"""
        Given a source file, returns True if the file compiles.

        >>> python.check_syntax('print("Hello, World!")')
        True
        >>> python.check_syntax('import java.util.*;')
        False
        >>> python.check_syntax('\x89PNG\x0D\x0A\x1A\x0A\x00\x00\x00\x0D')
        False
        >>> python.check_syntax(r"AWESOME_CHAR_ESCAPE = '\x0G'")
        False
        """

        # I have the sneaking suspicion that compile() puts stuff in a cache that
        # is NOT garbage collected! Since I consider this a pretty serious memory
        # leak, I implemented this batshit crazy technique. Basically, let the
        # operating system be our garbage collector.

        pid = os.fork()
        if pid == 0:
            # Child process. Let it crash!!!
            try:
                compile(source, '<unknown>', 'exec')
            except:
                # Use _exit so it doesn't raise a SystemExit exception.
                os._exit(-1)
            else:
                os._exit(0)
        else:
            # Parent process.
            child_pid, status = os.waitpid(pid, 0)
            return status == 0

    def summarize_tokens(self, tokens: Sequence[Token]) -> SourceSummary:
        r"""
        Calculates the word count of a Python source.

        >>> python.summarize('import sys\n\nsys.stdout.write("hello")\n')
        SourceSummary(sloc=2, n_tokens=12)
        """
        if any(tok.name == 'ERRORTOKEN' for tok in tokens):
            raise SyntaxError('ERRORTOKEN')

        tokens = [token for token in tokens if is_physical_token(token)]

        INTANGIBLE_TOKENS = {'DEDENT', 'NEWLINE'}
        # Special case DEDENT and NEWLINE tokens:
        # They're do not count towards the line count (they are often on empty
        # lines).
        unique_lines = set(lineno for token in tokens
                           for lineno in token.lines
                           if token.name not in INTANGIBLE_TOKENS)

        return SourceSummary(sloc=len(unique_lines), n_tokens=len(tokens))

    def vocabularize_tokens(self, source: Iterable[Token]) -> Iterable[str]:
        return self.pipeline.execute(source)


python: Language = Python()


def is_physical_token(token: Lexeme) -> bool:
    """
    Return True when the token is something that should be counted towards
    sloc (source lines of code).
    """
    FAKE_TOKENS = {
        'ENDMARKER', 'ENCODING', 'COMMENT', 'NL', 'ERRORTOKEN'
    }
    return token.name not in FAKE_TOKENS




def open_closed_tokens(token: Lexeme) -> str:
    """
    'Flattens' Python into tokens based on whether the token is open or
    closed.
    """

    # List of token names that whose text should be used verbatim as the type.
    VERBATIM_CLASSES = {
        "AMPER", "AMPEREQUAL", "ASYNC", "AT", "ATEQUAL", "AWAIT", "CIRCUMFLEX",
        "CIRCUMFLEXEQUAL", "COLON", "COMMA", "DOT", "DOUBLESLASH",
        "DOUBLESLASHEQUAL", "DOUBLESTAR", "DOUBLESTAREQUAL", "ELLIPSIS",
        "EQEQUAL", "EQUAL", "GREATER", "GREATEREQUAL", "LBRACE", "LEFTSHIFT",
        "LEFTSHIFTEQUAL", "LESS", "LESSEQUAL", "LPAR", "LSQB", "MINEQUAL",
        "MINUS", "NOTEQUAL", "OP", "PERCENT", "PERCENTEQUAL", "PLUS", "PLUSEQUAL",
        "RARROW", "RBRACE", "RIGHTSHIFT", "RIGHTSHIFTEQUAL", "RPAR", "RSQB",
        "SEMI", "SLASH", "SLASHEQUAL", "STAR", "STAREQUAL", "TILDE", "VBAR",
        "VBAREQUAL"
    }

    if token.name == 'NAME':
        if iskeyword(token.value):
            return token.value
        else:
            return 'IDENTIFIER'
    elif token.name in VERBATIM_CLASSES:
        assert ' ' not in token.value
        return token.value
    else:
        # Note: includes NUMBER and STRING
        return token.name
