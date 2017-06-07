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
Tokenizes JavaScript.

Requires Node.JS >= 4.0. The first invocation of any of the external commands
will automatically install all required Node.JS dependencies through NPM.
"""

import json
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Sequence, TextIO, cast

from .language.javascript import javascript
from .lexical_analysis import Token, Lexeme
from .vocabulary import vocabulary, Vind

warnings.warn("deprecated", DeprecationWarning)

THIS_DIRECTORY = Path(__file__).parent
TOKENIZE_JS_BIN = (str(THIS_DIRECTORY / 'tokenize-js' / 'wrapper.sh'),)
CHECK_SYNTAX_BIN = (*TOKENIZE_JS_BIN, '--check-syntax')


def synthetic_file(text: str) -> TextIO:
    """
    Creates an unnamed temporary file with the given text content.
    The returned file object always has a fileno.
    """
    file_obj = cast(TextIO, tempfile.TemporaryFile('w+t', encoding='utf-8'))
    file_obj.write(text)
    file_obj.seek(0)
    return file_obj


def tokenize(text: str) -> Sequence[Token]:
    """
    Tokenizes the given string.

    >>> tokens = tokenize('$("hello");')
    >>> len(tokens)
    5
    >>> isinstance(tokens[0], Token)
    True
    """
    warnings.warn("deprecated", DeprecationWarning)
    with synthetic_file(text) as f:
        return tokenize_file(f)


def check_syntax(source: str) -> bool:
    """
    Checks the syntax of the given JavaScript string.

    >>> check_syntax('function name() {}')
    True
    >>> check_syntax('function name() }')
    False
    """
    warnings.warn("deprecated", DeprecationWarning)
    with synthetic_file(source) as source_file:
        return check_syntax_file(source_file)


def tokenize_file(file_obj: TextIO) -> Sequence[Token]:
    """
    Tokenizes the given JavaScript file.

    """
    warnings.warn("deprecated", DeprecationWarning)
    return javascript.tokenize(file_obj)  # type: ignore


def check_syntax_file(source_file: TextIO) -> bool:
    """
    Check the syntax of the give JavaScript file.

    >>> with synthetic_file('$("hello");') as f:
    ...     assert check_syntax_file(f)
    >>> with synthetic_file('$("hello" + );') as f:
    ...     assert check_syntax_file(f)
    Traceback (most recent call last):
        ...
    AssertionError
    """
    warnings.warn("deprecated", DeprecationWarning)
    status = subprocess.run(CHECK_SYNTAX_BIN, stdin=source_file)
    return status.returncode == 0


def id_to_token(token_id: Vind) -> Optional[Lexeme]:
    """
    Return a synthetic token for the given token ID.

    Returns None if the token is not representable in code.

    >>> token = id_to_token(70)
    >>> token.name
    'Keyword'
    >>> token.value
    'function'
    >>> id_to_token(0) is None
    True
    """
    warnings.warn("deprecated", DeprecationWarning)
    if token_id not in range(1, 101):
        return None
    with synthetic_file(vocabulary.to_text(token_id)) as file_obj:
        return tokenize_file(file_obj)[0]
