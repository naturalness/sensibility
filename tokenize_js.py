#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Tokenizes JavaScript.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from token_utils import Token


THIS_DIRECTORY = Path(__file__).parent
TOKENIZE_JS_BIN = (str(THIS_DIRECTORY / 'tokenize-js' / 'wrapper.sh'),)
CHECK_SYNTAX_BIN = (*TOKENIZE_JS_BIN, '--check-syntax')


def synthetic_file(text):
    """
    Creates an unnamed temporary file with the given text content.
    The returned file object always has a fileno.
    """
    file_obj = tempfile.TemporaryFile('w+t', encoding='utf-8')
    file_obj.write(text)
    file_obj.seek(0)
    return file_obj


def tokenize(text):
    """
    Tokenizes the given string.

    >>> tokens = tokenize('$("hello");')
    >>> len(tokens)
    5
    >>> isinstance(tokens[0], Token)
    True
    """
    with synthetic_file(text) as f:
        return tokenize_file(f)


def check_syntax(source):
    """
    Checks the syntax of the given JavaScript string.

    >>> check_syntax('function name() {}')
    True
    >>> check_syntax('function name() }')
    False
    """
    with synthetic_file(source) as source_file:
        return check_syntax_file(source_file)


def tokenize_file(file_obj):
    """
    Tokenizes the given JavaScript file.

    >>> with synthetic_file('$("hello");') as f:
    ...     tokens = tokenize_file(f)
    >>> len(tokens)
    5
    >>> isinstance(tokens[0], Token)
    True
    """
    status = subprocess.run(TOKENIZE_JS_BIN,
                            check=True,
                            stdin=file_obj,
                            stdout=subprocess.PIPE)
    return [
        Token.from_json(raw_token)
        for raw_token in json.loads(status.stdout.decode('UTF-8'))
    ]


def check_syntax_file(source_file):
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
    status = subprocess.run(CHECK_SYNTAX_BIN, stdin=source_file)
    return status.returncode == 0
