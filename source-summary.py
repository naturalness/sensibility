#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Returns a summary of the number of source lines of code in the program.
Like a semantic wc.
"""

import sys

import sensibility.language.python
from sensibility.token_utils import Lexeme

def is_physical_token(token: Lexeme) -> bool:
    FAKE_TOKENS = {
        'ENDMARKER', 'ENCODING', 'COMMENT', 'NL'
    }
    return token.name not in FAKE_TOKENS

if __name__ == '__main__':
    language = sensibility.language.python.Python()
    tokens = [token for token in language.tokenize(sys.stdin.read())
              if is_physical_token(token)]
    n_tokens = len(tokens)

    INTANGIBLE_TOKENS = {'DEDENT', 'NEWLINE'}
    # Special case DEDENT and NEWLINE tokens:
    # They're not really "there" but should be counted as a token.
    sloc = len(set(lineno for token in tokens
                   for lineno in token.lines
                   if token.name not in INTANGIBLE_TOKENS))
    print(f"{sloc:8d} {n_tokens:8d}")
