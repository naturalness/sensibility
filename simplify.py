#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


"""
Turns a stream of tokens, into a stream of
"""

import sys
import pickle
from keyword import iskeyword

import sensibility.language.python
from sensibility.token_utils import Lexeme


def open_closed_tokens(token: Lexeme) -> str:
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
            return 'identifier'
    elif token.name == 'NUMBER':
        return '0'
    elif token.name == 'STRING':
        return '"string"'
    elif token.name in VERBATIM_CLASSES:
        assert ' ' not in token.value
        return token.value
    else:
        return token.name

if __name__ == '__main__':
    language = sensibility.language.python.Python()
    with open(sys.stdin.fileno(), 'rb') as input_file:
        tokens = pickle.load(input_file)

    print(' '.join(open_closed_tokens(t) for t in tokens))
