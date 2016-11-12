#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
>>> from corpus import Token
>>> stringify_token(Token(value='**=', type='Punctuator', loc=None))
'**='
>>> stringify_token(Token(value='var', type='Keyword', loc=None))
'var'
>>> stringify_token(Token(value='false', type='Boolean', loc=None))
'false'
>>> stringify_token(Token(value='NULL', type='Null', loc=None))
'null'
>>> stringify_token(Token(value='``', type='Template', loc=None))
'`standalone-template`'
>>> stringify_token(Token(value='``', type='Template', loc=None))
'`standalone-template`'
>>> stringify_token(Token(value='`${', type='Template', loc=None))
'`template-head${'
>>> stringify_token(Token(value='}`', type='Template', loc=None))
'}template-tail`'
>>> stringify_token(Token(value='}  ${', type='Template', loc=None))
'}template-middle${'
"""


def singleton(cls):
    return cls()


@singleton
class stringify_token:
    def __call__(self, token):
        try:
            fn = getattr(self, token.type)
        except AttributeError:
            raise TypeError('Unhandled type: %s' %(token.type,))
        return fn(token.value)

    def Boolean(self, text):
        return text

    def Identifier(self, text):
        return '$anyIdentifier'

    def Keyword(self, text):
        return text

    def Null(self, text):
        return 'null'

    def Numeric(self, text):
        return '/*any-number*/0'

    def Punctuator(self, text):
        return text

    def String(self, text):
        return '"any-string"'

    def RegularExpression(self, text):
        return '/any-regexp/'

    def Template(self, text):
        assert len(text) >= 2
        if text.startswith('`'):
            if text.endswith('`'):
                return '`standalone-template`'
            elif text.endswith('${'):
                return '`template-head${'
        elif text.startswith('}'):
            if text.endswith('`'):
                return '}template-tail`'
            elif text.endswith('${'):
                return '}template-middle${'
        raise TypeError('Unhandled template literal: ' + text)
