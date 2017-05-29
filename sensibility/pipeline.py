#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Tokenization Pipeline
"""

from abc import ABC, abstractmethod
from typing import Any, AnyStr, Callable, Sequence, overload

from .language import Language
from .token_utils import Token, Lexeme


PipelineStage = Callable[[Sequence[Any]], Sequence[Any]]


class Pipeline(ABC):
    """
    A tokenization pipeline that converts tokens to a format appropriate for
    ingestion into a language model.
    """
    language: Language

    @property
    @abstractmethod
    def stages(self) -> Sequence[PipelineStage]: ...

    @overload
    def execute(self, tokens: Sequence[Token]) -> Sequence[Any]: ...
    @overload
    def execute(self, tokens: AnyStr) -> Sequence[Any]: ...

    def execute(self, source):
        intermediate: Sequence[Any]
        if isinstance(source, bytes, str):
            intermediate = self.tokenize(source)
        else:
            intermediate = source

        for stage in self.stages:
            intermediate = stage(intermediate)
        return intermediate

    def tokenize(self, source: AnyStr) -> Sequence[Token]:
        return self.language.tokenize(source)


# TODO: Move from here down somewhere else, probably
from .language.python import Python
from keyword import iskeyword

class PythonPipeline(Pipeline):
    """
    Converts Python tokens to a format suitable for training and evaluating.
    """

    language = Python()

    @property
    def stages(self) -> Sequence[PipelineStage]:
        return self.vocabularize, self.prune

    def prune(self, tokens: Sequence[Any]) -> Sequence[Any]:
        ...

    def vocabularize(self, tokens: Sequence[Any]) -> Sequence[Any]:
        ...


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
