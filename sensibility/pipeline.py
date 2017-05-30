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
Tokenization Pipeline
"""

from abc import ABC, abstractmethod
from typing import Any, AnyStr, Callable, Optional, Sequence, Tuple, overload

from .language import Language
from .token_utils import Token, Lexeme, Location


PipelineStage = Callable[[Sequence[Any]], Sequence[Optional[Any]]]


class Pipeline(ABC):
    """
    A tokenization pipeline that converts tokens to a format appropriate for
    ingestion into a language model.

    TODO: make these composable, like cons-cells?
    """
    language: Language

    @property
    @abstractmethod
    def stages(self) -> Sequence[PipelineStage]: ...

    @overload
    def execute(self, tokens: AnyStr) -> Sequence[Any]: ...
    @overload
    def execute(self, tokens: Sequence[Token]) -> Sequence[Any]: ...

    def execute(self, source):
        """
        Executes all stages of the pipeline, returning a sequence of tokens,
        in a format specified by the pipeline.
        """
        intermediate: Sequence[Any]
        if isinstance(source, (bytes, str)):
            intermediate = self.tokenize(source)
        else:
            intermediate = source

        for stage in self.stages:
            intermediate = stage(intermediate)
        return intermediate

    @overload
    def execute_with_locations(self, tokens: AnyStr) -> Tuple[Location, Any]: ...
    @overload
    def execute_with_locations(self, tokens: Sequence[Token]) -> Tuple[Location, Any]: ...

    def execute_with_locations(self, source):
        """
        Same as #execute(), but returns pairs of (Location, token) pairs,
        where `token` is returned by the pipeline.
        """
        raise NotImplementedError

    def tokenize(self, source: AnyStr) -> Sequence[Token]:
        """
        Tokenizes the source code.
        """
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
        EXTRANEOUS_TOKENS = {
            'ENCODING', # Always occurs as the first token: internally
                        # indicates the file ecoding, but is irrelelvant once
                        # the stream is already tokenized
            'NL',       # Insignificant newline; not to be confused with
                        # NEWLINE
            'COMMENT',  # throw out comments
            'ENDMARKER',# Always occurs as the last token.
        }
        return [tok for tok in tokens if tok not in EXTRANEOUS_TOKENS]

    def vocabularize(self, tokens: Sequence[Any]) -> Sequence[Any]:
        return [open_closed_tokens(token) for token in tokens]


def open_closed_tokens(token: Lexeme) -> str:
    """
    'Flattens' Python into tokens based on whether the token is open or
    closed.
    """
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
