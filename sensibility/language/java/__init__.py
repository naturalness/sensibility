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
from pathlib import Path
from typing import (
    Any, AnyStr, Callable, IO, Iterable, Optional, Sequence, Tuple, Union,
    overload,
)

import javalang  # type: ignore

from .. import Language, SourceSummary
from ...lexical_analysis import Lexeme, Location, Position, Token
from ...vocabulary import Vocabulary


here = Path(__file__).parent


class Java(Language):
    """
    Defines the Java 8 programming language.
    """

    extensions = {'.java'}
    vocabulary = Vocabulary.from_json_file(Path(__file__).parent /
                                           'vocabulary.json')

    def tokenize(self, source: Union[str, bytes, IO[bytes]]) -> Sequence[Token]:
        tokens = javalang.tokenizer.tokenize('System.out.println("Hello " + "world");')
        raise NotImplementedError

    def check_syntax(self, source: Union[str, bytes]) -> bool:
        try:
            tree = javalang.parse.parse(source)
            return True
        except javalang.parser.JavaSyntaxError:
            return False

    def summarize_tokens(self, source: Iterable[Token]) -> SourceSummary:
        raise NotImplementedError

    def vocabularize_tokens(self, source: Iterable[Token]) -> Iterable[Tuple[Location, str]]:
        raise NotImplementedError


java: Language = Java()
