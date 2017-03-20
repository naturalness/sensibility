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

from typing import Sequence, Optional

from .corpus import Corpus
from .token_sequence import TokenSequence
from .token_utils import Token
from .vectors import Vectors
from .tokenize_js import tokenize


class SourceFile:
    """
    A source file from the corpus. Easy access to both the token sequence, and
    the source tokens themselves.
    """
    corpus: Optional[Corpus] = None
    vectors: Optional[Vectors] = None

    __slots__ = ('filehash', '_token_vector', '_source_tokens')

    def __init__(self, filehash: str) -> None:
        self.filehash = filehash
        # I used confusing names... TokenSequence != Sequence[Token]...
        self._token_vector: Optional[TokenSequence] = None
        self._source_tokens: Optional[Sequence[Token]] = None

    @property
    def vector(self) -> TokenSequence:
        if self._token_vector is not None:
            return self._token_vector
        if self.vectors is None:
            raise Exception('forgot to assign SourceFile.vectors')
        # Fetch the vector.
        _, tokens = self.vectors[self.filehash]
        self._token_vector = TokenSequence(tokens)
        return self._token_vector

    @property
    def source_tokens(self) -> Sequence[Token]:
        if self._source_tokens is not None:
            return self._source_tokens
        if self.corpus is None:
            raise Exception('forgot to assign SourceFile.corpus')
        # Fetch the source tokens.
        source = self.corpus.get_source(self.filehash)
        self._source_tokens = tokenize(source.decode('UTF-8'))
        return self._source_tokens

    def line_of_token(self, index: int) -> int: ...
