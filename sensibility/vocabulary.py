#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2016 Eddie Antonio Santos <easantos@ualberta.ca>
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

import json
import warnings
from os import PathLike
from typing import NewType, Sized, List, cast, Tuple, Dict, Iterable


# A vocabulary index that gets in your face.
Vind = NewType('Vind', int)

UNK_TOKEN = '/*<UNK>*/'
START_TOKEN = '/*<START>*/'
END_TOKEN = '/*<END>*/'


class Vocabulary(Sized):
    """
    A vocabulary contains all the contents for a file.
    """
    def __init__(self, entries: Iterable[str]) -> None:
        raise NotImplementedError

    @classmethod
    def from_json_file(cls, filename: PathLike) -> 'Vocabulary':
        with open(filename) as json_file:
            return cls(json.load(json_file))

    def __len__(self) -> int:
        raise NotImplementedError


class LegacyVocabulary(Sized):
    """
    One-to-one mapping of vocabulary strings to vocabulary indices (Vinds).

    >>> v = LegacyVocabulary([START_TOKEN, 'var', '$identifier', ';', END_TOKEN])
    >>> v.to_text(2)
    '$identifier'
    >>> v.to_index('var')
    1
    >>> len(v)
    5
    """

    __slots__ = ('_text2index', '_index2text')

    def __init__(self, array: List[str]) -> None:
        warnings.warn('deprecated', DeprecationWarning)
        assert array[0] == START_TOKEN
        assert array[-1] == END_TOKEN
        self._index2text = tuple(array)
        self._text2index = {text: index for index, text in enumerate(array)}
        assert self._text2index[START_TOKEN] == 0
        assert self._text2index[END_TOKEN] == len(array) - 1

    def to_text(self, index: Vind) -> str:
        return self._index2text[index]

    def to_index(self, text: str) -> Vind:
        return cast(Vind, self._text2index[text])

    def __len__(self) -> int:
        return len(self._index2text)

    start_token_index = Vind(0)

    @property
    def end_token_index(self) -> Vind:
        return Vind(len(self) - 1)

    @property
    def start_token(self) -> str:
        """
        Text of the start token.
        """
        return self.to_text(self.start_token_index)

    @property
    def end_token(self) -> str:
        """
        Text of the end token.
        """
        return self.to_text(self.end_token_index)


try:
    from .js_vocabulary import VOCAB
except ImportError:
    warnings.warn("Could not load generated vocabulary.")
else:
    vocabulary = LegacyVocabulary(VOCAB)
