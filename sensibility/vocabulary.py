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
from typing import NewType, Sized, List, cast, Tuple, Dict, Iterable, Sequence


# A vocabulary index that gets in your face.
Vind = NewType('Vind', int)
# A vocabulary entry
Entry = NewType('Entry', str)

UNK_TOKEN = '<UNK>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'


class Vocabulary(Sized):
    """
    One-to-one mapping of vocabulary strings to vocabulary indices (Vinds).

    >>> v = Vocabulary(['var', '<IDENTIFIER>', ';'])
    >>> v[4]
    '<IDENTIFIER>'
    >>> v.to_index(';')
    5
    >>> len(v)
    6
    """
    SPECIAL_ENTRIES = (UNK_TOKEN, START_TOKEN, END_TOKEN)

    def __init__(self, entries: Iterable[str]) -> None:
        self._index2text = cast(Sequence[Entry],
                                self.SPECIAL_ENTRIES + tuple(entries))
        self._text2index: Dict[str, Vind] = {
            text: Vind(index) for index, text in enumerate(self._index2text)
        }

        assert len(self._index2text) == len(set(self._index2text)), (
            'Duplicate entries in vocabulary'
        )

    def entries(self) -> Iterable[Entry]:
        """
        Yields all "true" entries of the vocabulary (all minus the special
        entries).
        """
        for ind in range(len(self.SPECIAL_ENTRIES), len(self)):
            yield self._index2text[cast(Vind, ind)]

    def to_text(self, index: Vind) -> str:
        return self._index2text[index]

    def to_index(self, text: str) -> Vind:
        return cast(Vind, self._text2index[text])

    def __len__(self) -> int:
        return len(self._index2text)

    def __getitem__(self, idx: Vind) -> Entry:
        return self._index2text[idx]

    # TODO: ???
    #def to_lexeme(self, idx: Vind) -> Lexeme: ...

    @classmethod
    def from_json_file(cls, filename: PathLike) -> 'Vocabulary':
        with open(filename) as json_file:
            return cls(json.load(json_file))

    unk_token_index = Vind(0)
    start_token_index = Vind(1)
    end_token_index = Vind(2)
    unk_token = UNK_TOKEN
    start_token = START_TOKEN
    end_token = END_TOKEN


class LegacyVocabulary(Sized):
    """
    One-to-one mapping of vocabulary strings to vocabulary indices (Vinds).

    >>> v = LegacyVocabulary([START_TOKEN, 'var', '$identifier', ';', END_TOKEN])
    >>> len(v)
    100
    """

    def __init__(self, array: List[str]) -> None:
        warnings.warn('deprecated', DeprecationWarning)
        from sensibility.language.javascript import javascript
        self._vocab = javascript.vocabulary
        assert len(self._vocab) == 101

    def to_text(self, index: Vind) -> str:
        if index == self.start_token_index:
            return self.start_token
        elif index == self.end_token_index:
            return self.end_token
        return self._vocab.to_text(index)

    def to_index(self, text: str) -> Vind:
        if text == self.start_token:
            return self.start_token_index
        elif text == self.end_token:
            return self.end_token_index
        return self._vocab.to_index(text)

    def __len__(self) -> int:
        return len(self._vocab) - 1

    start_token_index = Vind(0)
    end_token_index = Vind(99)
    start_token = '/*<START>*/'
    end_token = '/*<END>*/'


try:
    from .js_vocabulary import VOCAB
except ImportError:
    warnings.warn("Could not load generated vocabulary.")
else:
    vocabulary = LegacyVocabulary(VOCAB)
