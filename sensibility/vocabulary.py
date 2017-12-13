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
from typing import Dict, Iterable, NewType, Optional, Sequence, Sized, cast

__all__ = 'Vocabulary', 'Entry', 'Vind'

# A vocabulary index that gets in your face.
Vind = NewType('Vind', int)
# A vocabulary entry
Entry = NewType('Entry', str)

UNK_TOKEN = '<UNK>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'


class VocabularyError(Exception):
    """
    A generic vocabulary error.
    """


class OutOfVocabularyError(VocabularyError):
    """
    Raised when a token does not exist in the vocabulary.
    """


class NoSourceRepresentationError(VocabularyError):
    """
    Raise when there is no way to convert the Vocabulary index into a
    token that can be inserted into the file.
    """


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
        Yields all source-representable entries of the vocabulary
        i.e., everything excluding the special entries <s>, </s>, and <unk>.
        """
        for ind in self.representable_indicies():
            yield self._index2text[ind]

    def representable_indicies(self) -> Iterable[Vind]:
        return (Vind(i) for i in range(self.minimum_representable_index(),
                                       self.maximum_representable_index() + 1))

    def to_text(self, index: Vind) -> str:
        return self._index2text[index]

    def to_index(self, text: str) -> Vind:
        """
        Returns the cooresponding a vocabulary ID for the given entry.

        Raises OutOfVocabularyError if it's not found.
        """
        try:
            return self._text2index[text]
        except KeyError as cause:
            raise OutOfVocabularyError(text) from cause

    def to_index_or_unk(self, text: str) -> Vind:
        try:
            return self.to_index(text)
        except OutOfVocabularyError:
            return self.unk_token_index

    def minimum_representable_index(self) -> Vind:
        """
        The smallest vocabulary index of a source-representable token.

        All the entries from here to self.maximum_representable_index() are
        also source-representable.
        """
        # The special entries are always the first few entries; thus, the
        # entry immidiately after is the first source-representable entry.
        return Vind(len(self.SPECIAL_ENTRIES))

    def maximum_representable_index(self) -> Vind:
        """
        The largest vocabulary index of a source-representable token.
        """
        # The very last entry of the vocabulary.
        return Vind(len(self) - 1)

    def __len__(self) -> int:
        return len(self._index2text)

    def __getitem__(self, idx: Vind) -> Entry:
        return self._index2text[idx]

    def to_source_text(self, idx: Vind) -> str:
        # TODO: return a lexeme
        raise NotImplementedError

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
