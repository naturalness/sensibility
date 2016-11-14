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


import logging

logger = logging.getLogger(__name__)

UNK_TOKEN = '/*<unknown>*/'
START_TOKEN = '/*<start>*/'
END_TOKEN = '/*<end>*/'


class Vocabulary:
    """
    >>> v = Vocabulary([START_TOKEN, 'var', '$identifier', ';', END_TOKEN])
    >>> v.to_text(2)
    '$identifier'
    >>> v.to_index('var')
    1
    >>> len(v)
    5
    """

    __slots__ = ('_text2index', '_index2text')

    def __init__(self, array):
        assert isinstance(array, list)
        assert array[0] == START_TOKEN
        assert array[-1] == END_TOKEN
        self._index2text = tuple(array)
        self._text2index = {text: index for index, text in enumerate(array)}
        assert self._text2index[START_TOKEN] == 0
        assert self._text2index[END_TOKEN] == len(array) - 1

    def to_text(self, index):
        return self._index2text[index]

    def to_index(self, text):
        return self._text2index[text]

    def __len__(self):
        return len(self._index2text)

try:
    from calculated_vocabulary import VOCAB
except ImportError:
    logger.exception("Could not load generated vocabulary.")
else:
    vocabulary = Vocabulary(VOCAB)
