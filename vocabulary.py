#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


UNK_TOKEN   = '/*<unknown>*/'
START_TOKEN = '/*<start>*/'
END_TOKEN   = '/*<end>*/'


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
        assert array[0] is START_TOKEN
        assert array[-1] is END_TOKEN
        self._index2text = array
        self._text2index = {text: index for index, text in enumerate(array)}
        assert self._text2index[START_TOKEN] == 0
        assert self._text2index[END_TOKEN] == len(array) - 1

    def to_text(self, index):
        return self._index2text[index]

    def to_index(self, text):
        return self._text2index[text]

    def __len__(self):
        return len(self._index2text)
