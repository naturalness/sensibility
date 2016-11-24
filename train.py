#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Trains an LSTM using Keras.
"""

import numpy as np

from vocabulary import vocabulary

# Based on White et al. 2015
DEFAULT_SIZE = 20
SIGMOID_ACTIVATIONS = 300

class Sentences:
    """
    Generates samples from the given vector.

    >>> from corpus import Token
    >>> from vectorize_tokens import vectorize_tokens
    >>> t = Token(value='var', type='Keyword', loc=None)
    >>> v = vectorize_tokens([t])
    >>> sentences = Sentences(v, size=20)
    >>> len(sentences)
    0
    >>> list(iter(sentences))
    []

    >>> v = vectorize_tokens([t] * 19)
    >>> sentences = Sentences(v, size=20)
    >>> len(sentences)
    2
    >>> x, y = next(iter(sentences))
    """

    def __init__(self, vector, *, size=DEFAULT_SIZE):
        # TODO: Step?
        self.vector = vector
        self.size= size

    def __iter__(self):
        n_sentences = len(self)

        if n_sentences <= 0:
            raise StopIteration

        sentence_len = self.size
        token_vector = self.vector
        vocab_size = len(vocabulary)

        # Create empty one-hot vectors
        x = np.zeros((n_sentences, sentence_len, vocab_size), dtype=np.bool)
        y = np.zeros((n_sentences, vocab_size), dtype=np.bool)

        for sentence_id in range(n_sentences):
            for i, token_id in enumerate(range(sentence_id, sentence_id +
                                               sentence_len)):
                x[sentence_id, i, token_id] = 1
            y[sentence_id, token_id] = 1

        yield x, y

    def __len__(self):
        """
        Returns how many sentences this will produce. Can be zero!
        """
        sentences_possible = 1 + len(self.vector) - self.size
        return at_least(0, sentences_possible)


def at_least(value, *args):
    """
    Returns at least the given number.

    For Dr. Hindle's sanity.
    """
    return max(value, *args)
