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

"""
TODO: rename this file!
"""

from itertools import islice

import numpy as np
from more_itertools import chunked
from path import Path

from vocabulary import vocabulary
from condensed_corpus import CondensedCorpus


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
    1
    >>> x, y = next(iter(sentences))
    >>> list(x) == [0] + [86] * 19
    True
    >>> y
    99
    >>> sentences = Sentences(v, size=20, backwards=True)
    >>> x, y = next(iter(sentences))
    >>> list(x) == [86] * 19 + [99]
    True
    >>> y
    0
    """

    def __init__(self, vector, *, size=None, backwards=False):
        # TODO: Step?
        self.vector = vector
        self.size= size
        self.backwards = backwards
        if not isinstance(size, int):
            raise ValueError("Size must be an int.")

    def __iter__(self):
        n_sentences = len(self)

        if n_sentences <= 0:
            raise StopIteration


        sentence_len = self.size
        token_vector = self.vector

        if not self.backwards:
            # Forwards
            def make_sample(start, end):
                sentence = token_vector[start:end]
                return sentence, token_vector[end]
        else:
            # Backwards
            def make_sample(start, end):
                sentence = token_vector[start + 1:end + 1]
                return sentence, token_vector[start]

        # Fill in the vectors.
        for sentence_id in range(n_sentences):
            start = sentence_id
            end = sentence_id + sentence_len
            assert end < len(token_vector), "not: %d < %d" %(end,
                                                        len(token_vector))
            yield make_sample(start, end)

    def __len__(self):
        """
        Returns how many sentences this will produce. Can be zero!
        """
        sentences_possible = len(self.vector) - self.size
        return at_least(0, sentences_possible)


def one_hot_batch(batch, *, batch_size=None, sentence_length=None,
                  vocab_size=len(vocabulary), np=np):
    """
    Creates a one hot vector of the batch.
    >>> x, y = one_hot_batch([(np.array([36]), 48)], batch_size=1024, sentence_length=20)
    >>> x.shape
    (1, 20, 100)
    >>> x[0, 0, 36]
    1
    >>> y.shape
    (1, 100)
    >>> y[0, 48]
    1
    """
    # Create empty one-hot vectors
    x = np.zeros((batch_size, sentence_length, vocab_size), dtype=np.bool)
    y = np.zeros((batch_size, vocab_size), dtype=np.bool)

    # Fill in the vectors.
    for sentence_id, (sentence, last_token_id) in enumerate(batch):
        # Fill in the one-hot matrix for X
        for pos, token_id in enumerate(sentence):
            x[sentence_id, pos, token_id] = True

        # Add the last token for the one-hot vector Y.
        y[sentence_id, last_token_id] = True

    samples_produced = sentence_id + 1

    if samples_produced < batch_size:
        #print("warning: less samples than batch size:", samples_produced)
        x = np.resize(x, ((samples_produced, sentence_length, vocab_size)))
        y = np.resize(y, ((samples_produced, vocab_size)))

    return x, y


class LoopBatchesEndlessly:
    def __init__(self, corpus_filename, folds,
                 batch_size=None,
                 sentence_length=None,
                 backwards=False):
        assert Path(corpus_filename).exists()
        assert isinstance(batch_size, int)
        assert isinstance(sentence_length, int)
        self.filename = corpus_filename
        self.folds = folds
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.backwards = backwards
        self.samples_per_epoch = count_samples_slow(corpus_filename, folds,
                                                    sentence_length)

    def __iter__(self):
        batch_size = self.batch_size
        for batch in self._yield_batches_endlessly():
            yield one_hot_batch(batch, batch_size=batch_size,
                                sentence_length=self.sentence_length)

    def _yield_sentences_from_corpus(self):
        """
        Yields all sentences from the corpus exactly once.
        """
        sentence_length = self.sentence_length
        corpus = CondensedCorpus.connect_to(self.filename)
        for fold in self.folds:
            for file_hash in corpus.hashes_in_fold(fold):
                _, tokens = corpus[file_hash]
                yield from Sentences(tokens,
                                     size=sentence_length,
                                     backwards=self.backwards)
        corpus.disconnect()

        batch_size = self.batch_size

    def _yield_batches_endlessly(self):
        """
        Yields batches of samples, in vectorized format, but NOT one-hot
        encoded.
        """
        batch_size = self.batch_size
        while True:
            yield from chunked(self._yield_sentences_from_corpus(),
                               batch_size)

    @classmethod
    def for_training(cls, corpus_filename, fold, **kwargs):
        """
        Endlessly yields (X, Y) pairs from the corpus for training.
        """
        # XXX: hardcode a lot of stuff
        # Assume there are 10 folds.
        assert 0 <= fold <= 9
        return cls(corpus_filename, training_folds(fold), **kwargs)

    @classmethod
    def for_evaluation(cls, corpus_filename, fold, **kwargs):
        """
        Endlessly yields (X, Y) pairs from the corpus for evaluation.
        """
        # XXX: hardcode a lot of stuff
        # Assume there are 10 folds.
        assert 0 <= fold <= 9
        return cls(corpus_filename, testing_folds(fold), **kwargs)


def training_folds(fold, k=10):
    """
    Return a tuple of all the training folds.
    >>> training_folds(7)
    (0, 1, 2, 3, 4, 5, 6, 8, 9)
    """
    assert fold < k
    return tuple(num for num in range(10) if num != fold)


def testing_folds(fold, k=10):
    """
    Return a tuple of all the training folds.
    >>> testing_folds(4)
    (4,)
    """
    assert fold < k
    return (fold,)


def count_samples_slow(filename, folds, sentence_length):
    corpus = CondensedCorpus.connect_to(filename)
    n_samples = 0
    for n in folds:
        for file_hash in corpus.hashes_in_fold(n):
            _, tokens = corpus[file_hash]
            n_samples += len(Sentences(tokens, size=sentence_length))
    try:
        return n_samples
    finally:
        corpus.disconnect()


def at_least(value, *args):
    """
    Returns **at least** the given number.

    For Dr. Hindle's sanity.
    """
    return max(value, *args)
