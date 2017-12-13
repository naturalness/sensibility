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
Loops batches for training and for validation (development) forever.
"""

import logging
from pathlib import Path
from random import shuffle
from typing import Iterable, Iterator, Sequence, Set, Tuple, Union, cast

import numpy as np
from more_itertools import chunked

from sensibility.evaluation.vectors import Vectors
from sensibility.language import language
from sensibility.sentences import (Sentence, T, backward_sentences,
                                   forward_sentences)

Batch = Tuple[np.ndarray, np.ndarray]


class LoopBatchesEndlessly(Iterable[Batch]):
    """
    Loops batches of vectors endlessly from the given filehashes.
    """

    def __init__(self, *,
                 vectors_path: Path,
                 filehashes: Set[str],
                 batch_size: int,
                 context_length: int,
                 backwards: bool) -> None:
        assert vectors_path.exists()
        self.filename = vectors_path
        self.filehashes = list(filehashes)
        self.batch_size = batch_size
        self.context_length = context_length
        self.sentence_generator = (
            backward_sentences if backwards else forward_sentences
        )

        # Samples are number of tokens in the filehash set.
        self.samples_per_epoch = (
            Vectors.from_filename(vectors_path)
                   .length_of_vectors(filehashes)
        )

    def __iter__(self) -> Iterator[Batch]:
        logger = logging.getLogger(type(self).__name__)
        batch_size = self.batch_size
        for batch in self._yield_batches_endlessly():
            logger.debug("Batch{%s}", LogBatch(batch))
            yield one_hot_batch(batch,
                                batch_size=batch_size,
                                context_length=self.context_length)

    def _yield_sentences_from_corpus(self) -> Iterable[Sentence]:
        """
        Yields all sentences from the corpus exactly once.
        """
        context_length = self.context_length

        # Shuffle the files on each iteration.
        shuffle(self.filehashes)

        vectors = Vectors.from_filename(self.filename)
        for filehash in self.filehashes:
            # Shuffle sentences randomly from each file.
            # This minimizes class imbalance per batch in languages that might
            # exhibit a large amount of repeating tokens like
            #   <identifier> <identifier> = <identifier> . <identifier> ;
            # *cough* java *cough*
            tokens = cast(Sequence[int], vectors[filehash])
            sentences = list(self.sentence_generator(tokens, context=context_length))
            shuffle(sentences)
            yield from sentences
        vectors.disconnect()

    def _yield_batches_endlessly(self):
        """
        Yields batches of samples, in vectorized format, but NOT one-hot
        encoded.
        """
        batch_size = self.batch_size
        while True:
            yield from chunked(self._yield_sentences_from_corpus(),
                               batch_size)


def one_hot_batch(batch, *,
                  batch_size: int,
                  context_length: int,
                  vocabulary_size: int=None) -> Batch:
    """
    Creates one hot vectors (x, y arrays) of the batch.

    >>> x, y = one_hot_batch([(np.array([36]), 48)],
    ...                      batch_size=1024,
    ...                      context_length=20,
    ...                      vocabulary_size=100)
    >>> x.shape
    (1, 20, 100)
    >>> x[0, 0, 36]
    1
    >>> y.shape
    (1, 100)
    >>> y[0, 48]
    1
    """
    if vocabulary_size is None:
        vocabulary_size = len(language.vocabulary)
    # Create empty one-hot vectors
    x: np.ndarray[bool] = np.zeros((batch_size, context_length, vocabulary_size), dtype=np.bool)
    y: np.ndarray[bool] = np.zeros((batch_size, vocabulary_size), dtype=np.bool)

    # Fill in the vectors.
    for sentence_id, (sentence, last_token_id) in enumerate(batch):
        # Fill in the one-hot matrix for X
        for pos, token_id in enumerate(sentence):
            x[sentence_id, pos, token_id] = True

        # Add the last token for the one-hot vector Y.
        y[sentence_id, last_token_id] = True

    samples_produced = sentence_id + 1

    if samples_produced < batch_size:
        x = np.resize(x, ((samples_produced, context_length, vocabulary_size)))
        y = np.resize(y, ((samples_produced, vocabulary_size)))

    return x, y


Number = Union[int, float]


class LogBatch:
    """
    A hacky class to log the targets per batch.  This is to debug class
    imbalance issues.
    """
    __slots__ = 'batch',

    def __init__(self, batch: Sequence[Sentence]) -> None:
        self.batch = batch

    def __str__(self) -> str:
        from collections import Counter
        counter = Counter(target for _context, target in self.batch)

        def generate_parts():
            total = len(self.batch)
            accounted_for = 0
            for target, count in counter.most_common(5):
                token = language.vocabulary.to_text(target)
                yield f"{token}: {Pct(count, total)}"
                accounted_for += count
            remaining = total - accounted_for
            if remaining:
                yield f"[other]: {Pct(remaining, total)}"
        return ', '.join(generate_parts())


class Pct:
    __slots__ = 'pct',

    def __init__(self, amount: Number, total: Number) -> None:
        self.pct = 100. * amount / total

    def __format__(self, _format) -> str:
        return f"{self.pct:6.2f}%"
