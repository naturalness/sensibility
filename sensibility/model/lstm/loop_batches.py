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

from pathlib import Path
from typing import Iterable, Iterator, Sequence, Set, Tuple, cast

import numpy as np
from more_itertools import chunked

from sensibility.evaluation.vectors import Vectors
from sensibility.sentences import Sentence, T, forward_sentences, backward_sentences
from sensibility.language import language


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
        self.filename = str(vectors_path)
        self.filehashes = filehashes
        self.batch_size = batch_size
        self.context_length = context_length
        self.sentence_generator = (
            backward_sentences if backwards else forward_sentences
        )

        # Samples are number of tokens in the filehash set.
        # TODO: Get number of samples: sum the number of tokens
        # in each partition
        self.samples_per_epoch = get_samples_per_batches_hack(filehashes)

    def __iter__(self) -> Iterator[Batch]:
        batch_size = self.batch_size
        for batch in self._yield_batches_endlessly():
            yield one_hot_batch(batch,
                                batch_size=batch_size,
                                context_length=self.context_length)

    def _yield_sentences_from_corpus(self) -> Iterable[Sentence]:
        """
        Yields all sentences from the corpus exactly once.
        """
        context_length = self.context_length
        # XXX: determines from language
        vectors = Vectors()
        for filehash in self.filehashes:
            tokens = vectors[filehash]
            yield from self.sentence_generator(
                cast(Sequence[int], tokens), context=context_length
            )
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


def get_samples_per_batches_hack(hashes: Set[str]) -> int:
    """
    Temporary.

    >>> from sensibility.language import language
    >>> language.set_language('python') and None
    >>> s = {'7bd4a8a55e103450d99a049ac68daa0edee1646d416fbc96c97eeb73ff8a28d0'}
    >>> get_samples_per_batches_hack(s)
    252
    """
    # XXX: This should take from arguments, but relies on globals instead.
    from sensibility.evaluation.vectors import determine_from_language
    conn = determine_from_language()
    query = r'SELECT length(array) FROM vector where filehash = ?'
    n_tokens = 0
    for filehash in hashes:
        (count,), = conn.execute(query, (filehash,))
        n_tokens += count
    return n_tokens


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
