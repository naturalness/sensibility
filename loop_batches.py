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
from more_itertools import chunked
from typing import Iterable, Iterator, Sequence, cast, Any

from vectors import Vectors
from sentences import Sentence, T, forward_sentences, backward_sentences
from training_utils import one_hot_batch, training_folds, evaluation_folds


class LoopBatchesEndlessly(Iterable[Any]):
    def __init__(self,
                 *,
                 vectors_path: Path,
                 fold: int,
                 batch_size: int,
                 context_length: int,
                 backwards: bool) -> None:
        assert vectors_path.exists()
        self.filename = str(vectors_path)
        self.fold = fold
        self.batch_size = batch_size
        self.context_length = context_length
        self.sentence_generator = (
            backward_sentences if backwards else forward_sentences
        )

        # Samples are number of tokens in the fold.
        vectors = Vectors.connect_to(self.filename)
        self.samples_per_epoch = vectors.ntokens_in_fold(self.fold)
        vectors.disconnect()

    def __iter__(self) -> Iterator[Any]:
        batch_size = self.batch_size
        for batch in self._yield_batches_endlessly():
            yield one_hot_batch(batch, batch_size=batch_size,
                                # This parameter name is weird...
                                sentence_length=self.context_length)

    def _yield_sentences_from_corpus(self) -> Iterable[Sentence]:
        """
        Yields all sentences from the corpus exactly once.
        """
        context_length = self.context_length
        vectors = Vectors.connect_to(self.filename)
        for file_hash in vectors.hashes_in_fold(self.fold):
            _, tokens = vectors[file_hash]
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

    @classmethod
    def for_training(cls, fold: int, **kwargs) -> 'LoopBatchesEndlessly':
        """
        Endlessly yields (X, Y) pairs from the corpus for training.
        """
        # XXX: hardcode a lot of stuff
        # Assume there are 5 training folds.
        assert 0 <= fold < 5
        return cls(fold=fold, **kwargs)

    @classmethod
    def for_validation(cls, fold: int, **kwargs) -> 'LoopBatchesEndlessly':
        """
        Endlessly yields (X, Y) pairs from the corpus for validation.
        """
        # XXX: hardcoded assumptions
        # Assume there are 5 validation folds
        assert 5 < fold < 9
        return cls(fold=fold, **kwargs)
