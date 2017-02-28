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
from typing import Iterable

from vectors import Vectors
from sentences import Sentence, T, forward_sentences, backward_sentences
from training_utils import one_hot_batch, training_folds, evaluation_folds
from abram import at_least


class LoopBatchesEndlessly(Iterable[Sentence]):
    def __init__(self,
                 *,
                 vectors_path: Path,
                 fold: int,
                 batch_size: int,
                 sentence_length: int,
                 backwards: bool) -> None:
        assert vectors_path.exists()
        self.filename = vectors_path
        self.fold = fold
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.sentence_generator = (
            backward_sentences if backwards else forward_sentences
        )

        # Samples are number of tokens in the fold.
        vectors = Vectors.connect_to(str(vectors_path))
        self.samples_per_epoch = vectors.ntokens_in_fold(1)
        vectors.disconnect()

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
        return cls(corpus_filename, evaluation_folds(fold), **kwargs)
