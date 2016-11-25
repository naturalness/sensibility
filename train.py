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
Trains an LSTM from sentences in the vectorized corpus.
"""

import argparse

import numpy as np
from path import Path
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop

from vocabulary import vocabulary
from condensed_corpus import CondensedCorpus


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

        # Fill in the vectors.
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


class LoopSentencesEndlessly:
    """

    This test only works on my computer...

    >>> loop = LoopSentencesEndlessly.for_training('trial.sqlite3', fold=0)
    >>> x, y = next(iter(loop))
    """
    def __init__(self, corpus_filename, folds):
        assert Path(corpus_filename).exists()
        self.filename = corpus_filename
        self.folds = folds
        self.corpus = None

    def __iter__(self):
        self.corpus = corpus = CondensedCorpus.connect_to(self.filename)
        while True:
            for fold in self.folds:
                for file_hash in corpus.hashes_in_fold(fold):
                    _, tokens = corpus[file_hash]
                    yield from Sentences(tokens)

    def __del__(self):
        if self.corpus is not None:
            self.corpus.disconnect()

    @classmethod
    def for_training(cls, corpus_filename, fold):
        """
        Endlessly yields (X, Y) pairs from the corpus for training.
        """
        # XXX: hardcode a lot of stuff
        # Assume there are 10 folds.
        assert 0 <= fold <= 9
        training_folds = tuple(num for num in range(10) if num != fold)
        assert len(training_folds) == 9
        return cls(corpus_filename, training_folds)

    @classmethod
    def for_evaluation(cls, corpus_filename, fold):
        """
        Endlessly yields (X, Y) pairs from the corpus for evaluation.
        """
        # XXX: hardcode a lot of stuff
        # Assume there are 10 folds.
        assert 0 <= fold <= 9
        return cls(corpus_filename, (fold,))


def at_least(value, *args):
    """
    Returns **at least** the given number.

    For Dr. Hindle's sanity.
    """
    return max(value, *args)


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=Path)

def define_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(DEFAULT_SIZE, len(vocabulary))))
    model.add(Dense(len(vocabulary)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def main():
    args = parser.parse_args()
    assert args.filename.exists()

    # define a model
    model = define_model()

    # Number of tokens divided by 10?
    NUM_SAMPLES = 150000000

    # train the model
    training_data = LoopSentencesEndlessly.for_training(args.filename, fold=0)
    history = model.fit_generator(training_data,
                                  nb_epoch=10,
                                  samples_per_epoch=NUM_SAMPLES,
                                  pickle_safe=True)

    model.save('javascript')
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
