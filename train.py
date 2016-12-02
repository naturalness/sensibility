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
from itertools import islice

import numpy as np
from path import Path
from tqdm import tqdm
from more_itertools import chunked

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop

from vocabulary import vocabulary
from condensed_corpus import CondensedCorpus
from training_utils import Sentences, one_hot_batch


# Based on White et al. 2015
SENTENCE_LENGTH = 20
SIGMOID_ACTIVATIONS = 300

# This is arbitrary, but it should be fairly small.
#BATCH_SIZE = 128
BATCH_SIZE = 1024

if __name__ == '__main__':
    #filename = Path('/run/user/1004/corpus.sqlite3')
    #filename = Path('/run/user/1004/small-corpus.sqlite3')
    filename = Path('/dev/shm/vectors.sqlite3')
    assert filename.exists()

    # Define a model:
    model = Sequential()

    model.add(LSTM(SIGMOID_ACTIVATIONS,
                   input_shape=(SENTENCE_LENGTH, len(vocabulary))))
    model.add(Dense(len(vocabulary)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)

    print("Compiling the model...")
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print("Done")

    # Get the tokens from the 9 training folds.
    FOLD = 0
    training_folds = tuple(num for num in range(10) if num != FOLD)

    def generate_sentences(folds):
        corpus = CondensedCorpus.connect_to(filename)
        for fold in folds:
            for file_hash in corpus.hashes_in_fold(fold):
                _, tokens = corpus[file_hash]
                yield from Sentences(tokens, size=SENTENCE_LENGTH)
        corpus.disconnect()

    batch_of_vectors = chunked(generate_sentences(training_folds), BATCH_SIZE)

    print("Training")
    progress = tqdm(batch_of_vectors)
    for batch in progress:
        vocab_size = len(vocabulary)

        # Create empty one-hot vectors
        x = np.zeros((BATCH_SIZE, SENTENCE_LENGTH, vocab_size), dtype=np.bool)
        y = np.zeros((BATCH_SIZE, vocab_size), dtype=np.bool)

        # Fill in the vectors.
        for sentence_id, (sentence, last_token_id) in enumerate(batch):
            # Fill in the one-hot matrix for X
            for pos, token_id in enumerate(sentence):
                x[sentence_id, pos, token_id] = 1

            # Add the last token for the one-hot vector Y.
            y[sentence_id, last_token_id] = 1

        loss, acc = model.train_on_batch(x, y)
        progress.set_description("Loss => {}, acc => {}".format(loss, acc))

    model.save('javascript.h5')

    print("Evaluating")
    batch_of_vectors = chunked(generate_sentences((9,)), BATCH_SIZE)
    progress = tqdm(batch_of_vectors)
    for batch in progress:
        loss, acc = model.test_on_batch(x, y)
        progress.set_description("Loss => {}, acc => {}".format(loss, acc))

