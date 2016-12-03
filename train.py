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
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

from vocabulary import vocabulary
from training_utils import LoopBatchesEndlessly, Sentences


# Based on White et al. 2015
SENTENCE_LENGTH = 20
SIGMOID_ACTIVATIONS = 300

# This is arbitrary, but it should be fairly small.
BATCH_SIZE = 512
FOLD = 0


if __name__ == '__main__':
    filename = Path('/dev/shm/tiny-corpus.sqlite3')

    # Get the tokens from the 9 training folds.
    training_batches = LoopBatchesEndlessly\
        .for_training(filename, FOLD,
                      batch_size=BATCH_SIZE,
                      sentence_length=SENTENCE_LENGTH)
    eval_batches = LoopBatchesEndlessly\
        .for_evaluation(filename, FOLD,
                      batch_size=BATCH_SIZE,
                      sentence_length=SENTENCE_LENGTH)
    print("Will train on", training_batches.samples_per_epoch, "samples")

    # Defining the model:
    model = Sequential()
    model.add(LSTM(SIGMOID_ACTIVATIONS,
                   input_shape=(SENTENCE_LENGTH, len(vocabulary))))
    model.add(Dense(len(vocabulary)))
    model.add(Activation('softmax'))

    print("Compiling the model...")
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['categorical_accuracy'])
    print("Done")

    print("Training for one epoch...")
    model.fit_generator(iter(training_batches),
                        samples_per_epoch=training_batches.samples_per_epoch,
                        validation_data=iter(eval_batches),
                        nb_val_samples=eval_batches.samples_per_epoch // BATCH_SIZE,
                        verbose=1,
                        pickle_safe=True,
                        nb_epoch=1)

    print("Saving model.")
    model.save('javascript.h5')
