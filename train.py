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

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.visualize_util import plot

from vocabulary import vocabulary
from condensed_corpus import CondensedCorpus


# Based on White et al. 2015
DEFAULT_SIZE = 20
SIGMOID_ACTIVATIONS = 300

filename = Path('/run/user/1004/corpus.sqlite3')
assert filename.exists()

batch_size = 128

# define a model
model = Sequential()

model.add(LSTM(SIGMOID_ACTIVATIONS, input_shape=(DEFAULT_SIZE, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.001)

print("Compiling the model...")
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
print("Done")

# train the model
print("Training")
training_data = LoopSentencesEndlessly.for_training(args.filename, fold=0)
history = model.fit_generator(iter(training_data),
                              nb_epoch=10,
                              samples_per_epoch=n_samples,
                              verbose=2,
                              pickle_safe=True)

model.save('javascript')
