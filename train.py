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
from pathlib import Path
from typing import Optional, Tuple, Iterable

from vectors import Vectors
from sentences import Sentence, T, forward_sentences, backward_sentences
from vocabulary import vocabulary


# Based on White et al. 2015
SENTENCE_LENGTH = 20 + 1  # In the paper, context is set to 20
SIZE_OF_HIDDEN_LAYER = 300

# This is arbitrary, but it should be fairly small.
BATCH_SIZE = 512

# Create the argument parser.
parser = argparse.ArgumentParser(description='Train from corpus '
                                 'of vectors, with fold assignments')
parser.add_argument('-f', '--fold', type=int, required=True,
                    help='which fold to use')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--backwards', action='store_true')
group.add_argument('--forwards', action='store_false', dest='backwards')
parser.add_argument('--hidden-layer', type=int, default=SIZE_OF_HIDDEN_LAYER,
                    help='default: %d' % SIZE_OF_HIDDEN_LAYER)
parser.add_argument('--sentence-length', type=int, default=SENTENCE_LENGTH,
                    help='default: %d' % SENTENCE_LENGTH)
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                    help='default: %d' % BATCH_SIZE)
parser.add_argument('vectors_path', metavar='vectors',
                    help='corpus of vectors, with assigned folds')
#parser.add_argument('---continue', type=Path,
#                    help='Loads weights and biases from a previous run')


def compile_model(
        *,
        sentence_length: int,
        hidden_layer: int,
        learning_rate: float = 0.001
) -> object:
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers import LSTM
    from keras.optimizers import RMSprop

    # Timesteps are the context tokens.
    timesteps = sentence_length - 1
    model = Sequential()
    model.add(LSTM(hidden_layer,
                   input_shape=(timesteps, len(vocabulary))))
    # Output is number of samples x size of hidden layer
    model.add(Dense(len(vocabulary)))
    # To make a thing that looks like a probability distribution.
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=learning_rate),
                  metrics=['categorical_accuracy'])
    return model


def create_batches(batch_size: int) -> Tuple[Iterable[Sentence],
                                             Iterable[Sentence]]:
    """
    Creates training and validation batches
    """


def train(
        *,
        vectors_path: Path,
        fold: int,
        backwards: bool,
        hidden_layer: int,
        sentence_length: int,
        batch_size: int,
        previous_model: Optional[Path] = None
) -> None:
    assert vectors_path.exists()
    vectors = Vectors.connect_to(str(vectors_path))
    assert fold in vectors.fold_ids, (
        'Requested fold {} is not in {}'.format(fold, vectors.fold_ids)
    )
    vectors.disconnect()

    # TODO:
    #  - save model with architecture after each epoch
    #  - save acc, val_acc, loss, val_loss after each epoch
    #       - keras.callbacks.ModelCheckpoint.
    #  - point a symlink at the best model after each epoch

    model = compile_model(sentence_length=20,
                          hidden_layer=hidden_layer)

    """
    print("Creating batches with size:", batch_size)
    training_batches, eval_batches = recipe.create_batches(
        vector_filename, batch_size
    )

    print("Will train on", training_batches.samples_per_epoch, "samples")
    print("Training:", recipe.filename)
    model.fit_generator(iter(training_batches),
                        samples_per_epoch=training_batches.samples_per_epoch,
                        validation_data=iter(eval_batches),
                        nb_val_samples=eval_batches.samples_per_epoch // batch_size,
                        verbose=1,
                        pickle_safe=True,
                        nb_epoch=1)

    print("Saving model to", recipe.filename)
    model.save(recipe.filename)
    """


if __name__ == '__main__':
    args = parser.parse_args()
    train(**vars(args))
