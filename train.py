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
from loop_batches import LoopBatchesEndlessly
from vocabulary import vocabulary


# Based on White et al. 2015
SIZE_OF_HIDDEN_LAYER = 300
CONTEXT_LENGTH = 20

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
parser.add_argument('--context-length', type=int, default=CONTEXT_LENGTH,
                    help='default: %d' % CONTEXT_LENGTH)
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                    help='default: %d' % BATCH_SIZE)
parser.add_argument('vectors_path', metavar='vectors',
                    help='corpus of vectors, with assigned folds')
parser.add_argument('---continue', type=Path,
                    help='Loads weights and biases from a previous run')


def compile_model(
        *,
        context_length: int,
        hidden_layer: int,
        learning_rate: float=0.001
) -> object:
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers import LSTM
    from keras.optimizers import RMSprop

    model = Sequential()
    model.add(LSTM(hidden_layer,
                   input_shape=(context_length, len(vocabulary))))
    # Output is number of samples x size of hidden layer
    model.add(Dense(len(vocabulary)))
    # To make a thing that looks like a probability distribution.
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=learning_rate),
                  metrics=['categorical_accuracy'])
    return model


def create_batches(*, fold: int, **kwargs) -> Tuple[LoopBatchesEndlessly,
                                                    LoopBatchesEndlessly]:
    training = LoopBatchesEndlessly.for_training(fold=fold, **kwargs)
    validation = LoopBatchesEndlessly.for_validation(fold=fold + 5, **kwargs)
    return training, validation


def train(
        *,
        vectors_path: Path,
        fold: int,
        backwards: bool,
        hidden_layer: int,
        context_length: int,
        batch_size: int,
        previous_model: Optional[Path]=None
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

    model = compile_model(context_length=20,
                          hidden_layer=hidden_layer)

    training_batches, validation_batches = create_batches(
        fold=fold,
        backwards=backwards,
        batch_size=batch_size,
        vectors_path=vectors_path,
        context_length=context_length
    )
    print("Will train on", training_batches.samples_per_epoch, "samples",
          "using a batch size of", batch_size)

    from keras.callbacks import ModelCheckpoint, CSVLogger
    try:
        model.fit_generator(  # type: ignore
            iter(training_batches),
            validation_data=iter(validation_batches),
            nb_epoch=2**31 - 1,  # a long-ass time
            samples_per_epoch=training_batches.samples_per_epoch,
            nb_val_samples=validation_batches.samples_per_epoch // batch_size,
            callbacks=[
                ModelCheckpoint(
                    './models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                    save_best_only=False,
                    save_weights_only=False
                ),
                CSVLogger('./models/training.log', append=True)
            ],
            verbose=1,
            pickle_safe=True,
        )
    except KeyboardInterrupt:
        filename = './models/interrupted.hdf5'
        print("Saving model to", filename)
        model.save(filename)  # type: ignore


if __name__ == '__main__':
    args = parser.parse_args()
    train(**vars(args))
