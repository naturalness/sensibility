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

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, Iterable, Sequence

from vectors import Vectors
from loop_batches import LoopBatchesEndlessly
from vocabulary import vocabulary


# Based on White et al. 2015
HIDDEN_LAYERS = (300,)
CONTEXT_LENGTH = 20

# This is arbitrary, but it should be fairly small.
BATCH_SIZE = 512

MAX_EPOCHS = 30


def layers(string: str) -> Sequence[int]:
    """
    Parse hidden layer notation.

    >>> layers('2000')
    (2000,)
    >>> layers('300,300,300')
    (300, 300, 300)
    """
    result = tuple(int(layer) for layer in string.split(','))
    assert len(result) >= 1, "Must define at least one hidden layer!"
    return result


# Create the argument parser.
parser = argparse.ArgumentParser(description='Train from corpus '
                                 'of vectors, with fold assignments')
parser.add_argument('-f', '--fold', type=int, required=True,
                    help='which fold to use')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--backwards', action='store_true')
group.add_argument('--forwards', action='store_false', dest='backwards')
parser.add_argument('--hidden-layers', type=layers, default=HIDDEN_LAYERS,
                    help='default: %r' % HIDDEN_LAYERS)
parser.add_argument('--context-length', type=int, default=CONTEXT_LENGTH,
                    help='default: %d' % CONTEXT_LENGTH)
parser.add_argument('--max-epochs', type=int, default=None,
                    help='default: train forever')
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                    help='default: %d' % BATCH_SIZE)
parser.add_argument('--base-dir', type=Path, default=Path('~/Backups/models/'),
                    help='default: %d' % BATCH_SIZE)
parser.add_argument('vectors_path', type=Path, metavar='vectors',
                    help='corpus of vectors, with assigned folds')
parser.add_argument('---continue', type=Path, dest='previous_model',
                    help='Loads weights and biases from a previous run')


def compile_model(
        *,
        context_length: int,
        hidden_layers: Sequence[int],
        learning_rate: float=0.001
) -> object:
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers import LSTM
    from keras.optimizers import RMSprop

    model = Sequential()

    if len(hidden_layers) == 1:
        # One LSTM layer is simple:
        first_layer = hidden_layers[0]
        model.add(LSTM(first_layer,
                       input_shape=(context_length, len(vocabulary))))
    else:
        first_layer, *middle_layers, last_layer = hidden_layers
        # The first layer defines the input, so special case it.  Since there
        # are more layers, all higher-up layers must return sequences.
        model.add(LSTM(first_layer,
                       input_shape=(context_length, len(vocabulary)),
                       return_sequences=True))
        # Add the middle LSTM layers (if any).
        # These layers must also return sequences.
        for layer in middle_layers:
            model.add(LSTM(layer, return_sequences=True))
        # Add the final layer.
        model.add(LSTM(last_layer))

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
        hidden_layers: Sequence[int],
        context_length: int,
        batch_size: int,
        max_epochs: Optional[int],
        base_dir: Path,
        previous_model: Optional[Path]=None
) -> None:
    # Ensure the fold exists.
    assert vectors_path.exists()
    vectors = Vectors.connect_to(str(vectors_path))
    assert fold in vectors.fold_ids, (
        'Requested fold {} is not in {}'.format(fold, vectors.fold_ids)
    )
    vectors.disconnect()

    # Compile the model and prepare the training and validation samples.
    model = compile_model(context_length=context_length,
                          hidden_layers=hidden_layers)
    training_batches, validation_batches = create_batches(
        fold=fold,
        backwards=backwards,
        batch_size=batch_size,
        vectors_path=vectors_path,
        context_length=context_length
    )

    name = ','.join(str(layer) for layer in hidden_layers)
    model_dir = (base_dir / name).expanduser()
    assert not model_dir.exists(), "refusing to overwrite " + str(model_dir)

    # Create the directory.
    os.mkdir(str(model_dir))

    # We're ready to go!
    with open(str(model_dir / 'summary.txt'), 'w') as summary_file:
        print(model.to_json(), file=summary_file)  # type: ignore

    print("Saving data to", str(model_dir))
    print("Training on", training_batches.samples_per_epoch, "samples",
          "using a batch size of", batch_size)

    from keras.callbacks import ModelCheckpoint, CSVLogger
    try:
        model.fit_generator(  # type: ignore
            iter(training_batches),
            validation_data=iter(validation_batches),
            nb_epoch=max_epochs or 2**31 - 1,  # a long-ass time
            samples_per_epoch=training_batches.samples_per_epoch,
            nb_val_samples=validation_batches.samples_per_epoch // batch_size,
            callbacks=[
                ModelCheckpoint(
                    str(model_dir / 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                    save_best_only=False,
                    save_weights_only=False
                ),
                CSVLogger(str(model_dir / 'training.log'), append=True)
            ],
            verbose=1,
            pickle_safe=True,
        )
    except KeyboardInterrupt:
        filename = str(model_dir / 'interrupted.hdf5')
        print("Saving model to", filename)
        model.save(filename)  # type: ignore


if __name__ == '__main__':
    args = parser.parse_args()
    train(**vars(args))
