#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2016, 2017 Eddie Antonio Santos <easantos@ualberta.ca>
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

import glob
import os
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple, Iterable, Sequence, Set

from sensibility.language import language
from sensibility.utils import symlink_within_dir
from .loop_batches import LoopBatchesEndlessly


Batches = Tuple[LoopBatchesEndlessly, LoopBatchesEndlessly]


class ModelDescription:
    """
    Describes a trainable LSTM model, in a specified direction (forwards OR
    backwards).

    This model is intended for evaluation, so it must belong to a "partition"
    of the full corpus.
    """
    def __init__(self, *,
                 backwards: bool,
                 base_dir: Path,
                 batch_size: int,
                 context_length: int,
                 partition: int,
                 hidden_layers: Sequence[int],
                 learning_rate: float=0.001,
                 training_set: Set[str],
                 validation_set: Set[str],
                 vectors_path: Path) -> None:
        assert base_dir.exists()
        assert vectors_path.exists()
        self.backwards = backwards
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.context_length = context_length
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.vectors_path = vectors_path

        # The training and validation data. Note, each is provided explicitly,
        # but we ask for a partition for labelling purposes.
        self.partition = partition
        self.training_set = training_set
        self.validation_set = validation_set

    @property
    def name(self) -> str:
        direction = 'b' if self.backwards else 'f'
        return f"{language.id}-{direction}{self.partition}"

    @property
    def model_path(self) -> Path:
        return self.base_dir / f"{self.name}.hdf5"

    @property
    def log_path(self) -> Path:
        return self.base_dir / f"{self.name}.csv"

    @property
    def summary_path(self) -> Path:
        return self.base_dir / f"{self.name}.json"

    @property
    def weight_path_pattern(self) -> Path:
        return self.base_dir / (
            self.name + '-{epoch:02d}-{val_loss:.4f}.hdf5'
        )

    @property
    def interrupted_path(self) -> Path:
        return self.base_dir / (self.name + '.interrupted.hdf5')

    def train(self) -> None:
        # Compile the model and prepare the training and validation samples.
        model = self.compile_model()
        training_batches, validation_batches = self.create_batches()

        self.save_summary(model)
        print(f"Training on {training_batches.samples_per_epoch} samples",
              f"using a batch size of {self.batch_size}")

        from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping  # type: ignore
        try:
            model.fit_generator(  # type: ignore
                iter(training_batches),
                nb_epoch=2**31 - 1,  # Train indefinitely
                samples_per_epoch=training_batches.samples_per_epoch,
                validation_data=iter(validation_batches),
                nb_val_samples=(
                    validation_batches.samples_per_epoch // self.batch_size
                ),
                callbacks=[
                    ModelCheckpoint(
                        str(self.weight_path_pattern),
                        save_best_only=False,
                        save_weights_only=False
                    ),
                    CSVLogger(str(self.log_path), append=True),
                    EarlyStopping(patience=3)
                ],
                verbose=1,
                pickle_safe=True,
            )
        except KeyboardInterrupt:
            model.save(str(self.interrupted_path))  # type: ignore
        finally:
            self.update_symlink()

    def compile_model(self) -> object:
        from keras.models import Sequential  # type: ignore
        from keras.layers import Dense, Activation  # type: ignore
        from keras.layers import LSTM  # type: ignore
        from keras.optimizers import RMSprop  # type: ignore

        vocabulary = language.vocabulary

        model = Sequential()
        input_shape = (self.context_length, len(vocabulary))

        if len(self.hidden_layers) == 1:
            # One LSTM layer is simple:
            first_layer = self.hidden_layers[0]
            model.add(LSTM(first_layer, input_shape=input_shape))
        else:
            first_layer, *middle_layers, last_layer = self.hidden_layers
            # The first layer defines the input, so special case it.  Since
            # there are more layers, all higher-up layers must return
            # sequences.
            model.add(LSTM(first_layer, input_shape=input_shape,
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
                      optimizer=RMSprop(lr=self.learning_rate),
                      metrics=['categorical_accuracy'])
        return model

    def create_batches(self) -> Batches:
        """
        Return a tuple of infinite training and validation examples,
        respectively.
        """
        training = LoopBatchesEndlessly(
            filehashes={'<training>'},
            vectors_path=self.vectors_path,
            batch_size=self.batch_size,
            context_length=self.context_length,
            backwards=self.backwards
        )
        validation = LoopBatchesEndlessly(
            filehashes={'<training>'},
            vectors_path=self.vectors_path,
            batch_size=self.batch_size,
            context_length=self.context_length,
            backwards=self.backwards
        )
        return training, validation

    def save_summary(self, model: object) -> None:
        # We're ready to go!
        with open(str(self.summary_path), 'w') as summary_file:
            print(model.to_json(), file=summary_file)  # type: ignore

    def update_symlink(self) -> None:
        """
        Symlinks the model to the saved model with the least validation loss,
        i.e., the best model trained so far for this partition.
        """
        # Find all existing saved models.
        pattern = self.base_dir / f"{self.name}-*-*.hdf5"
        existing_models = glob.glob(str(pattern))
        if len(existing_models) == 0:
            warnings.warn(f"No models found matching pattern: {pattern}")
            return

        # Get the model with the lowest validation loss.
        best_model = min(existing_models, key=validation_loss_by_filename)
        # Make the symlink!
        try:
            symlink_within_dir(directory=self.base_dir,
                               source=Path(Path(best_model).name),
                               target=Path(Path(self.model_path).name))
        except Exception as error:
            warnings.warn(
                f"Could not link {self.model_path} to {best_model}: {type(error)}: {error}"
            )
        finally:
            print(f"The best model is {best_model}")


def validation_loss_by_filename(filename: str) -> float:
    """
    Returns the validation loss as written in the filename.

    >>> validation_loss_by_filename("models/python-f0-02-0.9375.hdf5")
    0.9375
    >>> validation_loss_by_filename("models/javascript-f0-02-1.125.hdf5")
    1.125
    """
    match = re.search(r'(\d{1,}[.]\d{1,})[.]hdf5$', filename)
    if match:
        return float(match.group(1))
    else:
        warnings.warn(f"Could not determine loss of {filename}")
        return 0.0


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
