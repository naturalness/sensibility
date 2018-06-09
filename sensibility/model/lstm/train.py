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

import argparse
import json
import logging
import os
import sys
import typing
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, cast

from sensibility._paths import (get_training_set_path, get_validation_set_path,
                                get_vectors_path)
from sensibility.language import language
from sensibility.miner.util import filehashes
from sensibility.utils import symlink_within_dir

from .loop_batches import LoopBatchesEndlessly

# === Default command line arguments === #

# Based on (practically inapplicable) findings by Bhatia and Singh 2016
HIDDEN_LAYERS = (128,)
CONTEXT_LENGTH = 9

# This is arbitrary, but it should be fairly small.
BATCH_SIZE = 32
# Ideally, you'll find the right value for this by experimentation.
LEARNING_RATE = 0.001

PATIENCE = 3
OPTIMIZER = 'rmsprop'

# === Other module-wide globals === #

logger = logging.getLogger(__name__)
INDEFINITE = 2 ** 32 - 1  # Yes, 2**32 is technically infinity


# === The big training class === #

class ModelDescription:
    """
    Describes a trainable LSTM model, in a specified direction (forwards OR
    backwards).

    This model is intended for evaluation, so it must belong to a "partition"
    of the full corpus.
    """

    # A pair of a training and a validation batch generators .
    Batches = Tuple[LoopBatchesEndlessly, LoopBatchesEndlessly]
    # Keras is poorly behaved, so only import these types when type-checking.
    if typing.TYPE_CHECKING:
        from keras.models import Sequential

    def __init__(self, *,
                 backwards: bool,
                 output_dir: Path,
                 batch_size: int,
                 context_length: int,
                 partition: int,
                 hidden_layers: Sequence[int],
                 learning_rate: float,
                 patience: int,
                 dropout: Optional[float],
                 optimizer: str,
                 training_set: Set[str],
                 validation_set: Set[str],
                 vectors_path: Path) -> None:

        self.backwards = backwards
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.context_length = context_length
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.optimizer = optimizer
        self.patience = patience

        # The training and validation data. Note, each is provided explicitly,
        # but we ask for a partition for labelling purposes.
        self.partition = partition
        self.training_set = training_set
        self.validation_set = validation_set
        self.vectors_path = vectors_path

    def train(self) -> None:
        """
        Start traing this model.
        """
        if self.incomplete_path.exists():
            self.train_from_existing()
        else:
            self.train_from_scratch()

    def train_from_scratch(self) -> None:
        """
        Start training in a temporary directory.
        """
        assert not self.output_dir.exists()
        assert not self.incomplete_path.exists()
        self.incomplete_path.mkdir()
        self.save_manifest()
        self._train()

    def train_from_existing(self) -> None:
        """
        Continue training from an existing directory.
        """
        assert self.incomplete_path.exists()

        # Continue from a saved intermediate model.
        intermediates = list(self.incomplete_path.glob('intermediate-*.hdf5'))

        if len(intermediates) > 0:
            # There are intermediate files!
            # Figure out the last epoch, and continue from there.
            last_epoch = max(intermediates, key=epoch_from_path)
            self._train(continue_from=last_epoch)
        else:
            # There might not be an intermediate file, so pretend we're
            # starting from scratch.
            self._train()

    def _train(self, continue_from: Path = None) -> None:
        logger.info("Saving model to %s", self.model_path)
        logger.info("%d training files", len(self.training_set))
        logger.info("%d validation files", len(self.validation_set))
        logger.info("Loading file vectors from %s", self.vectors_path)

        # Compile the model and prepare the training and validation samples.
        self._ensure_vectors_exist()
        training_batches, validation_batches = self.create_batches()

        logger.info(f"Training on {training_batches.samples_per_epoch} samples "
                    f"using a batch size of {self.batch_size}")

        from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
        model = self.compile_model()

        # Load weights if we're continuing
        if continue_from is None:
            initial_epoch = 0
        else:
            logger.info('Continuing from %s', continue_from)
            model.load_weights(str(continue_from))
            initial_epoch = epoch_from_path(continue_from) + 1

        try:
            model.fit_generator(
                iter(training_batches),
                self._batches_per_epoch(training_batches.samples_per_epoch),
                epochs=INDEFINITE,  # Train until EarlyStopping says so.
                validation_data=iter(validation_batches),
                validation_steps=self._batches_per_epoch(validation_batches.samples_per_epoch),
                verbose=0,  # Use a callback instead to monitor progress.
                callbacks=[
                    ModelCheckpoint(str(self.weight_path_pattern),
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode='auto'),
                    CSVLogger(str(self.progress_path), append=True),
                    EarlyStopping(patience=self.patience, mode='auto')
                ],
                use_multiprocessing=True,
                initial_epoch=initial_epoch,
            )
        except KeyboardInterrupt:
            model.save(str(self.interrupted_path))
        else:
            # Move the file over to the correct file path, so that Make can
            # confirm that this model has completed training.
            self.incomplete_path.rename(self.output_dir)

    def compile_model(self) -> 'Sequential':
        from keras.models import Sequential
        from keras.layers import Dense, Activation, Dropout
        from keras.layers import LSTM
        from keras.optimizers import Optimizer, RMSprop, Nadam, Adam

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

        # Add some form of regularization:
        # http://theorangeduck.com/page/neural-network-not-working#dropout
        # The rate should ideally be tweaked according to the size and nature
        # of the training data.
        if self.dropout is not None:
            model.add(Dropout(self.dropout))
        # Output is number of samples x size of hidden layer
        model.add(Dense(len(vocabulary)))
        # Softmax makes the output look like a probability distribution.
        model.add(Activation('softmax'))

        optimizer: Optimizer
        if self.optimizer == 'rmsprop':
            optimizer = RMSprop(lr=self.learning_rate)
        elif self.optimizer == 'adam':
            optimizer = Adam(lr=self.learning_rate)
        elif self.optimizer == 'nadam':
            optimizer = Nadam(lr=self.learning_rate)
        else:
            logger.error("Unknown optimizer: %r", self.optimizer)
            sys.exit(2)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['categorical_accuracy'])
        return model

    def create_batches(self) -> Batches:
        """
        Return a tuple of infinite training and validation examples,
        respectively.
        """
        training = LoopBatchesEndlessly(
            filehashes=self.training_set,
            vectors_path=self.vectors_path,
            batch_size=self.batch_size,
            context_length=self.context_length,
            backwards=self.backwards
        )
        validation = LoopBatchesEndlessly(
            filehashes=self.validation_set,
            vectors_path=self.vectors_path,
            batch_size=self.batch_size,
            context_length=self.context_length,
            backwards=self.backwards
        )
        return training, validation

    def save_manifest(self) -> None:
        """
        Saves a manifest with all of the relevant parameters for training this
        model.
        """
        properties = (
            'direction partition training_set_size validation_set_size '
            'hidden_layers context_length batch_size '
            'dropout optimizer learning_rate patience '
        ).split()

        manifest = {prop: getattr(self, prop) for prop in properties}
        with open(self.manifest_path, 'w') as summary_file:
            json.dump(manifest, summary_file, indent=4)

    def _batches_per_epoch(self, training_samples: int) -> int:
        """
        Number of batches per sample.
        """
        # TODO: Move this method to batch?
        return (training_samples // self.batch_size +
                bool(training_samples % self.batch_size))

    def _ensure_vectors_exist(self) -> None:
        if not self.vectors_path.exists():
            raise Exception(f"Could not find vectors at: {self.vectors_path}")

    @property
    def incomplete_path(self) -> Path:
        return Path(str(self.output_dir) + '.incomplete')

    @property
    def model_path(self) -> Path:
        return self.incomplete_path / f"model.hdf5"

    @property
    def progress_path(self) -> Path:
        return self.incomplete_path / f"progress.csv"

    @property
    def manifest_path(self) -> Path:
        return self.incomplete_path / f"manifest.json"

    @property
    def weight_path_pattern(self) -> Path:
        return self.incomplete_path / (
            'intermediate-{val_loss:.4f}-{epoch:02d}.hdf5'
        )

    @property
    def interrupted_path(self) -> Path:
        return self.incomplete_path / 'interrupted.hdf5'

    @property
    def training_set_size(self) -> int:
        return len(self.training_set)

    @property
    def validation_set_size(self) -> int:
        return len(self.validation_set)

    @property
    def direction(self) -> str:
        return 'backwards' if self.backwards else 'forwards'


def validation_loss(filename: Path) -> float:
    """
    Determine the validation loss encoded in a model filename.

    >>> validation_loss(Path('models/intermediate-2.1664-00.hdf5'))
    2.1664
    """
    import re
    pat = re.compile(r'intermediate-(\d+\.\d+)-')
    m = pat.match(filename.stem)
    if m is None:
        raise ValueError(f"Not an intermediate model filename: {filename}")
    return float(m.group(1))


def epoch_from_path(filename: Path) -> int:
    """
    Determine the epoch encded in a model filename.

    >>> epoch_from_path(Path('models/intermediate-2.1664-07.hdf5'))
    7
    """
    return int(filename.stem.split('-')[-1], base=10)


def path_to_best_model(model_dir: Path) -> Path:
    """
    Given a directory, returns the path to the model with the lowest
    validation loss.
    """
    assert model_dir.is_dir()
    models = list(model_dir.glob('intermediate-*.hdf5'))
    return min(models, key=validation_loss)


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
parser = argparse.ArgumentParser(description='Train from corpus of vectors')

# Input options
parser.add_argument('-p', '--partition', type=int, required=True,
                    help='which partition to use')
group = parser.add_mutually_exclusive_group(required=True)

# Output options
parser.add_argument('-o', '--output-dir', type=Path, required=True,
                    help=f"Name of the directory to output")

# LSTM options.
group.add_argument('-f', '--forwards', action='store_true')
group.add_argument('-b', '--backwards', action='store_true')
parser.add_argument('--hidden-layers', type=layers, default=HIDDEN_LAYERS,
                    help=f"default: {HIDDEN_LAYERS}")
parser.add_argument('--context-length', type=int, default=CONTEXT_LENGTH,
                    help=f"default: {CONTEXT_LENGTH}")
parser.add_argument('--train-set-size', type=int,
                    default=11_000, help='Number of files to train on')
parser.add_argument('--validation-set-size', type=int,
                    default=5500, help='Number of files to validate against')
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                    help=f"default: {BATCH_SIZE}")
parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                    help=f"default: {LEARNING_RATE}")
parser.add_argument('--dropout', type=float, default=None,
                    help=f"default: disabled")
parser.add_argument('--optimizer', choices=('rmsprop', 'adam', 'nadam'),
                    default=OPTIMIZER, help=f"default: {OPTIMIZER}")
parser.add_argument('--patience', type=int, default=PATIENCE,
                    help='Number of bad epochs to wait before stopping'
                    f' (default: {PATIENCE})')

# GPU settings.
parser.add_argument('--gpu', type=int, default=None,
                    help=f"Which GPU to use.")


def main() -> None:
    # TODO: Figure out where to dump logging info
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    # Get the appropriate sets for the give partition
    partition = cast(int, args.partition)
    training_set = slurp(get_training_set_path(partition))
    validation_set = slurp(get_validation_set_path(partition))

    configure_gpu(args.gpu)

    # Determine language first!
    model = ModelDescription(
        partition=partition,
        training_set=subset(training_set, args.train_set_size),
        validation_set=subset(validation_set, args.validation_set_size),
        vectors_path=get_vectors_path(),
        backwards=args.backwards,
        output_dir=args.output_dir,
        context_length=args.context_length,
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
        patience=args.patience,
        batch_size=args.batch_size,
        dropout=args.dropout,
        optimizer=args.optimizer,
    )

    model.train()


def slurp(filename: Path) -> List[str]:
    """
    Read the file into one big list of filehashes.
    """
    with open(filename, 'r') as hashes_file:
        return list(filehashes(hashes_file))


def subset(xs: List[str], max_size: int) -> Set[str]:
    """
    Return a subset of files from the list.
    """
    return set(xs[:max_size])


def full_cores_available() -> int:
    # TODO: use this? lscpu -pcore | sed '/^#/d' | sort -u
    threads = os.cpu_count()
    if threads is None:
        raise RuntimeError("Could not determine CPU cores")
    else:
        # HACK FOR FOS: It's hyperthreaded so half the number of logical CPUs.
        return threads // 2


def configure_gpu(prefered: Optional[int]) -> None:
    """
    Configure the CUDA GPU (if applicable).
    """
    try:
        import GPUtil  # type: ignore
    except ImportError:
        warnings.warn("Could not import GPUtil: using default GPU.")
    else:
        # Use an equal slice of the machine as the number of physical cores.
        # This is usually the ideal number of processes before resource contention.
        max_resources = 1. - 1. / full_cores_available()
        limits = dict(maxLoad=max_resources, maxMemory=max_resources)
        if prefered is None:
            # Select an available GPU.
            device_id, = GPUtil.getFirstAvailable(
                order='random',
                attempts=2 * 60,  # try for an hour to get a GPU
                interval=30,  # Try every 30 seconds.
                verbose=True,
                **limits
            )
        else:
            # Get the GPUs with the LEAST memory utilization.
            available = GPUtil.getAvailable(order='memory', limit=16, **limits)
            assert len(available) >= 1
            if prefered in available:
                device_id = prefered
            else:
                logger.warn("Requested GPU %d unavaible", prefered)
                # Since the preference is unavailable, use the GPU with the least
                # allocated memory.
                device_id = available[0]

        # Set the prefered GPU ID.
        logger.info("Using GPU %d", device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


if __name__ == '__main__':
    main()
