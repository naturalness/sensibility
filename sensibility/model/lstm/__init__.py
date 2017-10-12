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

import logging
import os
import sys
from abc import ABC
from pathlib import Path
from typing import Any, Sequence, Iterable, NamedTuple, Type, Union
from typing import TYPE_CHECKING

from .loop_batches import one_hot_batch
from sensibility.vocabulary import Vind

import numpy as np

# Keras is poorly-behaved and does things when you import it,
# so only import it statically (during type-checking).
if TYPE_CHECKING:
    from keras.models import Model


class TokenResult(NamedTuple):
    forwards: np.ndarray  # noqa
    backwards: np.ndarray


class DualLSTMModel(ABC):
    """
    A wrapper for accessing an individual Keras-defined model, for prediction
    only!
    """

    def predict_file(self, vector: Sequence[Vind]) -> Iterable[TokenResult]:
        """
        Produces prediction results for each token in the file.

        A stream of of (forwards, backwards) tuples, one for each
        cooresponding to a token in the file.

        forwards and backwards are each arrays of floats, having the size of
        the vocabulary. Each element is the probability of that vocabulary
        entry index being in the source file at the particular location.
        """


class KerasDualLSTMModel(DualLSTMModel):
    def __init__(self, *, forwards: 'Model', backwards: 'Model') -> None:
        self.forwards = forwards
        self.backwards = backwards
        assert model_context_length(forwards) == model_context_length(backwards)
        self.context_length = model_context_length(forwards)

    def predict_file(self, vector: Sequence[Vind]) -> Sequence[TokenResult]:
        """
        TODO: Create predict() for entire file as a batch?
        """
        raise NotImplementedError
        """
        x, _ = one_hot_batch([(vector, 0)], batch_size=1,
                             context_length=self.context_length)
        return self.model.predict(x, batch_size=1)[0]
        """

    @classmethod
    def from_directory(cls, dirname: Union[Path, str]) -> 'KerasDualLSTMModel':
        """
        Load the two models from the given directory.
        """
        return cls(forwards=cls.from_filename(Path(dirname) / 'forwards.hdf5'),
                   backwards=cls.from_filename(Path(dirname) / 'backwards.hdf5'))

    @staticmethod
    def from_filename(path: Path) -> 'Model':
        logger = logging.getLogger(__name__)

        from keras.models import load_model
        logger.info('Loading model %s...', path)
        model = load_model(os.fspath(path))
        logger.info('Finished loading model %s:', path)

        return model


def model_context_length(model: 'Model') -> int:
    """
    Return the context-length of a Keras model.

    This should simply be the width of the input layer.
    """
    length: int
    try:
        _, length, _vocab = model.layers[0].batch_input_shape  # type: ignore
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f'Could not determine shape of model')
    else:
        return length


def test():
    """
    I'd write this test if I had a model...
    """
    from sensibility._paths import MODEL_DIR
    model = Model.from_filename(MODEL_DIR / 'javascript-f0.hdf5')
    comma = vocabulary.to_index(',')
    answer = model.predict([comma] * 20)
    assert len(answer) == len(vocabulary)
    assert answer[comma] > 0.5
