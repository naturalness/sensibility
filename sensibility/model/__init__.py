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

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .lstm.loop_batches import one_hot_batch
from ..vocabulary import Vind


# TODO: extend this for both LSTM and NGram models.

class Model:
    """
    A wrapper for accessing an individual Keras-defined model, for prediction
    only!
    """
    # TODO: correct type for Keras model declared as Any type.
    def __init__(self, model, *,
                 backwards: bool=False,
                 context_length: int) -> None:
        self.model = model
        self.backwards = backwards
        self.context_length = context_length

    @property
    def forwards(self) -> bool:
        return not self.backwards

    def predict(self, vector: Sequence[Vind]) -> np.ndarray:
        """
        TODO: Create predict() for entire file as a batch?
        """
        x, _ = one_hot_batch([(vector, 0)], batch_size=1,
                             context_length=self.context_length)
        return self.model.predict(x, batch_size=1)[0]

    @classmethod
    def from_filename(cls, path: Path,
                      backwards: bool=False) -> 'Model':
        from keras.models import load_model
        print('Loading model:', path, file=sys.stderr)
        model = load_model(str(path))
        print('Finished loading model:', path, file=sys.stderr)

        try:
            length: int
            _, length, _vocab = model.layers[0].batch_input_shape  # type: ignore
        except (IndexError, AttributeError) as e:
            raise RuntimeError(f'Could not determine shape of model: {path!s}')
        else:
            return cls(model, backwards=backwards, context_length=length)


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
