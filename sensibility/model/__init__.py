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


from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..loop_batches import one_hot_batch
from ..vocabulary import Vind


# TODO: extend this for both LSTM and NGram models.

class Model:
    """
    A wrapper for accessing an individual Keras-defined model, for prediction
    only!
    """
    def __init__(self, model: Any, *,
                 backwards: bool=False,
                 context_length: int=20) -> None:
        # XXX: Keras model declared as Any type.
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
    def from_filename(cls,
                      path: Path,
                      backwards: bool=False,
                      **kwargs) -> 'Model':
        from keras.models import load_model
        model = load_model(str(path))

        return cls(model, backwards=backwards, **kwargs)


def test():
    """
    I'd write this test if I had a model...
    """
    from ._paths import MODEL_DIR
    model = Model.from_filename(MODEL_DIR / 'javascript-f0.hdf5')
    comma = vocabulary.to_index(',')
    answer = model.predict([comma] * 20)
    assert len(answer) == len(vocabulary)
    assert answer[comma] > 0.5
