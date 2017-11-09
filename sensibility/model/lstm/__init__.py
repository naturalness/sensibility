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

from sensibility.vocabulary import Vind
from sensibility.language import language
from sensibility.sentences import forward_sentences, backward_sentences
from sensibility.sentences import Sentence

import numpy as np

# Keras is poorly-behaved and does things when you import it,
# so only import it statically (during type-checking).
if TYPE_CHECKING:
    from keras.models import Model


class TokenResult(NamedTuple):
    forwards: np.ndarray
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
        self.logger = logging.getLogger(type(self).__name__)
        self.forwards = forwards
        self.backwards = backwards
        assert model_context_length(forwards) == model_context_length(backwards)
        self.context_length = model_context_length(forwards)
        self.one_hot = OneHotter(context_length=self.context_length,
                                 vocabulary_size=len(language.vocabulary))
        self.logger.info('Loaded models with context length %d (window size %d)',
                         self.context_length, self.context_length + 1)

    def predict_file(self, vector: Sequence[Vind]) -> Sequence[TokenResult]:
        """
        TODO: Create predict() for entire file as a batch?
        """
        fw = self.one_hot.forwards(vector)
        bw = self.one_hot.backwards(vector)
        fw_predictions = self.forwards.predict(fw)
        bw_predictions = self.backwards.predict(bw)

        assert len(vector) == len(fw_predictions) == len(bw_predictions)

        def generate_pairs():
            for index in range(len(vector)):
                yield TokenResult(fw_predictions[index],
                                  bw_predictions[index])
        return tuple(generate_pairs())

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


class OneHotter:
    def __init__(self, *, context_length: int, vocabulary_size: int) -> None:
        self.context_length = context_length
        self.vocabulary_size = vocabulary_size

    def forwards(self, vector: Sequence[Vind]) -> np.ndarray:
        return self._one_hot(vector, forward_sentences)

    def backwards(self, vector: Sequence[Vind]) -> np.ndarray:
        return self._one_hot(vector, backward_sentences)

    def _one_hot(self, vector: Sequence[Vind], sentenizer) -> np.ndarray:
        """
        Create a 3D matrix, the size of the vector on the largest axis.
        Each "slice" of the matrix is a sentence from the vector, one-hot
        encoded.
        """
        dim = (len(vector), self.context_length, self.vocabulary_size)
        xs: np.ndarray[bool] = np.zeros(dim, dtype=np.bool)
        sentences: Iterable[Sentence] = sentenizer(vector, context=self.context_length)

        # Fill in the matrix, sentence-by-sentence.
        for index, (sentence, _adjacent_token) in enumerate(sentences):
            for pos, vocab_id in enumerate(sentence):
                xs[index, pos, vocab_id] = True

        return xs


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


def test(dirname: Path=None) -> None:
    from sensibility._paths import REPOSITORY_ROOT
    from sensibility.source_vector import to_source_vector
    if dirname is None:
        dirname = REPOSITORY_ROOT / 'tests'

    language.set('java')
    model = KerasDualLSTMModel.from_directory(dirname)
    source = to_source_vector(rb'''
        package ca.ualberta.cs;

        class HelloWorld {
            public static void main(String args[] /* Syntax error, delete token[19] to fix */ ... ) {
                System.out.println("Hello, World!");
            }
        }
    ''')

    answer = model.predict_file(source)
    assert len(answer) == len(source)
    text = language.vocabulary.to_source_text
    for expected, predictions in zip(source, answer):
        actual_fw = text(predictions.forwards.argmax())  # type: ignore
        actual_bw = text(predictions.backwards.argmax())  # type: ignore
        print(f"{actual_fw:>14}\t{actual_bw:>14}\t{text(expected)}")
