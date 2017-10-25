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


import pytest
import numpy as np

from sensibility.model.lstm import DualLSTMModel, TokenResult
from sensibility.language import current_language
from sensibility.edit import Deletion

from sensibility.fix import LSTMFixerUpper


def setup():
    current_language.set('java')


def test_fixer_upper(rigged_lstm_model: DualLSTMModel, i) -> None:
    """
    Test whether the fixer returns reasonable results.
    The fake LSTM is rigged such that it claims everywhere in the file looks
    100% as it should be, and only the area where the syntax error is looks
    suspicious.
    """

    broken_source = b'''
        package ca.ualberta.cs;

        class HelloWorld {
            public static void main(String args[] /* Syntax error, delete token[19] to fix */ ... ) {
                System.out.println("Hello, World!");
            }
        }
    '''

    fixer = LSTMFixerUpper(rigged_lstm_model)
    results = fixer.fix(broken_source)
    assert len(results) >= 1
    top_result = results[0]
    if isinstance(top_result, Deletion):
        assert top_result.index == 19
        assert top_result.original_token == i('...')
    else:
        pytest.fail(f"Did not get deletion; instead got: {top_result}")


def test_fixer_upper_oov(rigged_lstm_model):
    """
    Regression: fixer upper should work, even in the presence of OoV tokens.
    """

    broken_source = b'''
        package ca.ualberta.cs;

        class HelloWorld {
            public static void main(String args[] /* Syntax error, delete token[19] to fix */ # ) {
                System.out.println("Hello, World!");
            }
        }
    '''

    # Set up the rigged model to report error tokens as unks.
    UNK = current_language.vocabulary.unk_token_index
    rigged_lstm_model.bad_token = UNK
    fixer = LSTMFixerUpper(rigged_lstm_model)

    # This is is where it used to crash:
    results = fixer.fix(broken_source)
    assert len(results) >= 1
    top_result = results[0]

    if isinstance(top_result, Deletion):
        assert top_result.index == 19
        assert top_result.original_token == UNK
    else:
        pytest.fail(f"Did not get deletion; instead got: {top_result}")


def one_hot_result(entry: int, flip: bool=False) -> np.ndarray:
    """
    Turns a vocabulary index into a one-hot probability distributions. i.e.,

                   ⎧  100%, token == entry
        P(token) = ⎨
                   ⎩    0%, otherwise
    """
    vector: np.ndarray[float]
    args = (len(current_language.vocabulary),), np.float32
    if flip:
        vector = np.ones(*args)
        vector[entry] = 0.0
    else:
        vector = np.zeros(*args)
        vector[entry] = 1.0
    return vector


def normalize(a: np.ndarray) -> np.ndarray:
    """
    Given a "one-cold" vector (one-of-k, where it's all 1), returns a vector
    with sum == 1.
    """
    assert len(a) >= 2
    factor = len(a) - 1.
    return a / factor


@pytest.fixture
def rigged_lstm_model(i) -> DualLSTMModel:
    """
    Create a fake model that is rigged to an inverted probability distribution
    for an ellipse token.

    This needs to be a fixure, because it depends on another fixture: c().
    """

    class FakeModel(DualLSTMModel):
        @property
        def bad_token(self):
            if not hasattr(self, '_bad_token'):
                self._bad_token = i('...')
            return self._bad_token

        @bad_token.setter
        def bad_token(self, value):
            self._bad_token = value

        def predict_file(self, tokens):
            def generate_pairs():
                for token in tokens:
                    # Invert the distribution when we get to the rigged token.
                    if token == self.bad_token:
                        distr = normalize(one_hot_result(token, flip=True))
                    else:
                        distr = one_hot_result(token)
                    assert distr.shape == (len(current_language.vocabulary),)
                    assert distr.sum() == pytest.approx(1.0)
                    # Perfect agreement.
                    yield TokenResult(distr, distr)
            return tuple(generate_pairs())

    return FakeModel()
