#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pytest
import numpy as np

from sensibility.model.lstm import DualLSTMModel, TokenResult
from sensibility.language import language
from sensibility.edit import Deletion

from sensibility.fix import LSTMFixerUpper


def test_fixer_upper(rigged_lstm_model: DualLSTMModel, i) -> None:
    """
    Test whether the fixer returns reasonable results.
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


def one_hot_result(entry: int, flip: bool=False) -> np.ndarray:
    """
    Turns a vocabulary index into a one-hot probability distributions. i.e.,

                   ⎧  100%, token == entry
        P(token) = ⎨
                   ⎩    0%, otherwise
    """
    vector: np.ndarray[float]
    args = len(language.vocabulary), np.float32
    if flip:
        vector = np.ones(*args)
        vector[entry] = 0.0
    else:
        vector = np.zeros(*args)
        vector[entry] = 1.0
    return vector


# Taken from: https://stackoverflow.com/a/21032099/6626414
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


@pytest.fixture
def rigged_lstm_model(i) -> DualLSTMModel:
    """
    Create a fake model that is rigged to an inverted probability distribution
    for an ellipse token.

    This needs to be a fixure, because it depends on another fixture: c().
    """

    class FakeModel(DualLSTMModel):
        def predict_file(self, tokens):
            def generate_pairs():
                for token in tokens:
                    # Invert the distribution when we get to the rigged token.
                    if token == i('...'):
                        distr = normalized(one_hot_result(token, flip=True))
                    else:
                        distr = one_hot_result(token)
                    # Perfect agreement.
                    yield TokenResult(distr, distr)
            return tuple(generate_pairs())

    return FakeModel()
