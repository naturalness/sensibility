from typing import Tuple

from . import ActivationFunction, Initializer, Regularizer

class Layer:
    def __init__(self, input_shape: Tuple[int, ...]=None) -> None: ...

class Dense(Layer):
    def __init__(self, *args, **kwargs) -> None: ...

class Activation(Layer):
    def __init__(self, fn: ActivationFunction) -> None: ...

class Recurrent(Layer):
    def __init__(self, return_sequences: bool=False) -> None: ...

class LSTM(Layer):
    def __init__(self, units: int, activation: ActivationFunction='tanh', recurrent_activation: ActivationFunction='hard_sigmoid', use_bias: bool=True, kernel_initializer: Initializer='glorot_uniform', recurrent_initializer: Initializer='orthogonal', bias_initializer: Initializer='zeros', unit_forget_bias: bool=True, kernel_regularizer: Regularizer=None, recurrent_regularizer: Regularizer=None, bias_regularizer: Regularizer=None, activity_regularizer: Regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout: float=0.0, recurrent_dropout=0.0, **kwargs) -> None: ...
