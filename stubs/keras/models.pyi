from typing import Sequence, Iterator, Tuple

from numpy import ndarray

from .optimizers import Optimizer
from . import Loss, Metric
from .layers import Layer
from .callbacks import Callback

Sample = Tuple[ndarray, ndarray]

class Model:
    def compile(self, optimizer: Optimizer, loss: Loss, metrics: Sequence[Metric]=None, loss_weights=None, sample_weight_mode=None, **kwargs) -> None: ...
    def fit_generator(self, generator: Iterator[Sample], steps_per_epoch: int, epochs: int=1, verbose: int=1, callbacks: Sequence[Callback]=None, validation_data: Iterator[Sample]=None, validation_steps=None, class_weight=None, max_queue_size: int=10, workers: int=1, use_multiprocessing: bool=False, shuffle=True, initial_epoch: int=0) -> None: ...

class Sequential(Model):
    def add(self, layer: Layer) -> None: ...
    def summary(self) -> None: ...

def load_model(filename: str) -> Model: ...
