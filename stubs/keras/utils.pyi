from typing import Sized

from numpy import ndarray

class Sequence(Sized):
    def __getitem__(self, idx: int) -> ndarray: ...
    def on_epoch_end(self) -> None: ...
