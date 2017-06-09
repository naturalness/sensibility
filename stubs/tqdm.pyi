from typing import TypeVar, Iterable

T = TypeVar('T')
def tqdm(it: Iterable[T]) -> Iterable[T]: ...
