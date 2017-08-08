from typing import TypeVar, Iterable

T = TypeVar('T')

class Tqdm(Iterable[T]):
    def set_description(self, msg: str) -> None: ...

def tqdm(it: Iterable[T], total: int=None, miniters: int=None) -> Tqdm[T]: ...
