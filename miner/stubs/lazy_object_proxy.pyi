from typing import TypeVar, Callable
T = TypeVar('T')

def Proxy(factory: Callable[[], T]) -> T: ...
