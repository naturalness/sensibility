#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import hashlib
from itertools import product
from functools import reduce
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, List, Mapping, Set, Sized
from pathlib import PurePath


class Configuration(Mapping[str, Any]):
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(**kwargs)

    @property
    def slug(self) -> PurePath:
        items = sorted(self.__dict__.items())
        return PurePath(','.join(f"{key}-{val}" for key, val in items))

    @property
    def hashed_slug(self) -> PurePath:
        binary = str(self.slug).encode('UTF-8')
        return PurePath(hashlib.sha256(binary).hexdigest())

    def __getitem__(self, key: str) -> Any:
        "For use like a dictionary."
        return self.__dict__[key]

    def __getattr__(self, key: str) -> Any:
        "For use like a namespace."
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)


class Configurations(Sized, Iterable[Configuration]):
    def __init__(self, **configs: Set[Any]) -> None:
        self.options = configs

    def __iter__(self) -> Iterator[Configuration]:
        """
        Generates the cartesian product of all configurations.
        """
        names = list(self.options.keys())
        for prod in product(*(self.options[opt] for opt in names)):
            yield Configuration(**dict(zip(names, prod)))

    def __len__(self) -> int:
        """
        Return the number of possible configurations.
        """
        return reduce(lambda acc, val: acc * len(val), self.options.values(), 1)

    def __radd__(self, other: List[Configuration]) -> List[Configuration]:
        return other + list(self)

    def __add__(self, other: Iterable[Configuration]) -> List[Configuration]:
        return list(self) + list(other)
