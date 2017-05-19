#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
from typing import Set, Sequence, Union
from abc import ABC, abstractmethod
from lazy_object_proxy import Proxy

from ..token_utils import Token


class Language(ABC):
    """
    A programming language.
    """

    extensions: Set[str]

    @property
    def id(self) -> str:
        return self.name.lower()

    @property
    def name(self) -> str:
        if hasattr(self, 'fullname'):
            return self.fullname  # type: ignore
        else:
            return type(self).__name__

    def __str__(self) -> str:
        return self.id

    def matches_extension(self, path: Union[os.PathLike, str]) -> bool:
        filename = os.fspath(path)
        return any(filename.endswith(ext) for ext in self.extensions)

    @abstractmethod
    def tokenize(self, source: str) -> Sequence[Token]: ...

    @abstractmethod
    def check_syntax(self, source: str) -> bool: ...



class JavaScript(Language):
    extensions = {'.js'}

from .python import Python
language = Python
