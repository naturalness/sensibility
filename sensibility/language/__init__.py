#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Represents a language and actions you can do to its source code.
"""

import os
from typing import IO, Sequence, Set, Union
from abc import ABC, abstractmethod

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
    def tokenize(self, source: Union[str, bytes, IO[bytes]]) -> Sequence[Token]: ...

    @abstractmethod
    def check_syntax(self, source: Union[str, bytes]) -> bool: ...

    # TODO: vocabulary?


# TODO: move this to its own file.
class JavaScript(Language):
    extensions = {'.js'}


from .python import python
language: Language = python
