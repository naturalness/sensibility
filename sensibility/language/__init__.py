#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Represents a language and actions you can do to its source code.
"""

import os
from typing import Any, IO, Sequence, Set, Union, NamedTuple, cast, overload
from abc import ABC, abstractmethod

from ..token_utils import Token



class SourceSummary(NamedTuple):
    sloc: int
    n_tokens: int


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

    @abstractmethod
    def tokenize(self, source: Union[str, bytes, IO[bytes]]) -> Sequence[Token]: ...

    @abstractmethod
    def check_syntax(self, source: Union[str, bytes]) -> bool: ...

    @abstractmethod
    def summarize_tokens(self, tokens: Sequence[Token]) -> SourceSummary: ...

    def matches_extension(self, path: Union[os.PathLike, str]) -> bool:
        """
        Check if the given path matches any of the registered extensions for
        this language.
        """
        filename = os.fspath(path)
        return any(filename.endswith(ext) for ext in self.extensions)

    @overload
    def summarize(self, source: Sequence[Token]) -> SourceSummary: ...
    @overload
    def summarize(self, source: Union[str, bytes, IO[bytes]]) -> SourceSummary: ...

    def summarize(self, source: Any) -> SourceSummary:
        if isinstance(source, (str, bytes, IO)):
            tokens = self.tokenize(source)
        else:
            tokens = cast(Sequence[Token], source)
        return self.summarize_tokens(tokens)

    # TODO: vocabulary?


# TODO: crazy proxy object
class LanguageProxy(Language):
    @property
    def wrapped(self) -> Language:
        from .python import python
        return python

    def tokenize(self, *args):
        return self.wrapped.tokenize(*args)

    def check_syntax(self, *args):
        return self.wrapped.check_syntax(*args)

    def summarize_tokens(self, *args):
        return self.wrapped.summarize_tokens(*args)

language: Language = LanguageProxy()
