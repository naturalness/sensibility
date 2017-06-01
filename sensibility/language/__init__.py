#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Represents a language and actions you can do to its source code.
"""

import os
from typing import Any, IO, NamedTuple, Sequence, Set, Union, cast, overload
from abc import ABC, abstractmethod

from ..token_utils import Token



class SourceSummary(NamedTuple):
    sloc: int
    n_tokens: int


class Language(ABC):
    """
    A programming language.
    """

    #@property
    #@abstractmethod
    #def extensions(self) -> Set[str]: ...
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


class LanguageProxy(Language):
    """
    Provides uniform access to the language
    """

    def __init__(self) -> None:
        self._language: Language = None

    @property
    def is_initialized(self) -> bool:
        return self._language is not None

    @property
    def wrapped_language(self) -> Language:
        # TODO: smart things here. With logging!
        if self._language is None:
            from .python import python
            self._language = python
        return self._language

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._language!r})"

    def __getattr__(self, name: str) -> Any:
        """
        Delegate to wrapped_language.
        """
        # Workaround for nosy inspect.unwrap():
        if name == '__wrapped__':
            raise AttributeError

        return getattr(self.wrapped_language, name)

    # Wrapped methods and properties

    @property
    def name(self) -> str:
        return self.wrapped_language.name

    def tokenize(self, *args):
        return self.wrapped_language.tokenize(*args)

    def check_syntax(self, *args):
        return self.wrapped_language.check_syntax(*args)

    def summarize_tokens(self, *args):
        return self.wrapped_language.summarize_tokens(*args)

language = LanguageProxy()
