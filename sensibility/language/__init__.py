#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2017 Eddie Antonio Santos <easantos@ualberta.ca>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Represents a language and actions you can do to its source code.
"""

import os
import logging
from typing import (
    Any, IO, NamedTuple, Iterable, Sequence, Set, Tuple, Optional, Union
)
from typing import no_type_check, cast, overload
from abc import ABC, abstractmethod

from ..vocabulary import Vocabulary, Vind
from ..lexical_analysis import Token, Location


# Alias for simply an iterable of Tokens.
Tokens = Iterable[Token]
SourceCode = Union[str, bytes, IO[bytes]]


class SourceSummary(NamedTuple):
    sloc: int
    n_tokens: int


class Language(ABC):
    """
    A programming language.
    """

    ############################################################################
    # Public API                                                               #
    ############################################################################

    extensions: Set[str]
    vocabulary: Vocabulary

    @property
    def id(self) -> str:
        return self.name.lower()

    @property
    def name(self) -> str:
        if hasattr(self, 'fullname'):
            return self.fullname  # type: ignore
        else:
            return type(self).__name__

    def matches_extension(self, path: Union[os.PathLike, str]) -> bool:
        """
        Check if the given path matches any of the registered extensions for
        this language.
        """
        filename = os.fspath(path)
        return any(filename.endswith(ext) for ext in self.extensions)

    @overload
    def summarize(self, source: Tokens) -> SourceSummary: ...

    @overload
    def summarize(self, source: SourceCode) -> SourceSummary: ...

    def summarize(self, source: Any) -> SourceSummary:
        """
        Return a SourceSummary, containing the number of tokens,
        and source lines of code. In both cases, the numbers only count tokens
        that are "physically" intended in the file.
        """
        return self.summarize_tokens(self._as_tokens(source))

    def vocabularize(self, source: Union[SourceCode, Tokens]) -> Iterable[str]:
        """
        Produces a stream of normalized types (string representations of
        vocabulary entries) to be insterted into a language model.
        """
        stream = self.vocabularize_tokens(self._as_tokens(source))
        return (tok for _loc, tok in stream)

    def vocabularize_with_locations(self, source: Union[SourceCode, Tokens]
                                    ) -> Iterable[Tuple[Location, str]]:
        """
        As with vocabularize, but also emits locations.
        """
        return self.vocabularize_tokens(self._as_tokens(source))

    def token_locations(self, source: Union[SourceCode, Tokens]) -> Iterable[Location]:
        stream = self.vocabularize_tokens(self._as_tokens(source))
        return (loc for loc, _tok in stream)

    # API that delegates to vocabulary
    def to_index(self, entry: str) -> Vind:
        return self.vocabulary.to_index(entry)

    def to_index_or_unk(self, entry: str) -> Vind:
        return self.vocabulary.to_index_or_unk(entry)

    def to_text(self, entry: Vind) -> str:
        return self.vocabulary.to_text(entry)

    def to_source_text(self, entry: Vind) -> str:
        return self.vocabulary.to_source_text(entry)

    # Dunder methods

    def __str__(self) -> str:
        return self.id

    # API implemented by inheritors.

    @abstractmethod
    def tokenize(self, source: Union[str, bytes, IO[bytes]]) -> Iterable[Token]: ...

    @abstractmethod
    def check_syntax(self, source: Union[str, bytes]) -> bool: ...

    @abstractmethod
    def summarize_tokens(self, tokens: Iterable[Token]) -> SourceSummary: ...

    @abstractmethod
    def vocabularize_tokens(self, source: Iterable[Token]) -> Iterable[Tuple[Location, str]]:
        """
        Given tokens, this should produce a stream of normalized types (string
        representations of vocabulary entries) to be insterted into a language
        model, attached with their location in the original source.
        """

    def _as_tokens(self, source: Union[SourceCode, Tokens]) -> Tokens:
        """
        Ensures that anything that goes is returned as tokens.
        """
        if isinstance(source, (str, bytes, IO)):
            return self.tokenize(source)
        else:
            return source


class LanguageNotSpecifiedError(Exception):
    """
    Raised when Sensibility could not infer the programming language that
    should be used.

    You can explicitly define the environment variable SENSIBILITY_LANGUAGE=
    as a fallback, however, all other methods of inferring the language are
    preferred to this (such as inferring the language from filename
    extensions, or from command line arguments).
    """
    pass


class LanguageProxy(Language):
    """
    Defines a uniform interface to access the currently active language.
    """

    def __init__(self) -> None:
        self._language: Optional[Language] = None

    @property
    def is_initialized(self) -> bool:
        """
        Is a language currently defined?
        """
        return self._language is not None

    @property
    def wrapped_language(self) -> Language:
        if self._language is None:
            self._language = self.determine_language()
        return self._language

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._language!r})"

    def forget(self) -> None:
        """
        Unsets the active language.
        """
        self._language = None

    def set(self, name: str) -> 'LanguageProxy':
        """
        Less redundant alias of set_language().
        """
        return self.set_language(name)

    def set_language(self, name: str) -> 'LanguageProxy':
        """
        Explicitly set the language to load.
        """
        self._language = self.load_langauge_by_name(name)
        return self

    def unwrap(self) -> Language:
        """
        Returns the underlying Language object.
        """
        if self._language is not None:
            return self._language
        raise LanguageNotSpecifiedError

    def determine_language(self) -> Language:
        logger = logging.getLogger(self.__class__.__name__)

        # TODO: set language from filename?
        # TODO: set language from database? -- but don't put this in this file.

        # Try loading from the environment LAST.
        name_from_env = os.getenv('SENSIBILITY_LANGUAGE')
        if name_from_env is not None:
            logger.info('Language inferred from environment: %s',
                        name_from_env)
            return self.load_langauge_by_name(name_from_env)

        raise LanguageNotSpecifiedError

    def load_langauge_by_name(self, name: str):
        """
        Loads a Language dynamically by looking for the appropriate module,
        and returning the attribute with the same name.
        """
        from importlib import import_module
        name = name.lower()
        module = import_module(f".{name}", package='sensibility.language')
        if not hasattr(module, name):
            raise ImportError(f'could not find {name} in module {module}')
        return getattr(module, name)

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

    def vocabularize_tokens(self, *args, **kwargs):
        return self.wrapped_language.vocabularize_tokens(*args, **kwargs)

    def to_index(self, *args, **kwargs):
        return self.wrapped_language.to_index(*args, **kwargs)

    def to_index_or_unk(self, *args, **kwargs):
        return self.wrapped_language.to_index_or_unk(*args, **kwargs)

    def to_text(self, *args, **kwargs):
        return self.wrapped_language.to_text(*args, **kwargs)

    def to_source_text(self, *args, **kwargs) -> str:
        return self.wrapped_language.to_source_text(*args, **kwargs)


class ConcreteLanguageProxy(LanguageProxy):
    """
    __getattr__() is defined here so that mypy doesn't think the LanguageProxy
    interface can respond to ANY method; however, __getattr__() does the
    delegation to underlying active language object.
    """
    def __getattr__(self, name: str) -> Any:
        """
        Delegate to wrapped_language.
        """
        # Workaround for nosy inspect.unwrap():
        if name == '__wrapped__':
            raise AttributeError

        return getattr(self.wrapped_language, name)


current_language: LanguageProxy = ConcreteLanguageProxy()
# XXX: Deprecated! Use current_language instead.
language = current_language
