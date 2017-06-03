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
from typing import Any, IO, NamedTuple, Sequence, Set, Union
from typing import no_type_check, cast, overload
from abc import ABC, abstractmethod

from ..lexical_analysis import Token
from ..pipeline import Pipeline


class SourceSummary(NamedTuple):
    sloc: int
    n_tokens: int


class Language(ABC):
    """
    A programming language.
    """

    extensions: Set[str]
    pipeline: Pipeline

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
    Provides uniform access to the language
    """

    def __init__(self) -> None:
        self._language: Language = None

    @property
    def is_initialized(self) -> bool:
        return self._language is not None

    @property
    def wrapped_language(self) -> Language:
        if self._language is None:
            self._language = self.determine_language()
        return self._language

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._language!r})"

    @no_type_check
    def __getattr__(self, name: str) -> Any:
        """
        Delegate to wrapped_language.
        """
        # Workaround for nosy inspect.unwrap():
        if name == '__wrapped__':
            raise AttributeError

        return getattr(self.wrapped_language, name)

    def set_language(self, name: str) -> 'LanguageProxy':
        """
        Explicitly set the language to load.
        """
        self._language = self.load_langauge_by_name(name)
        return self

    def determine_language(self) -> Language:
        logger = logging.getLogger(self.__class__.__name__)

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

    # TODO: set language from filename?
    # TODO: set language from database? -- but don't put this in this file.

# TODO: Use even MORE redirection to expose LanguageProxy interface,
#       but let __getattr__ exist in a different class.
language: Language = LanguageProxy()
