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
Tokenization Pipeline
"""

from abc import ABC, abstractmethod
from typing import Any, AnyStr, Callable, Iterable, Optional, Sequence, Tuple, overload

from .lexical_analysis import Token, Location

PipelineStage = Callable[[Any], Optional[Any]]


class Pipeline(ABC):
    """
    A tokenization pipeline that converts tokens to a format appropriate for
    ingestion into a language model.

    TODO: make these composable, like cons-cells?
    """

    @property
    @abstractmethod
    def stages(self) -> Sequence[PipelineStage]: ...

    @overload
    def execute(self, tokens: AnyStr) -> Iterable[Any]: ...
    @overload
    def execute(self, tokens: Sequence[Token]) -> Iterable[Any]: ...

    def execute(self, source):
        """
        Executes all stages of the pipeline, yielding elements in a format
        specified by the pipeline.
        """
        for _, element in self.execute_with_locations(source):
            yield element

    def run_pipeline(self, element: Any) -> Optional[Any]:
        intermediate: Any = element
        for stage in self.stages:
            intermediate = stage(intermediate)
            if intermediate is None:
                return None
        return intermediate

    @overload
    def execute_with_locations(self, tokens: AnyStr) -> Iterable[Tuple[Location, Any]]: ...
    @overload
    def execute_with_locations(self, tokens: Sequence[Token]) -> Iterable[Tuple[Location, Any]]: ...

    def execute_with_locations(self, source):
        """
        Same as #execute(), but returns pairs of (Location, token) pairs,
        where `token` is returned by the pipeline.
        """
        # Ensure we START with a token stream.
        if isinstance(source, (bytes, str)):
            tokens = self.tokenize(source)
        else:
            tokens = source

        # Yield the elements.
        for token in tokens:
            location = token.location
            element = self.run_pipeline(token)
            if element is not None:
                yield location, element

    @abstractmethod
    def tokenize(self, source: AnyStr) -> Sequence[Token]: ...
