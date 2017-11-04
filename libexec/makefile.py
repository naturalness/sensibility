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

import io
import os
from contextlib import redirect_stdout
from typing import Any, Dict, List, NamedTuple, Set, Union
from pprint import pprint
from pathlib import Path

StrPath = Union[str, os.PathLike]


class Command:
    def __init__(self, bin: str, *args: str, **kwargs) -> None:
        self.bin = bin
        self.arguments: List[str] = []
        self._add_arguments(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> 'Command':
        return self.copy()._add_arguments(*args, **kwargs)

    def _add_arguments(self, *args: str, **kwargs: Any) -> 'Command':
        def generate_kwargs():
            for option, value in kwargs.items():
                if len(option) == 1:
                    yield '-' + option
                else:
                    yield '--' + option.replace('_', '-')
                yield str(value)
        self.arguments += args + tuple(generate_kwargs())
        return self

    def copy(self) -> 'Command':
        return type(self)(self.bin, *self.arguments)

    def __str__(self) -> str:
        return ' '.join([self.bin] + self.arguments)


class Rule:
    def __init__(self, targets: List[StrPath], sources: List[StrPath],
                 recipe: List[Command]) -> None:
        self.targets = targets
        self.sources = sources
        self.recipe = recipe

    @classmethod
    def creates(self, *items: StrPath) -> 'Rule':
        return Rule(targets=list(items), sources=[], recipe=[])

    def set_recipe(self, *commands: Command) -> 'Rule':
        self.recipe.extend(commands)
        return self

    def print(self) -> None:
        print(*self.targets, sep=' ', end=': ')
        print(*self.sources, sep=' ', end='\n')
        for command in self.recipe:
            print('\t', command, sep='', end='\n')
