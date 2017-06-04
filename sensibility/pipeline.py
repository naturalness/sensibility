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
# TODO: Just get rid of this file!
#   A new method in Language.

from abc import ABC, abstractmethod
from typing import Any, AnyStr, Callable, Iterable, Optional, Sequence, Tuple, overload

from .lexical_analysis import Token, Location



class Pipeline(ABC):
    """
    A tokenization pipeline that converts tokens to a format appropriate for
    ingestion into a language model.

    TODO: make these composable, like cons-cells?
    """
