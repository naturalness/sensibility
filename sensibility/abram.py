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
Utilites designed to maintain Dr. Hindle's sanity.
"""

from typing import TypeVar, Union

Number = TypeVar('Number', int, float, complex)


def at_least(clamp: Number, number: Number) -> Number:
    """
    Return **at least** the clamp_value, else the number.

    >>> at_least(0, 1)
    1
    >>> at_least(0, -100)
    0
    """
    return max(clamp, number)


def at_most(clamp: Number, number: Number) -> Number:
    """
    Return **at most** the clamp value, else the number

    >>> at_most(40, 39.8)
    39.8
    >>> at_most(40, 537)
    40
    """
    return min(clamp, number)
