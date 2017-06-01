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
Miscellaneous utilities.
"""


from typing import Type, TypeVar


C = TypeVar('C')


def singleton(cls: Type[C]) -> C:
    """
    Given a class, instantiates it. To be used as a decorator to create
    psuedo-singletons.
    """
    return cls()


def test_singleton() -> None:
    class Foo:
        baz = 'quux'

    @singleton
    class bar(Foo):
        baz = 'fizz'

    assert isinstance(bar, Foo)
    assert bar.baz == 'fizz'
