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

import pytest

from sensibility.language import language
from sensibility.edit import Insertion, Deletion, Substitution
from sensibility.vocabulary import Vocabulary, Vind
from sensibility.source_vector import to_source_vector


def setup():
    language.set_language('java')


def test_delete() -> None:
    source_code = to_source_vector(b"""
    class Hello {
        }
    }
    """)
    edit = Deletion(3, to_index('}'))
    mutant = edit.apply(source_code)
    assert b"class identifier { }" == mutant.to_source_code()


def to_index(text: str) -> Vind:
    return language.vocabulary.to_index(text)
