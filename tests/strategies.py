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

from hypothesis.strategies import lists, integers, composite  # type: ignore

from sensibility import SourceVector, vocabulary
from sensibility.language import language


@composite
def programs(draw):
    """
    Generate SourceVector instances with random sequences of vectors.

    TODO: rename to source_vectors()? Ew, kinda gross.
    """
    vocabulary = language.vocabulary
    tokens = integers(min_value=vocabulary.minimum_representable_index(),
                      max_value=vocabulary.maximum_representable_index())
    vectors = draw(lists(tokens, min_size=1))
    return SourceVector(vectors)
