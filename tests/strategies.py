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

from hypothesis.strategies import lists, integers, composite

from sensibility import TokenSequence, vocabulary


@composite
def programs(draw):
    """
    Generate TokenSequence instances with random token sequences.

    TODO: rename to token_sequences()? Ew, kinda gross.
    """
    tokens = integers(min_value=vocabulary.start_token_index + 1,
                      max_value=vocabulary.end_token_index - 1)
    vectors = draw(lists(tokens, min_size=1))
    return TokenSequence(vectors)
