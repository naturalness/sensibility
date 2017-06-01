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
Language definition for JavaScript.
"""

from typing import IO, Sequence, Union

from ..token_utils import Token
from . import Language


class JavaScript(Language):
    extensions = {'.js'}

    def tokenize(self, source: Union[str, bytes, IO[bytes]]) -> Sequence[Token]:
        # TODO: use Esprima
        raise NotImplementedError

    def check_syntax(self, source: Union[str, bytes]) -> bool:
        # TODO: use Esprima
        raise NotImplementedError

    def summarize_tokens(self, *args):
        # TODO: use Esprima
        raise NotImplementedError

javascript = JavaScript()
