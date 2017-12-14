#!/usr/bin/env python3

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
Sensibility --- detect and fix syntax errors in source code.
"""

import pkg_resources

from .edit import Edit, Insertion, Deletion, Substitution
from .language import Language, current_language
from .lexical_analysis import Lexeme, Token, Location, Position
from .source_vector import SourceVector
from .vocabulary import Vocabulary, Vind

# Get the current version from setup.py
__version__ = pkg_resources.get_distribution(__name__).version

# XXX: Deprecated: this alias
language = current_language

__all__ = [
    'Edit', 'Insertion', 'Deletion', 'Substitution',
    'Language', 'language', 'current_language',
    'Lexeme', 'Token', 'Location', 'Position',
    'SourceVector',
    'Vocabulary', 'Vind',
]
