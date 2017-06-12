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

from .agreement import Agreement
from .corpus import Corpus
from .edit import Edit, Insertion, Deletion, Substitution
from .language import Language, language
from .lexical_analysis import Token, Location, Position
from .sentences import Sentence, forward_sentences, backward_sentences
from .source_file import SourceFile
from .source_vector import SourceVector
from .vectorize_tokens import serialize_tokens
from .vectors import Vectors
from .vocabulary import Vind, LegacyVocabulary, vocabulary

# TODO: Temporary?
from .fix import Sensibility, FixResult

__all__ = [
    'Edit', 'Insertion', 'Deletion', 'Substitution',
    'Language', 'language',
    'SourceFile',  # heh?
    'Vind',
    'Corpus',  # TODO: NO!
    'LegacyVocabulary',  # TODO: NO!
    'serialize_tokens',  # TODO: NO!
    'Vectors',   # TODO: NO!
    'vocabulary',
    'Lexeme', 'Token', 'Location', 'Position',
    'Sentence', 'forward_sentences', 'backward_sentences',  # TODO: maybe?
    'Agreement',
    'Sensibility', 'FixResult'
]
