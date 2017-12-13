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
Determines the Levenshtein distance between two source files.
"""

from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, cast

from Levenshtein import editops  # type: ignore

from sensibility import Edit
from sensibility.abram import at_least
from sensibility.language import language
from sensibility.lexical_analysis import Token
from sensibility.vocabulary import Vind

TokenConverter = Callable[[Token], str]
EditOp = Tuple[str, int, int, int, int]


class FixEvent:
    def __init__(self, fix: Edit, line_no: int) -> None:
        # Feature request: use the current language for each fix, then to
        # decode the token at instantiation time.
        self.fix = fix
        self.line_no = line_no

    @property
    def mistake(self) -> Edit:
        return -self.fix

    @property
    def old_token(self) -> Optional[str]:
        _code, _pos, _new, old = self.fix.serialize()
        # assert new == f_old and old == f_new
        return language.vocabulary.to_text(old) if old is not None else None

    @property
    def new_token(self) -> Optional[str]:
        # assert new == f_old and old == f_new
        _code, _pos, new, _old = self.fix.serialize()
        return language.vocabulary.to_text(new) if new is not None else None


class TokenDistance:
    """
    Determines the distance between two sequences of tokens.

    Get the Levenshtein distance by calling .distance().
    Get the single-token fix by calling .determine_fix().
    """
    def __init__(self, a: Iterable[Token], b: Iterable[Token],
                 convert: TokenConverter) -> None:
        self.src_toks = tuple(a)
        self.dest_toks = tuple(b)

        # Convert each token to an appropriate stringified representation.
        self.src_text = tuple(convert(tok) for tok in self.src_toks)
        self.dest_text = tuple(convert(tok) for tok in self.dest_toks)

        # Because python-Levenshtein calculates string distances exclusively,
        # synthesize "strings" by mapping each of the strings in the token
        # sequence to a single character.
        mapper = PrivateUseAreaMapper()
        src_str = ''.join(mapper[token] for token in self.src_text)
        dest_str = ''.join(mapper[token] for token in self.dest_text)

        # Determine the Levenstein edit operations.
        self._edit_ops = editops(src_str, dest_str)

    def distance(self) -> int:
        """
        Levenshtein edit distance of the two sequences.
        """
        return len(self._edit_ops)

    def determine_fix(self) -> FixEvent:
        """
        Edit operations between the two sequences.
        """
        ops = self._edit_ops
        # This only works for files with one edit!
        assert len(ops) == 1, f'{self.distance()} differences: cannot determine single fix'

        new: Optional[Vind]
        old: Optional[Vind]
        src = self.src_toks
        dest = self.dest_toks

        def to_index(token: Token) -> Vind:
            return language.vocabulary.to_index_or_unk(token.name)

        # Decode editop's format into our "database-friendly" format.
        (type_name, src_pos, dest_pos), = ops
        if type_name == 'insert':
            code, new, old = 'i', to_index(dest[dest_pos]), None
        elif type_name == 'delete':
            code, new, old = 'x', None, to_index(src[src_pos])
        elif type_name == 'replace':
            code, new, old = 's', to_index(dest[dest_pos]), to_index(src[src_pos])
        else:
            raise ValueError(f'Cannot handle operation: {ops}')

        # Note: in the case of insertions, src_pos will be the index to insert
        # BEFORE, which is exactly what Edit.deserialize wants; src_position also
        # works for deletions, and substitutions.
        edit = Edit.deserialize(code, src_pos, new, old)
        error_token = src[at_least(0, src_pos - 1) if code == 'i' else src_pos]
        return FixEvent(edit, error_token.line)

    @classmethod
    def of(cls, file_a: bytes, file_b: bytes, abstract: bool=False) -> 'TokenDistance':
        """
        Return a TokenDistance configured for the given files.
        """
        from operator import attrgetter
        assert language.name == 'Java'
        return cls(language.tokenize(file_a),
                   language.tokenize(file_b),
                   convert=attrgetter('name') if abstract else attrgetter('value'))


class PrivateUseAreaMapper:
    """
    Maps arbitrary strings into private use unicode code points.
    This allows one to encode arbitrary sequences for use with
    CERTAIN edit distance libraries that operate on plain strings exclusively.
    Why private use code points? So that strings produced by this operation
    are not as easily confused with real data.

    >>> mapper = PrivateUseAreaMapper()
    >>> sentence = 'A rose is a rose is a rose'.split()
    >>> mappings = [mapper[word] for word in sentence]
    >>> len(mappings)
    8
    >>> len(set(mappings))
    4
    """

    # Supplementary Private Use Area B
    PUA_B_START = 0x100000
    # This is the largest possible Unicode code point.
    MAXIMUM = 0x10FFFF

    def __init__(self) -> None:
        self._next_code_point = self.PUA_B_START
        self._map: Dict[str, str] = {}

    def __getitem__(self, string: str) -> str:
        try:
            return self._map[string]
        except KeyError:
            # Store the next free code point in the dictionary,
            # and return it.
            return self._map.setdefault(string, self._get_next_code_point())

    def _get_next_code_point(self) -> str:
        code_point = self._next_code_point
        self._next_code_point += 1
        if self._next_code_point > self.MAXIMUM:
            raise OverflowError('Ran out of code points!')
        return chr(code_point)


# ################################ Old API ################################ #

def tokenwise_distance(file_a: bytes, file_b: bytes, abstract_open_classes=True) -> int:
    """
    Calculates the token-wise Levenshtein distance between two source files.
    """
    return TokenDistance.of(file_a, file_b, abstract_open_classes).distance()


def determine_edit(file_a: bytes, file_b: bytes, abstract_open_classes=True) -> Edit:
    """
    Determine the single edit made in file_b that will fix file_a.
    """
    return determine_fix_event(file_a, file_b, abstract_open_classes).fix


def determine_fix_event(file_a: bytes, file_b: bytes, abstract_open_classes=True) -> FixEvent:
    """
    For two source files with Levenshtein distance of one, this returns the
    edit that converts the first file into the second file.
    """
    return TokenDistance.of(file_a, file_b, abstract_open_classes).determine_fix()
