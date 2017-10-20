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

from edit_distance import SequenceMatcher  # type: ignore
from Levenshtein import distance, editops  # type: ignore

from sensibility.lexical_analysis import Token
from sensibility.language import language
from sensibility.vocabulary import Vind
from sensibility import Edit
from sensibility.abram import at_least


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
    """
    def __init__(self, a: Iterable[Token], b: Iterable[Token],
                 convert: TokenConverter) -> None:
        self.src_toks = tuple(a)
        self.dest_toks = tuple(b)
        # Pre-compute the converted token stream.
        self.src_text = tuple(convert(tok) for tok in self.src_toks)
        self.dest_text = tuple(convert(tok) for tok in self.dest_toks)
        self._mapper = PrivateUseAreaMapper()

    def distance(self) -> int:
        """
        Levenshtein edit distance of the two sequences.
        """
        mapper = self._mapper
        seq_a = ''.join(mapper[token] for token in self.src_text)
        seq_b = ''.join(mapper[token] for token in self.dest_text)
        return distance(seq_a, seq_b)

    def determine_fix(self) -> FixEvent:
        """
        Edit operations between the two sequences.
        """
        src_toks = self.src_toks
        dest_toks = self.dest_toks

        src = self.src_text
        dest = self.dest_text

        ops = differences_only(SequenceMatcher(a=src, b=dest).get_opcodes())
        # This only works for files with one edit!
        assert len(ops) == 1

        new: Optional[Vind]
        old: Optional[Vind]

        def to_index(token):
            return language.vocabulary.to_index_or_unk(token.name)

        # Decode editop's format into our "Database-friendly" format.
        (type_name, src_pos, src_end, dest_pos, dest_end), = ops
        assert src_end in (src_pos, src_pos + 1)
        assert dest_end in (dest_pos, dest_pos + 1)
        if type_name == 'insert':
            code, new, old = 'i', to_index(dest_toks[dest_pos]), None
            original_token_index = src_pos
            edit_index = dest_pos
        elif type_name == 'delete':
            code, new, old = 'x', None, to_index(src_toks[src_pos])
            original_token_index = edit_index = src_pos
        elif type_name == 'replace':
            code, new, old = 's', to_index(dest_toks[dest_pos]), to_index(src_toks[src_pos])
            original_token_index = edit_index = src_pos
        else:
            raise ValueError(f'Cannot handle operation: {ops}')

        # Note: in the case of insertions, src_pos will be the index to insert
        # BEFORE, which is exactly what Edit.deserialize wants; src_position also
        # works for deletions, and substitutions.
        edit = Edit.deserialize(code, edit_index, new, old)
        error_token = src_toks[original_token_index]

        return FixEvent(edit, error_token.line)

    @classmethod
    def of(cls, file_a: bytes, file_b: bytes, abstract: bool) -> 'TokenDistance':
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
            return self._map.setdefault(string, self._get_next_code_point())

    def _get_next_code_point(self) -> str:
        code_point = self._next_code_point
        self._next_code_point += 1
        if self._next_code_point > self.MAXIMUM:
            raise OverflowError('Ran out of code points!')
        return chr(code_point)


def tokenwise_distance(file_a: bytes, file_b: bytes, abstract_open_classes=True) -> int:
    """
    Calculates the token-wise Levenshtein distance between two source files.
    """
    return TokenDistance.of(file_a, file_b, abstract_open_classes).distance()


def determine_edit(file_a: bytes, file_b: bytes, abstract_open_classes=True) -> Edit:
    return determine_fix_event(file_a, file_b, abstract_open_classes).fix


def differences_only(ops) -> Sequence[EditOp]:
    return tuple(op for op in ops if op[0] != 'equal')


def determine_fix_event(file_a: bytes, file_b: bytes, abstract_open_classes=True) -> FixEvent:
    """
    For two source files with Levenshtein distance of one, this returns the
    edit that converts the first file into the second file.
    """
    return TokenDistance.of(file_a, file_b, abstract_open_classes).determine_fix()


def to_abstraced_tokens(source: bytes) -> Sequence[str]:
    """
    Turns the source code to a sequence of vocabulary tokens.
    """
    return tuple(language.vocabularize(source))


def to_value_stream(source: bytes) -> Sequence[str]:
    """
    Turns the source code to sequence of tokens values.
    """
    return tuple(token.value for token in language.tokenize(source))


def lists(it):
    a_list = []
    b_list = []
    for a, b in it:
        a_list.append(a)
        b_list.append(b)
    return a_list, b_list
