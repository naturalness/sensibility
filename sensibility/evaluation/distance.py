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

from typing import Iterable, Optional, cast

from Levenshtein import distance, editops  # type: ignore

from sensibility.language import language
from sensibility.vocabulary import Vind
from sensibility import Edit
from sensibility.abram import at_least


# Supplementary Private Use Area B
PUA_B_START = 0x100000


def encode(entries: Iterable[str]) -> str:
    """
    Converts a sequence of tokens into an encoded string, that preserves
    distinctness between tokens. This string can be operated by the
    Levenshtein module.
    """
    # Injective maps of each vocabulary entry to a code point in the private
    # use area (PUA).
    # distance() works on code points, so this effectively makes distance()
    # work on token classes.
    to_index = language.vocabulary.to_index_or_unk
    return ''.join(chr(PUA_B_START + to_index(entry)) for entry in entries)


def tokenwise_distance(file_a: bytes, file_b: bytes) -> int:
    """
    Calculates the token-wise Levenshtein distance between two source files.
    """
    seq_a = encode(language.vocabularize(file_a))
    seq_b = encode(language.vocabularize(file_b))
    return distance(seq_a, seq_b)


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


def determine_edit(file_a: bytes, file_b: bytes) -> Edit:
    return determine_fix_event(file_a, file_b).fix


def determine_fix_event(file_a: bytes, file_b: bytes) -> FixEvent:
    """
    For two source files with Levenshtein distance of one, this returns the
    edit that converts the first file into the second file.
    """
    src_locs, src = lists(language.vocabularize_with_locations(file_a))
    dest = list(language.vocabularize(file_b))
    ops = editops(encode(src), encode(dest))
    # This only works for files with one edit!
    assert len(ops) == 1

    new: Optional[Vind]
    old: Optional[Vind]
    to_index = language.vocabulary.to_index_or_unk

    # Decode editop's format into our "Database-friendly" format.
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
    error_token = src_locs[at_least(0, src_pos - 1) if code == 'i' else src_pos]
    return FixEvent(edit, error_token.line)


def lists(it):
    a_list = []
    b_list = []
    for a, b in it:
        a_list.append(a)
        b_list.append(b)
    return a_list, b_list
