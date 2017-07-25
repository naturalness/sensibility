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
    to_index = language.vocabulary.to_index
    return ''.join(chr(PUA_B_START + to_index(entry)) for entry in entries)


def tokenwise_distance(file_a: bytes, file_b: bytes) -> int:
    """
    Calculates the token-wise Levenshtein distance between two source files.
    """
    seq_a = encode(language.vocabularize(file_a))
    seq_b = encode(language.vocabularize(file_b))
    return distance(seq_a, seq_b)


class FixEvent:
    """
    A fix event is a collection of the edit that converts a file from good
    syntax to syntax error (the edit); from bad syntax to good syntax (the
    fix); and the line number of the token affected.
    """
    def __init__(self, edit: Edit, line_no: int) -> None:
        self.edit = edit
        self.line_no = line_no

    @property
    def fix(self):
        return -self.edit


def determine_edit(file_a: bytes, file_b: bytes) -> Edit:
    return determine_fix_event(file_a, file_b).edit


def determine_fix_event(file_a: bytes, file_b: bytes) -> FixEvent:
    """
    For two source files with Levenshtein distance of one, this returns the
    edit that converts the first file into the second file.
    """
    src = list(language.vocabularize(file_a))
    dest_locs, dest = lists(language.vocabularize_with_locations(file_b))
    ops = editops(encode(src), encode(dest))
    # This only works for files with one edit!
    assert len(ops) == 1

    # Decode editop's format.
    (type_name, src_pos, dest_pos), = ops

    new: Optional[Vind]
    old: Optional[Vind]
    to_index = language.vocabulary.to_index

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
    return FixEvent(edit, dest_locs[dest_pos].start.line)


def lists(it):
    a_list = []
    b_list = []
    for a, b in it:
        a_list.append(a)
        b_list.append(b)
    return a_list, b_list
