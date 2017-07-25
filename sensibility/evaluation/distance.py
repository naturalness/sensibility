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

import sqlite3
from typing import Iterable, Iterator, NewType, Optional, Tuple, cast

from javalang.tokenizer import LexerError  # type: ignore
from Levenshtein import distance, editops  # type: ignore

from tqdm import tqdm  # type: ignore

from sensibility.language import language
# TODO: unhardcode Java things
from sensibility.language.java import java, java2sensibility
from sensibility.vocabulary import Vind
from sensibility.lexical_analysis import Lexeme
from .mistakes import Mistakes, Mistake
# TODO: use sensibility's well-tested classes instead
from .mistakes import Edit


# Supplementary Private Use Area B
PUA_B_START = 0x100000


def to_edit_code(name: str) -> str:
    """
    Convert a string returned from Levenshtein.editops into a constant.
    """
    return {
        'replace': 's',
        'insert': 'i',
        'delete': 'x'
    }[name]


def tokens2seq(tokens: Iterable[Lexeme],
               to_entry=java2sensibility) -> str:
    """
    Converts a sequence of tokens into an encoded string, that preserves
    distinctness between tokens. This string can be operated by the
    Levenshtein module.
    """
    # Injective maps of each vocabulary entry to a codepoint in the private
    # use area.
    # distance() works on codepoints, so this effectively makes distance()
    # work on token classes.
    to_index = language.vocabulary.to_index
    return ''.join(
        chr(PUA_B_START + to_index(to_entry(token)))
        for token in tokens
    )


def tokenwise_distance(file_a: bytes, file_b: bytes) -> int:
    """
    Calculates the token-wise Levenshtein distance between two source files.
    """
    seq_a = tokens2seq(java.tokenize(file_a))
    seq_b = tokens2seq(java.tokenize(file_b))
    # TODO: use editops as a post-processing step!
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
    src = tokens2seq(java.tokenize(file_a))
    dest = tokens2seq(java.tokenize(file_b))
    ops = editops(src, dest)
    # This only works for files with one edit!
    assert len(ops) == 1

    # Decode editop's format.
    (type_name, src_pos, dest_pos), = ops

    new: Optional[Vind]
    old: Optional[Vind]

    if type_name == 'insert':
        code, new, old = 'i', from_pua(dest[dest_pos]), None
    elif type_name == 'delete':
        code, new, old = 'x', None, from_pua(src[src_pos])
    elif type_name == 'replace':
        code, new, old = 's', from_pua(dest[dest_pos]), from_pua(src[src_pos])
    else:
        raise ValueError(f'Cannot handle operation: {ops}')
    # Note: in the case of insertions, src_pos will be the index to insert
    # BEFORE, which is exactly what Edit.deserialize wants; src_position also
    # works for deletions, and substitutions.
    edit = Edit.deserialize(code, src_pos, new, old)
    return FixEvent(edit, 0)


def from_pua(char: str) -> Vind:
    """
    Undoes the injective mapping between vocabulary IDs and Private Use Area
    code poiunts.
    """
    assert ord(char) >= 0x100000
    return cast(Vind, ord(char) & 0xFFFF)


if __name__ == '__main__':
    conn = sqlite3.connect('java-mistakes.sqlite3')
    # HACK! Make writes super speedy by disregarding durability.
    conn.execute('PRAGMA synchronous = OFF')
    mistakes = Mistakes(conn)
    for mistake in tqdm(mistakes):
        try:
            dist = tokenwise_distance(mistake.before, mistake.after)
        except LexerError:
            continue
        mistakes.insert_distance(mistake, dist)
