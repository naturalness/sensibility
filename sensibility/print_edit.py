#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from io import StringIO
from typing import Sequence, Iterable, Iterator, List, IO

from sensibility.edit import Edit, Insertion, Deletion, Substitution
from sensibility.lexical_analysis import Token, Lexeme
from sensibility.language import language
from sensibility.vocabulary import Vind



def determine_modification_index(edit: Edit) -> int:
    """
    Return the index WHEN things get strange.
    Everythng BEFORE that index is fine to print verbatim.
    """
    # It turns out that in every case, it's edit.index
    return edit.index


class Printer:
    def __init__(self, file: IO[str]) -> None:
        self.line_no = 1
        self.column = 0
        self.file = file

    def print_token(self, token: Token) -> None:
        if not token.spans_single_line:
            raise NotImplementedError

        # Create enough whitespace until the token.
        self.reach_line(token.line)
        self.reach_column(token.column)

        span = len(token.value)
        self.file.write(token.value)
        self.column += span

    def reach_column(self, column: int) -> None:
        assert self.column <= column
        length = column - self.column
        print(length)
        self.file.write(' ' * length)
        self.column += length

    def reach_line(self, line_no: int) -> None:
        assert self.line_no <= line_no
        while self.line_no < line_no:
            self.file.write("\n")
            self.line_no += 1
            self.column = 0

    def ending_newline(self) -> None:
        self.file.write('\n')


def print_edit(original: bytes, edit: Edit) -> bytes:
    tokens = list(language.tokenize(original))
    modification_index = determine_modification_index(edit)

    with StringIO() as file:
        p = Printer(file)
        for index in range(modification_index):
            p.print_token(tokens[index])

        return contents(file)


def contents(file: IO[str]) -> bytes:
    file.flush()
    file.seek(0)
    return file.read().encode('UTF-8')
