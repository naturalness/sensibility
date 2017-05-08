#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class Language:
    """
    A programming language.
    """
    def __init__(self, name: str) -> None:
        self.name = name.lower()

    def __str__(self):
        return self.name.lower()

    def tokenize(self) -> None: ...
    def check_syntax(self) -> bool: ...

language = Language('python')
