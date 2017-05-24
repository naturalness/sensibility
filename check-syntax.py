#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Checks if Python file has valid syntax. Returns non-zero otherwise.
"""

import sys

import sensibility.language.python


if __name__ == '__main__':
    language = sensibility.language.python.Python()
    if language.check_syntax(sys.stdin.read()):
        exit(0)
    else:
        exit(1)
