#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Tokenize stdin. Outputs pickled data.
"""

import sys
import pickle

import sensibility.language.python


if __name__ == '__main__':
    language = sensibility.language.python.Python()
    with open(sys.stdin.fileno(), 'rb') as input_file:
        with open(sys.stdout.fileno(), 'wb') as output_file:
            pickle.dump(language.tokenize(input_file), output_file)
