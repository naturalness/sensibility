#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from sensibility.language.python import Python


test_file = r"""#!/usr/bin/env python
# coding: us-ascii

'''
This is an example file.
'''

from __future__ import print_function


if __name__ == '__main__':
    print(f"This is the documentation: {__doc__}")
"""


def test_tokenize():
    python = Python()
    tokens = python.tokenize(test_file)
    # TODO: better test for this.
    assert len(tokens) == 30
