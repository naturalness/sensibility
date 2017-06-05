#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os

from sensibility.language import language


def test_environment_variables():
    os.environ['SENSIBILITY_LANGUAGE'] = 'JavaScript'
    assert not language.is_initialized

    assert language.name == 'JavaScript'
    # TODO: Capture logging output?
    # [INFO]: Initializing language from environment: python

    # TODO: Actually try it out.
    """
    tokens = language.tokenize('import {Language} from "sensibility";')

    # TODO: Capture logging output?
    # [INFO]: Language: inferred from corpus as Python
    [Token(...), ...]
    """

    language.set_language('Python')
    assert language.name == 'Python'
    summary = language.summarize('from sensibility import language')
    assert summary.n_tokens == 4


# TODO: test to infer from the corpus?

def test_match_extension():
    language.set_language('JavaScript')
    assert language.matches_extension('hello.js')
    language.set_language('Python')
    assert language.matches_extension('hello.py')
