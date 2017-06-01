#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os


def test_environment_variables():
    os.putenv('SENSIBILITY_LANGAUGE', 'JavaScript')

    from sensibility.language import language
    assert not language.is_initialized

    assert language.name == 'javascript'
    # TODO: Capture logging output?
    # [INFO]: Initializing language from environment: python

    tokens = language.tokenize('import {Language} from "sensibility";')
    [Token(...), ...]

    language.set_language('Python')
    assert language.name == 'python'

    # TODO: Capture logging output?
    # [INFO]: Language: inferred from corpus as Python
    # language.tokenize('from sensibility import language')
    #[Token(...), ...]

# TODO: test to infer from the corpus?
# TODO: test for match_extensions()!
