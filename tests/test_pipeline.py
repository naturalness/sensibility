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


from sensibility.language.python import PythonPipeline
from sensibility.token_utils import Position
from location_factory import LocationFactory

pipeline = PythonPipeline()


source = r'''#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("Hello, world!")
exit(1)
'''


def test_works_on_source():
    tokens = list(pipeline.execute(source))
    assert len(tokens) == 10
    assert tokens == [
        'IDENTIFIER', '(', 'STRING', ')', 'NEWLINE',
        'IDENTIFIER', '(', 'NUMBER', ')', 'NEWLINE',
    ]
    # Assert we can stringify it and convert it back?


def test_returns_locations():
    tokens = list(pipeline.execute_with_locations(source))
    assert len(tokens) == 10
    loc = LocationFactory(Position(line=4, column=0))
    assert tokens == [
        (loc.across(len("print")), 'IDENTIFIER'),
        (loc.single(), '('),
        (loc.across(len('"Hello, World!"')), 'STRING'),
        (loc.single(), ')'),
        (loc.newline(), 'NEWLINE'),

        (loc.across(len("exit")), 'IDENTIFIER'),
        (loc.single(), '('),
        (loc.across(len('0')), 'NUMBER'),
        (loc.single(), ')'),
        (loc.newline(), 'NEWLINE')
    ]
