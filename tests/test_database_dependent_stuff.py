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
Stuff that depends on the actual databases that exist.
"""

import pytest


@pytest.mark.xfail
def test_get_samples_per_batches_hack() -> None:
    from sensibility.language import language
    from sensibility.model.lstm.loop_batches import get_samples_per_batches_hack
    language.set_language('python')
    assert 252 == get_samples_per_batches_hack({
        '7bd4a8a55e103450d99a049ac68daa0edee1646d416fbc96c97eeb73ff8a28d0'
    })
