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
Compatibility file.
"""

import warnings

warnings.warn("Should 'from sensibility import Vectors' instead",
              DeprecationWarning)

# Pretend we're the old module.
from sensibility import Vectors as CondensedCorpus

# TODO: remove this this is weird here.
def unblob(blob):
    assert isinstance(blob, bytes)
    with io.BytesIO(blob) as filelike:
        return np.load(filelike)
