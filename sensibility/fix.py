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


from typing import Sequence

from sensibility.edit import Edit
from sensibility.model.lstm import DualLSTMModel
from sensibility.source_vector import to_source_vector


class LSTMFixerUpper:
    """
    Suggests fixes for syntax errors in a file with a dual LSTM model (which
    you must provide).

    TODO: Make an abc, probably.
    """

    def __init__(self, model: DualLSTMModel) -> None:
        self.model = model

    def fix(self, file: bytes) -> Sequence[Edit]:
        """
        Produces a ranked sequence of possible edits that will fix the file.
        If there are no possible fixes, the sequence will be empty.
        """
        vector = to_source_vector(file)
        predictions = self.model.predict_file(vector)

        return []
