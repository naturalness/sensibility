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
Provides a model-like class that queries a remote model via XMLRPC.
"""

from typing import Iterable, Sequence
from xmlrpc.client import Fault, ServerProxy  # type: ignore

import numpy as np

from sensibility import current_language
from sensibility.source_vector import SourceVector
from sensibility.vocabulary import Vind

from . import DualLSTMModel, TokenResult


class RemoteDualLSTMModel(DualLSTMModel):
    """
    Talks to a remotc XMLRPC server (see bin/prediction-server).
    """
    def __init__(self, server: ServerProxy) -> None:
        self.server = server

    @property
    def language_name(self) -> str:
        """
        The name of the current language.
        Intended use: the client can set its own language to match the one
        reported by the remote.
        """
        return self.server.get_language_name()

    def predict_file(self, vector: Sequence[Vind]) -> Iterable[TokenResult]:
        # The remote API is not quite the same.  It requires vocabulary
        # indices as bytes.
        serialized = SourceVector(vector).to_bytes()
        result = self.server.predict_file(serialized)

        # The result is returned as a triple-nested list:
        # Outermost list, corresponds to tokens.
        # Inside that are pairs of predictions (forwards, backwards).
        # Innermost is an array of float predictions.
        def deserialize_result():
            for fw, bw in result:
                yield TokenResult(np.array(fw, dtype=np.float32),
                                  np.array(bw, dtype=np.float32))
        return tuple(deserialize_result())

    @classmethod
    def connect(cls, port: int=8080) -> 'RemoteDualLSTMModel':
        server = ServerProxy(f'http://localhost:{port}')
        return cls(server)
