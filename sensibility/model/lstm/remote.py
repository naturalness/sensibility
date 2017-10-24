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

from typing import Sequence, Iterable
from xmlrpc.client import ServerProxy, Fault  # type: ignore

from sensibility.source_vector import SourceVector
from sensibility.vocabulary import Vind
from . import DualLSTMModel, TokenResult


class RemoteDualLSTMModel:
    """
    Talks to a remotc XMLRPC server (see bin/prediction-server).
    """
    def __init__(self, server: ServerProxy) -> None:
        self.server = server

    def predict_file(self, vector: Sequence[Vind]) -> Iterable[TokenResult]:
        from pprint import pprint

        # The remote API is not quite the same.
        # It requires vocabulary indices as bytes.
        serialized = SourceVector(vector).to_bytes()
        result = self.server.predict_file(serialized)
        pprint(result)
        raise NotImplementedError

    @classmethod
    def connect(cls, port: int=8080) -> 'RemoteDualLSTMModel':
        server = ServerProxy(f'http://localhost:{port}')
        return cls(server)
