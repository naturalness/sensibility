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
Provides RepositoryInfo

>>> RepositoryID.parse('erlang/otp')
RepositoryID(owner='erlang', name='otp')
>>> str(RepositoryID('torvalds', 'linux'))
'torvalds/linux'
"""

import re
import hashlib
import datetime
from pathlib import PurePosixPath
from typing import NamedTuple


__all__ = [
    'RepositoryID'
    'RepositoryMetadata',
    'SourceFile',
    'SourceFileInRepository',
]
""


class _RepositoryID(NamedTuple):
    owner: str
    name: str


class RepositoryID(_RepositoryID):
    """
    Represents a repository
    """

    def __str__(self) -> str:
        return f'{self.owner}/{self.name}'

    @classmethod
    def parse(cls, text: str) -> 'RepositoryID':
        match = re.match(r'''^
            (?P<owner>[\w_\-.]+) / (?P<name>[\w_\-.]+)
        $''', text, re.VERBOSE)

        if match is None:
            raise ValueError(text)

        return cls(match.group('owner'), match.group('name'))


class RepositoryMetadata(NamedTuple):
    owner: str
    name: str
    revision: str
    license: str
    commit_date: datetime.datetime


class SourceFile:
    def __init__(self, source: bytes) -> None:
        self.source = source

    @property
    def filehash(self):
        m = hashlib.sha256()
        m.update(self.source)
        return m.hexdigest()

    def __repr__(self) -> str:
        return f"SourceFile({self.filehash!r}, source=...)"


class SourceFileInRepository(NamedTuple):
    repository: RepositoryMetadata
    source_file: SourceFile
    path: PurePosixPath
