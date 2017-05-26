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

import datetime
from pathlib import PurePosixPath

import pytest  # type: ignore
from sqlalchemy import create_engine  # type: ignore
from sqlalchemy.sql import select  # type: ignore

from sensibility.miner.database import Database
from sensibility.miner.models import (
    RepositoryMetadata, SourceFile,
    SourceFileInRepository
)



def test_create(engine):
    db = database()


def test_inserts(database, repo_file):
    repository, _, _ = repo_file
    database.insert_repository(repository)
    database.insert_source_file_from_repo(repo_file)


@pytest.fixture
def repo_file() -> SourceFileInRepository:
    return SourceFileInRepository(
        repository=repository(),
        source_file=SourceFile(
            source=b'import sys\n\nprint("hello, world")\n'
        ),
        path=PurePosixPath('hello.py')
    )


@pytest.fixture
def repository() -> RepositoryMetadata:
    return RepositoryMetadata(
        owner='owner', name='name',
        revision='01b474d88e84cf745ab1d96405fd48279fcb5a11',
        license='mit',
        commit_date=datetime.datetime.utcnow()
    )


@pytest.fixture
def database():
    return Database(engine=engine())


@pytest.fixture
def engine():
    return create_engine('sqlite://', echo=True)
