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

# This is the WRONG place to store the WordCount class!
from sensibility.language import SourceSummary
from sensibility.miner.corpus import Corpus
from sensibility.miner.models import (
    RepositoryMetadata, SourceFile,
    SourceFileInRepository
)


def test_create(engine):
    db = empty_corpus()


def test_insert_source_summary(empty_corpus, repo_file) -> None:
    repository, _, _ = repo_file
    empty_corpus.insert_repository(repository)
    empty_corpus.insert_source_file_from_repo(repo_file)
    empty_corpus.insert_source_summary(repo_file.filehash,
                                       SourceSummary(2, 3))


def test_insert_and_retrieve(corpus, source_file):
    source_code = corpus.get_source(source_file.filehash)
    assert source_code == source_file.source
    assert SourceFile(source_code).filehash == source_file.filehash
    # Ensure we can the item back from the empty_corpus.
    assert corpus[source_file.filehash] == source_file.source


def test_metadata(corpus) -> None:
    assert corpus.language == 'Python'


def test_insert_duplicate(corpus: Corpus,
                          repo_file: SourceFileInRepository) -> None:
    """
    Insert the same file with two DIFFERENT paths.
    """
    with pytest.raises(Exception):
        corpus.insert_source_file_from_repo(repo_file)

    # Insert the SAME file, under a different path.
    repo_file2 = SourceFileInRepository(
        repository=repo_file.repository,
        source_file=repo_file.source_file,
        path=PurePosixPath('world.py')
    )

    # This should work:
    corpus.insert_source_file_from_repo(repo_file2)


################################## Fixtures ##################################

@pytest.fixture
def source_file() -> SourceFile:
    return SourceFile(
        source=b'import sys\n\nprint("hello, world")\n'
    )


@pytest.fixture
def repo_file() -> SourceFileInRepository:
    return SourceFileInRepository(
        repository=repository(),
        source_file=source_file(),
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
def corpus() -> Corpus:
    db = empty_corpus()
    db.set_metadata(language='Python',
                    started=datetime.datetime.utcnow())
    entry = repo_file()
    db.insert_repository(entry.repository)
    db.insert_source_file_from_repo(entry)
    return db


@pytest.fixture
def empty_corpus() -> Corpus:
    return Corpus(engine=engine())


@pytest.fixture
def engine():
    return create_engine('sqlite://', echo=True)


"""
Tests the Corpus class using a test.sql.
"""

'''
import sqlite3
from pathlib import Path

from sensibility.corpus import Corpus

_DIRECTORY = Path(__file__).parent

import pytest
slow = pytest.mark.skipif(
        not pytest.config.getoption("--runslow"),
        reason="need --runslow option to run"
)

def test_meta():
    """
    Tests opening and iterating the sample corpus.
    """
    corpus = Corpus(new_connection_for_testing())
    assert len(corpus) == len(list(corpus))
    hashes = next(iter(corpus))
    assert all(isinstance(x, str) for x in hashes)


def test_iterate():
    corpus = Corpus(new_connection_for_testing())
    files = tuple(corpus)
    assert len(files) == 2
    test_file_hash = '86cc829b0a086a9f655b942278f6be5c9e5057c34459dafafa312dfdfa3a27d0'
    assert next((corpus.iterate())) == test_file_hash


def test_get_source():
    corpus = Corpus(new_connection_for_testing())
    file_hash = next(iter(corpus))
    source = corpus.get_source(file_hash)
    assert isinstance(source, bytes)
    assert source == b'(name) => console.log(`Hello, ${name}!`);'


def test_get_by_prefix():
    corpus = Corpus(new_connection_for_testing())
    fh = '86cc829b0a086a9f655b942278f6be5c9e5057c34459dafafa312dfdfa3a27d0'
    hashes = corpus.get_hashes_by_prefix('86cc829')
    assert len(hashes) == 1
    assert hashes[0] == fh


@slow
def test_real_database():
    """
    Connects to the real, actual, database, and ENSURES all the files exists
    and have sources.

    The database MUST be in the following directory:

    ../data/javascript-source.sqlite3
    """
    path = _DIRECTORY.parent / 'data' / 'javascript-sources.sqlite3'
    assert path.exists()
    corpus = Corpus.connect_to(path)
    assert len(corpus) >= 400_000
    for file_hash in corpus:
        source = corpus.get_source(file_hash)
        assert isinstance(source, bytes)
        _, _, path = corpus.file_info(file_hash)
        assert path.endswith('.js'), 'Found non-javascript files?'
        assert not path.endswith('.min.js'), 'Found minified file'


def new_connection_for_testing():
    """
    Return an SQLite3 connection suitable for testing.
    """
    conn = sqlite3.connect(':memory:')
    with open(str(_DIRECTORY / 'test.sql')) as sqlfile:
        conn.executescript(sqlfile.read())
    return conn
'''
