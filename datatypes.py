#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright 2016 Eddie Antonio Santos <easantos@ualberta.ca>
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
Nothing can stop me from writing this module.
"""

import json
from collections import namedtuple

from utils import is_hash, is_sha, sha256


class RepositoryID(namedtuple('RepositoryID', 'owner name')):
    """
    >>> repo = RepositoryID.parse('eddieantonio/bop')
    >>> repo.archive_url()
    'https://api.github.com/repos/eddieantonio/bop/zipball/master'
    """

    def __str__(self):
        return r'{}/{}'.format(self.owner, self.name)

    def archive_url(self, format="zipball", revision="master"):
        return (
            "https://api.github.com"
            "/repos/{owner}/{repo}/{format}/{rev}"
        ).format(owner=self.owner, repo=self.name,
                 format=format, rev=revision)

    @property
    def api_url(self):
        return (
            "https://api.github.com/repos/{owner}/{repo}"
        ).format(owner=self.owner, repo=self.name)

    @classmethod
    def parse(cls, string):
        owner, name = string.split('/', 1)
        return cls(owner, name)


class Repository(namedtuple('Repository', 'id license revision')):
    def __init__(self, id_, license, revision):
        assert isinstance(id_, RepositoryID)
        if license is not None:
            assert license.lower() == license
        # Originally, I wanted a specific git SHA, but now I'll take
        # any valid branch name.
        #assert is_sha(revision)

    @classmethod
    def create(self, owner, name, license, revision):
        return Repository(RepositoryID(owner, name), license, revision)

    @property
    def owner(self):
        return self.id.owner

    @property
    def name(self):
        return self.id.name


class SourceFile(namedtuple('SourceFile', 'repo hash source_bytes path')):
    def __init__(self, repo, hash_, source_bytes, path):
        assert isinstance(source_bytes, bytes)
        assert isinstance(repo, RepositoryID)
        assert is_hash(hash_)

    @classmethod
    def create(cls, repo, source_bytes, path):
        assert isinstance(source_bytes, bytes)
        if isinstance(repo, Repository):
            repo = repo.id
        file_hash = sha256(source_bytes)
        return cls(repo, file_hash, source_bytes, path)

    @property
    def owner(self):
        return self.repo.owner

    @property
    def name(self):
        return self.repo.name

    @property
    def source(self):
        return self.source_bytes.decode('UTF-8')


class ParsedSource(namedtuple('ParsedSource', 'hash tokens ast')):
    def __init__(self, hash_, tokens, ast):
        assert is_hash(hash_)
        assert isinstance(tokens, list)
        assert isinstance(ast, dict)

    @property
    def tokens_as_json(self):
        return json.dumps(self.tokens)

    @property
    def ast_as_json(self):
        return json.dumps(self.ast)
