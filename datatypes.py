#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

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


class SourceFile(namedtuple('SourceFile', 'repo hash source path')):
    def __init__(self, repo, hash_, source, path):
        assert isinstance(repo, RepositoryID)
        assert is_hash(hash_)

    @classmethod
    def create(cls, repo, source, path):
        if isinstance(repo, Repository):
            repo = repo.id
        if isinstance(source, str):
            source_bytes = source.encode('UTF-8')
        else:
            assert isinstance(source, bytes)
            source_bytes = source
        file_hash = sha256(source_bytes)
        return cls(repo, file_hash, source, path)

    @property
    def owner(self):
        return self.repo.owner

    @property
    def name(self):
        return self.repo.name


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
