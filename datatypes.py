#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Nothing can stop me from writing this module.
"""

from collections import namedtuple

from utils import is_hash, is_sha, sha256


class RepositoryID(namedtuple('RepositoryID', 'owner name')):
    def __str__(self):
        return r'{}/{}'.format(self.owner, self.name)
    # TODO: URLs


class Repository(namedtuple('Repository', 'id license revision')):
    def __init__(self, id_, license, revision):
        assert isinstance(id_, RepositoryID)
        assert license.lower() == license
        assert is_sha(revision)

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
        file_hash = sha256(source.encode('UTF-8'))
        return cls(repo, file_hash, source, path)

    @property
    def owner(self):
        return self.repo.owner

    @property
    def name(self):
        return self.repo.name


class ParsedSource(namedtuple('ParsedSource', 'id tokens ast')):
    def __init__(self, id_, tokens, ast):
        assert is_hash(id_)
