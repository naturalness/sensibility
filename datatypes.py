#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Nothing can stop me from writing this module.
"""

from collections import namedtuple

from utils import is_hash, is_sha, sha256


class RepositoryID(namedtuple('RepositoryID', 'owner name')):
    pass


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
    def __init__(self, id_, license, revision):
        assert isinstance(id_, RepositoryID)

    @classmethod
    def create(cls, repo, source, path):
        file_hash = sha256(source.encode('UTF-8'))
        return cls(repo, file_hash, source, path)

    @property
    def owner(self):
        return self.id.owner

    @property
    def name(self):
        return self.id.name


class ParsedSource(namedtuple('ParsedSource', 'id tokens ast')):
    def __init__(self, id_, tokens, ast):
        assert is_hash(id_)
