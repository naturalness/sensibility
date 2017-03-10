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
Programs, and the edits that can be done to them.
"""

import abc
import random
from typing import Any, Hashable, Tuple, Optional

from .vocabulary import vocabulary, Vind
from .program import Program


# Serialization format:
#   - Name of edit class
#   - The location (exact semantics depend on edit class)
#   - New token (optional)
#   - Original token (optional)
Serialization = Tuple[str, int, Optional[Vind], Optional[Vind]]
PartialSerialization = Tuple[int, Optional[Vind], Optional[Vind]]


class Edit(abc.ABC, Hashable):
    """
    An abstract base class for edits:

     * Insertion
     * Deletion
     * Substitution

    All edits MUST hold the following property:

        program + edit + (-edit) == program
    """

    @abc.abstractmethod
    def additive_inverse(self) -> 'Edit':
        """
        Return the additive inverse of this edit.

        That is, adding this edit, and then adding the inverse will result in
        the original program:::

            program + edit + (-edit) == program
        """

    @abc.abstractmethod
    def apply(self, program: Program) -> Program:
        """
        Applies the edit to a program.
        """

    @abc.abstractmethod
    def serialize_components(self) -> PartialSerialization:
        """
        Return a tuple of the edit location (token stream index) and any
        relelvant vocabulary index.
        """

    @classmethod
    @abc.abstractmethod
    def create_random_mutation(cls, program: Program) -> 'Edit':
        """
        Creates a random mutation of this kind for the given program.
        """

    # The rest of the functions are given for free.

    @property
    def name(self) -> str:
        """
        Returns the name of this class.
        """
        return type(self).__name__.lower()

    def serialize(self) -> Serialization:
        """
        Return a triple (3-tuple) of the (name, location, token), useful for
        serializing and recreating Edit instances.
        """
        return (self.name, *self.serialize_components())

    def __neg__(self) -> 'Edit':
        """
        Return the additive inverse of this edit.
        """
        return self.additive_inverse()

    def __radd__(self, other: Program) -> Program:
        """
        Applies the edit to a program.
        """
        return self.apply(other)

    def __eq__(self, other: Any) -> bool:
        if type(self) == type(other):
            return self.serialize() == other.serialize()
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.serialize())


class Insertion(Edit):
    """
    An edit that wedges in a token at a random position in the file, including
    at the very end.

        A token is chosen randomly in the file. A random token from the
        vocabulary is inserted before this token (the end of the file is also
        considered a “token” for the purposes of the insertion operation).
    """

    def __init__(self, index: int, token: Vind) -> None:
        self.token = token
        self.index = index

    def additive_inverse(self) -> Edit:
        raise NotImplementedError

    def apply(self, program: Program) -> Program:
        return program.with_token_inserted(self.index, self.token)

    def serialize_components(self):
        raise NotImplementedError

    @classmethod
    def create_random_mutation(cls, program: Program) -> 'Insertion':
        """
        Creates a random insertion for the given program.
        """
        index = program.random_insertion_point()
        token = random_vocabulary_entry()
        return Insertion(index, token)


class Deletion(Edit):
    """
    An edit that deletes one token from the program

        A token is chosen randomly in the file. This token is removed from the
        file.
    """

    def __init__(self, index: int) -> None:
        self.index = index

    def additive_inverse(self) -> Edit:
        raise NotImplementedError

    def apply(self, program: Program) -> Program:
        return program.with_token_removed(self.index)

    def serialize_components(self):
        raise NotImplementedError

    @classmethod
    def create_random_mutation(cls, program: Program) -> 'Deletion':
        """
        Creates a random deletion for the given program.
        """
        index = program.random_token_index()
        return Deletion(index)


class Substitution(Edit):
    """
    An edit that swaps one token for another one.

        A token is chosen randomly in the file. This token is replaced with a
        random token from the vocabulary.
    """

    __slots__ = 'token', 'index', 'original_token'

    def __init__(self, index: int, *,
                 original_token: Vind,
                 replacement: Vind) -> None:
        self.token = replacement
        self.original_token = original_token
        self.index = index

    def additive_inverse(self) -> 'Substitution':
        return Substitution(self.index,
                            original_token=self.token,
                            replacement=self.original_token)

    def apply(self, program: Program) -> Program:
        return program.with_substitution(self.index, self.token)

    def serialize_components(self) -> PartialSerialization:
        return (self.index, self.token, self.original_token)

    @classmethod
    def create_random_mutation(cls, program: Program) -> 'Substitution':
        """
        Creates a random substitution for the given program.

        Ensures that the new token is NOT the same as the old token!
        """
        index = program.random_token_index()
        original_token = program[index]

        # Generate a token that is NOT the same as the one that is already in
        # the program!
        token = original_token
        while token == program[index]:
            token = random_vocabulary_entry()

        return Substitution(index, original_token=original_token,
                            replacement=token)


def random_vocabulary_entry() -> Vind:
    """
    Returns a random vocabulary index.
    """
    return Vind(random.randint(vocabulary.start_token_index + 1,
                               vocabulary.end_token_index - 1))
