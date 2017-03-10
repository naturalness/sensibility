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
from typing import Any, Hashable, Tuple

from .vocabulary import vocabulary, Vind
from .program import Program


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
    def serialize_components(self) -> Tuple[int, Vind]:
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

    def serialize(self) -> Tuple[str, int, Vind]:
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


class Substitution(Edit):
    """
    An edit that swaps one token for another one.

    Campbell et al. 2014:

        A token was chosen at random and replaced with a random token
        found in the same file.
    """

    __slots__ = 'token', 'index'

    def __init__(self, index: int, token: Vind) -> None:
        self.token = token
        self.index = index

    def additive_inverse(self) -> Edit:
        ...

    def apply(self, program: Program) -> Program:
        return program.with_substitution(self.index, self.token)

    def serialize_components(self):
        ...

    @classmethod
    def create_random_mutation(cls, program: Program) -> 'Substitution':
        """
        Creates a random substitution for the given program.

        Ensures that the new token is NOT the same as the old token!
        """
        index = program.random_token_index()

        # Generate a token that is NOT the same as the one that is already in
        # the program!
        token = program[index]
        while token == program[index]:
            token = random_vocabulary_entry()

        return Substitution(index, token)


def random_vocabulary_entry() -> Vind:
    """
    Returns a random vocabulary index.
    """
    return Vind(random.randint(vocabulary.start_token_index + 1,
                               vocabulary.end_token_index - 1))
