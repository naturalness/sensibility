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
from typing import Any, Hashable, Tuple

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
    def serialize(self) -> Tuple[str, int, str]:
        """
        Return a tuple of the type name, the token, and the token insertion
        point.
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
