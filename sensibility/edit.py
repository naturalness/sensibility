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
The edits that can performed to SourceVector instances.
"""

import random
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Hashable, Optional, Tuple, Type, TypeVar

from .source_vector import SourceVector
from .vocabulary import Vind

# Serialization format:
#   - Name of edit class
#   - The location (exact semantics depend on edit class)
#   - New token (optional)
#   - Original token (optional)
Serialization = Tuple[str, int, Optional[Vind], Optional[Vind]]
PartialSerialization = Tuple[int, Optional[Vind], Optional[Vind]]


class Edit(metaclass=ABCMeta):
    """
    An abstract base class for edits:

     * Insertion
     * Deletion
     * Substitution

    All edits MUST hold the following property:

        program + edit + (-edit) == program
    """

    code: str
    index: int
    _subclasses: Dict[str, Type['Edit']] = {}

    def __init_subclass__(cls: Type['Edit']) -> None:
        """
        Registers each subclass (Insertion, Deletion, Substitution) with a
        single-letter code. Used for serialization and deserialization.
        """
        assert hasattr(cls, 'code')
        code = cls.code
        assert code not in Edit._subclasses, (
            f"Error creating {cls.__name__}: code {code!r} already exists"
        )
        Edit._subclasses[code] = cls

    @abstractmethod
    def additive_inverse(self) -> 'Edit':
        """
        Return the additive inverse of this edit.

        That is, adding this edit, and then adding the inverse will result in
        the original program:::

            program + edit + (-edit) == program
        """

    @abstractmethod
    def apply(self, program: SourceVector) -> SourceVector:
        """
        Applies the edit to a program.
        """

    @abstractmethod
    def serialize_components(self) -> PartialSerialization:
        """
        Return a tuple of the edit location (token stream index) and any
        relelvant vocabulary index.
        """

    @classmethod
    @abstractmethod
    def create_random_mutation(cls, program: SourceVector) -> 'Edit':
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
        Return a quadruplet (4-tuple) of
        (code, location, token, original_token),
        useful for serializing and recreating Edit instances.
        """
        return (self.code, *self.serialize_components())

    def __neg__(self) -> 'Edit':
        """
        Return the additive inverse of this edit.
        """
        return self.additive_inverse()

    def __radd__(self, other: SourceVector) -> SourceVector:
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

    @classmethod
    def deserialize(cls, code: str, location: int,
                    token: Optional[Vind],
                    original_token: Optional[Vind]) -> 'Edit':
        """
        Deserializes an edit from tuple notation.
        """
        subclass = cls._subclasses[code]
        if subclass is Insertion:
            assert original_token is None
            return Insertion(location, not_none(token))
        elif subclass is Deletion:
            return Deletion(location, not_none(original_token))
        else:
            assert subclass is Substitution
            return Substitution(
                location,
                replacement=not_none(token),
                original_token=not_none(original_token)
            )


class Insertion(Edit):
    """
    An edit that wedges in a token at a random position in the file, including
    at the very end.

        A token is chosen randomly in the file. A random token from the
        vocabulary is inserted before this token (the end of the file is also
        considered a “token” for the purposes of the insertion operation).

    Index refers to the index in the token stream to insert BEFORE. Hence it
    has a range of [0, len(file)] inclusive, where inserting at index 0 means
    inserting BEFORE the first token, and inserting at index len(file) means
    inserting after the last token in the file (pedantically, it means
    inserting before the imaginary end-of-file token).
    """

    __slots__ = 'index', 'token'

    code = 'i'

    def __init__(self, index: int, token: Vind) -> None:
        self.token = token
        self.index = index

    def __repr__(self) -> str:
        from sensibility.language import language
        text_token = language.vocabulary.to_text(self.token)
        return f'Insertion({self.index}, {self.token} or {text_token!r})'

    def additive_inverse(self) -> Edit:
        return Deletion(self.index, self.token)

    def apply(self, program: SourceVector) -> SourceVector:
        return program.with_token_inserted(self.index, self.token)

    def serialize_components(self) -> PartialSerialization:
        return (self.index, self.token, None)

    @staticmethod
    def create_mutation(program: SourceVector,
                        index: int, token: Vind) -> 'Insertion':
        return Insertion(index, token)

    @classmethod
    def create_random_mutation(cls, program: SourceVector) -> 'Insertion':
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

    __slots__ = 'index', 'original_token'

    code = 'x'

    def __init__(self, index: int, original_token: Vind) -> None:
        self.index = index
        self.original_token = original_token

    def __repr__(self) -> str:
        from sensibility.language import language
        as_text = language.vocabulary.to_text(self.original_token)
        return f'Deletion({self.index}, {self.original_token} or {as_text!r})'

    def additive_inverse(self) -> Edit:
        # Insert the deleted token back again
        return Insertion(self.index, self.original_token)

    def apply(self, program: SourceVector) -> SourceVector:
        return program.with_token_removed(self.index)

    def serialize_components(self) -> PartialSerialization:
        return (self.index, None, self.original_token)

    @staticmethod
    def create_mutation(program: SourceVector, index: int) -> 'Deletion':
        return Deletion(index, program[index])

    @classmethod
    def create_random_mutation(cls, program: SourceVector) -> 'Deletion':
        """
        Creates a random deletion for the given program.
        """
        index = program.random_token_index()
        return Deletion(index, program[index])


class Substitution(Edit):
    """
    An edit that swaps one token for another one.

        A token is chosen randomly in the file. This token is replaced with a
        random token from the vocabulary.
    """

    __slots__ = 'token', 'index', 'original_token'

    code = 's'

    def __init__(self, index: int, *,
                 original_token: Vind,
                 replacement: Vind) -> None:
        self.token = replacement
        self.original_token = original_token
        self.index = index

    def __repr__(self) -> str:
        from sensibility.language import language
        new_text = language.vocabulary.to_text(self.token)
        old_text = language.vocabulary.to_text(self.original_token)
        return (
            f'Substitution({self.index}, '
            f'original_token={self.original_token} or {old_text!r}, '
            f'new_token={self.token} or {new_text!r})'
        )

    def additive_inverse(self) -> 'Substitution':
        # Simply swap the tokens again.
        return Substitution(self.index,
                            original_token=self.token,
                            replacement=self.original_token)

    def apply(self, program: SourceVector) -> SourceVector:
        return program.with_substitution(self.index, self.token)

    def serialize_components(self) -> PartialSerialization:
        return (self.index, self.token, self.original_token)

    @staticmethod
    def create_mutation(program: SourceVector,
                        index: int, token: Vind) -> 'Substitution':
        return Substitution(index,
                            original_token=program[index],
                            replacement=token)

    @classmethod
    def create_random_mutation(cls, program: SourceVector) -> 'Substitution':
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

        return Substitution(index,
                            original_token=original_token,
                            replacement=token)


def random_vocabulary_entry() -> Vind:
    """
    Returns a random vocabulary index. Excludes the start and end tokens.
    """

    from sensibility.language import language
    vocabulary = language.vocabulary
    return Vind(random.randint(vocabulary.end_token_index + 1,
                               len(vocabulary)))


T = TypeVar('T')


def not_none(item: Optional[T]) -> T:
    """
    Return the item unchanged, but raises ValueError if the value is None.
    """
    if item is not None:
        return item
    raise ValueError('Item cannot be None')
