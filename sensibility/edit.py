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


class Edit(abc.ABC):
    """
    An abstract base class for edits:

     * Insertion
     * Deletion
     * Substitution

    All edits MUST hold the following property:

        program + edit + (-edit) == program
    """

    @abc.abstractmethod
    def __neg__(self) -> 'Edit':
        """
        Return the additive inverse of this edit.

        That is, adding this edit, and then adding the inverse will result in
        the original program:::

            program + edit + (-edit) == program

        """
