#!/usr/bin/env python
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
Mutate and predict on one fold.

Does not **evaluate**, simply mutates files and performs predictions.

This evaluation:
 - is intended to test the algorithm given a number of different scenarios
 - demonstrates theoretical efficacy
 - is not representative of "real-life" errors

"""

import argparse
import json
import random
import sqlite3
import struct
import sys
import tempfile

from math import inf
from collections import OrderedDict
from itertools import islice
from pathlib import Path

from tqdm import tqdm

from condensed_corpus import CondensedCorpus
from detect import (
    Model, Agreement, check_syntax_file, check_syntax, tokenize_file, chop_prefix,
    PREFIX_LENGTH, consensus, index_of_max, rank, id_to_token
)
from model_recipe import ModelRecipe
from token_utils import Token
from training_utils import Sentences, one_hot_batch
from vectorize_tokens import vectorize_tokens
from vocabulary import vocabulary, START_TOKEN, END_TOKEN

# According to Campbell et al. 2014
MAX_MUTATIONS = 120

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('corpus', type=CondensedCorpus.connect_to)
parser.add_argument('model', type=ModelRecipe.from_string)
parser.add_argument('-k', '--mutations', type=int, default=MAX_MUTATIONS)

# Prefer /dev/shm, unless it does not exist. Use /dev/shm, because it is
# usually mounted as tmpfs ⇒ fast.
DATABASE_LOCATION = Path('/dev/shm') if Path('/dev/shm').exists() else Path('/tmp')
DATABASE_FILENAME = DATABASE_LOCATION / 'mutations.sqlite3'

# Schema for the results database.
SCHEMA = r"""
PRAGMA encoding = "UTF-8";
PRAGMA journal_mode = WAL;

CREATE TABLE mutant (
    hash        TEXT NOT NULL,  -- file hash
    type        TEXT NOT NULL,  -- 'addition', 'deletion', or 'substitution'
    location    TEXT NOT NUL,   -- location in the file (0-indexed)
    token       INTEGER,        -- changed token (not always applicable)

    PRIMARY KEY (hash, type, location, token)
);

CREATE TABLE prediction (
    model   TEXT NOT NULL,      -- model that created the prediction
    context BLOB NOT NULL,      -- input of the prediction

    data    BLOB NOT NULL,      -- prediction data, as a numpy array

    PRIMARY KEY (model, context)
);

-- same as `mutant`, but contains syntacitcally correct mutants.
CREATE TABLE correct_mutant (
    hash        TEXT NOT NULL,
    type        TEXT NOT NULL,
    location    TEXT NOT NUL,
    token       INTEGER,

    PRIMARY KEY (hash, type, location, token)
);
"""


class Sensibility:
    """
    A dual-intuition syntax error locator and fixer.
    """

    def __init__(self, recipe):
        forwards = recipe
        backwards = recipe.flipped()
        if backwards.forwards:
            forwards, backwards = backwards, forwards

        self.forwards_model = forwards
        self.backwards_model = backwards
        self.sentence_length = recipe.sentence
        # TODO: caches!
        # TODO: persistence!

    def predict(self, filename):
        """
        Predicts at each position in the file.

        As a side-effect, writes predictions to persistence.
        """

        # Get file vector for this (incorrect) file.
        with open(str(filename), 'rt', encoding='UTF-8') as script:
            tokens = tokenize_file(script)
        file_vector = vectorize_tokens(tokens)

        # Prepare every context.
        sent_forwards = Sentences(file_vector,
                                  size=self.sentence_length,
                                  backwards=False)
        sent_backwards = Sentences(file_vector,
                                   size=self.sentence_length,
                                   backwards=True)
        contexts = zip(sent_forwards, chop_prefix(sent_backwards))

        # Create predictions.
        for ((prefix, token), (suffix, _)) in contexts:
            self.forwards_model.predict(prefix)
            self.backwards_model.predict(suffix)

    @staticmethod
    def is_okay(filename):
        """
        Check if the syntax is okay.
        """
        with open(filename, 'rb') as source_file:
            return check_syntax_file(source_file)


class classproperty(object):
    """
    Like @property, but for classes!
    """
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)


class Mutation:
    """
    Base class for all mutations.  Provides methods for comparing and hashing
    mutations.
    """

    __slots__ = ('__weakref__',)

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and
            all(getattr(self, attr) == getattr(other, attr) for attr in
                self.__slots__)
        )

    def __hash__(self):
        if 'i' in self.__slots__:
            import pdb; pdb.set_trace()
        return hash(tuple(getattr(self, attr) for attr in self.__slots__))

    def __repr__(self):
        cls = type(self).__name__
        args = ', '.join(repr(getattr(self, attr)) for attr in self.__slots__)
        return '{}({})'.format(cls, args)

    @classproperty
    def name(cls):
        return cls.__name__


def random_token_from_vocabulary():
    """
    Gets a uniformly random token from the vocabulary as a vocabulary index.
    """
    # Generate anything EXCEPT the start and the end token.
    return random.randint(vocabulary.start_token_index + 1,
                          vocabulary.end_token_index - 1)


class Addition(Mutation):
    __slots__ = ('insertion_point', 'token')

    def __init__(self, insertion_point, token):
        self.insertion_point = insertion_point
        self.token = token

    def format(self, program, file=sys.stdout):
        """
        Applies the mutation to the source code and writes it to a file.
        """
        assert isinstance(program, SourceCode)
        insertion_point = self.insertion_point
        for index, token in enumerate(program):
            if index == insertion_point:
                file.write(vocabulary.to_text(self.token))
                file.write(' ')
            file.write(vocabulary.to_text(token))
            file.write(' ')
        file.write('\n')

    @classmethod
    def create_random_mutation(cls, program,
                               random_token=random_token_from_vocabulary):
        """
        Campbell et al. 2014:

            A location in the source file was chosen at random and a random
            token found in the same file was inserted there.

        random_token() is a function that returns a random token AS A
        VOCABULARY INDEX!
        """

        insertion_point = program.random_insertion_point()
        token = random_token()
        return cls(insertion_point, token)

    @property
    def location(self):
        return self.insertion_point


class Deletion(Mutation):
    __slots__ = ('index',)

    # Only one deletion.
    token = None

    def __init__(self, index):
        self.index = index

    def format(self, program, file=sys.stdout):
        """
        Applies the mutation to the source code and writes it to a file.
        """
        assert isinstance(program, SourceCode)
        delete_index = self.index
        for index, token in enumerate(program):
            if index == delete_index:
                continue
            file.write(vocabulary.to_text(token))
            file.write(' ')
        file.write('\n')

    @classmethod
    def create_random_mutation(cls, program):
        """
        Campbell et al. 2014:

            A token (lexeme) was chosen at random from the input source file
            and deleted. The file was then run through the querying and
            ranking process to determine where the first result with adjacent
            code appeared in the suggestions.
        """
        victim_index = program.random_index()
        return cls(victim_index)

    @property
    def location(self):
        return self.index


class Substitution(Mutation):
    __slots__ = ('index', 'token')
    def __init__(self, index, token):
        self.index = index
        self.token = token

    def format(self, program, file=sys.stdout):
        """
        Applies the mutation to the source code and writes it to a file.
        """
        assert isinstance(program, SourceCode)
        sub_index = self.index
        for index, token in enumerate(program):
            if index == sub_index:
                token = self.token
            file.write(vocabulary.to_text(token))
            file.write(' ')
        file.write('\n')

    @classmethod
    def create_random_mutation(cls, program,
                               random_token=random_token_from_vocabulary):
        """
        Campbell et al. 2014:

            A token was chosen at random and replaced with a random token
            found in the same file.

        random_token() is a function that returns a random token AS A
        VOCABULARY INDEX!
        """
        victim_index = program.random_index()
        token = random_token()
        return cls(victim_index, token)

    @property
    def location(self):
        return self.index


class SourceCode(Mutation):
    """
    A source code file.
    """
    def __init__(self, file_hash, tokens):
        self.hash = file_hash
        self.tokens = tokens
        self.first_index = 1 if tokens[0] == vocabulary.start_token_index else 0
        last_index = len(tokens) - 1
        self.last_index = (
            last_index - 1 if tokens[-1] == vocabulary.end_token_index else last_index
        )

        # XXX: hardcoded sentence lengths.
        self.usable_range = (self.first_index + 20, self.last_index - 20)

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    @property
    def usable_length(self):
        """
        How many tokens can we mutate in this file?
        """
        lower, upper = self.usable_range
        return max(0, upper - lower)

    @property
    def inner_length(self):
        return self.last_index - self.first_index + 1

    def random_insertion_point(self, randint=random.randint):
        """
        Produces a random insertion point in the program. Does not include start and end
        tokens.
        """
        return self.random_index(randint)
        #assert self.tokens[-1] == vocabulary.end_token_index
        #return randint(self.first_index, self.last_index + 1)


    def random_index(self, randint=random.randint):
        """
        Produces a random insertion point in the program. Does not include start and end
        tokens.
        """
        lower, upper = self.usable_range
        return randint(lower, upper)


def test():
    """
    Test source code mutations.
    """
    # The token stream INCLUDES start and stop tokens.
    program = SourceCode('DEADBEEF', [0, 86, 5, 31, 99])
    a = Addition.create_random_mutation(program)
    b = Addition.create_random_mutation(program)
    c = Addition.create_random_mutation(program)
    assert len({a, b, c, c, b, a, c, b}) == 3

    d1 = Addition(1, 1)
    d2 = Addition(1, 1)
    s = Substitution(1, 1)
    assert d1 == d2
    assert s != d1

    mutation = Addition.create_random_mutation(program)
    mutation.format(program)


class Persistence:
    """
    Persist every mutation, and enough data to reconstruct every single
    prediction.
    """

    def __init__(self):
        self._program = None

    @property
    def program(self):
        return self._program

    @property
    def current_source_hash(self):
        return self._program.hash

    @program.setter
    def program(self, new_program):
        assert isinstance(new_program, SourceCode)

    def add_mutant(self, mutation):
        assert isinstance(mutation, Mutation)
        raise

    def add_correct_file(self, mutation):
        """
        Records that a mutation created a correct file.
        """
        assert isinstance(mutation, Mutation)
        raise

    def add_prediction(self, *, model_recipe=None, context=None,
                       prediction=None):
        # TODO: add prediction to database.
        raise

    def get_prediction(self, *, model_recipe=None, context=None):
        # TODO: fetch prediction from the database
        raise

    def __enter__(self):
        # TODO: Open da database.
        raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: Close da database.
        raise


def main():
    # Requires: corpus, model data (backwards and forwards)
    args = parser.parse_args()

    corpus = args.corpus
    model_recipe = args.model
    fold_no = model_recipe.fold

    # Loads the parallel models.
    sensibility = Sensibility.from_model_recipe(model_recipe)

    # TODO: Corpus needs to return UN-ASSIGNED hashes!

    with Persistence() as persist:
        for file_hash, tokens in tqdm(()):
            program = SourceCode(file_hash, tokens)

            if program.usable_length < 0:
                # Program is useless for evaluation
                continue

            persist.program = program

            # TODO: Initialize the cache here!
            progress = tqdm(total=args.mutations * 3, leave=False)

            for mutation_kind in Addition, Deletion, Substitution:
                progress.set_description(mutation_kind.name)
                failures = 0
                mutations_seen = set()

                # Clamp down the maximum number of mutations.
                max_mutations = min(args.mutations, program.usable_length)
                max_failures = max_mutations

                while failures < max_failures and len(mutations_seen) < max_mutations:
                    if failures:
                        progress.set_description('Failures: {}'.format(failures))

                    mutation = mutation_kind.create_random_mutation(program)
                    if mutation in mutations_seen:
                        failures += 1
                        continue

                    # Write out the mutated file.
                    with tempfile.NamedTemporaryFile(mode='w+t', encoding='UTF-8') as mutated_file:
                        # Apply the mutatation and write it to disk.
                        mutation.format(program, mutated_file)
                        mutated_file.flush()

                        # Try the file, reject if it compiles.
                        if sensibility.is_okay(mutated_file.name):
                            persist.add_correct_file(mutation)
                            failures += 1
                            continue

                        # Do it!
                        predictions = sensibility.predict(mutated_file.name)

                    persist.add_mutant(mutation)
                    progress.update(1)
                    mutations_seen.add(mutation)

            progress.close()


if __name__ == '__main__':
    main()
