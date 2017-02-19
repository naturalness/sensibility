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
import random
import sqlite3
import sys
import tempfile
import io
import functools

from pathlib import Path

import numpy as np
from tqdm import tqdm

from condensed_corpus import CondensedCorpus, unblob
from detect import Model, check_syntax_file, tokenize_file, chop_prefix
from model_recipe import ModelRecipe
from token_utils import Token
from training_utils import Sentences, one_hot_batch
from vectorize_tokens import vectorize_tokens
from vocabulary import vocabulary, START_TOKEN, END_TOKEN

FAST_FILESYSTEM = False

# According to Campbell et al. 2014
MAX_MUTATIONS = 120

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('corpus', type=CondensedCorpus.connect_to)
parser.add_argument('model', type=ModelRecipe.from_string)
parser.add_argument('test_set', type=Path)
parser.add_argument('-k', '--mutations', type=int, default=MAX_MUTATIONS)
parser.add_argument('-n', '--limit', type=int, default=None)
parser.add_argument('-o', '--offset', type=int, default=0)

if FAST_FILESYSTEM:
    # Prefer /dev/shm, unless it does not exist. Use /dev/shm, because it is
    # usually mounted as tmpfs ⇒ fast.
    DATABASE_LOCATION = Path('/dev/shm') if Path('/dev/shm').exists() else Path('/tmp')
else:
    # Just choose the current working directory (it's probably a big HD/SSD)
    DATABASE_LOCATION = Path('.')

DATABASE_FILENAME = DATABASE_LOCATION / 'mutations.sqlite3'

# Schema for the results database.
SCHEMA = r"""
PRAGMA encoding = "UTF-8";
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS mutant (
    hash        TEXT NOT NULL,  -- file hash
    type        TEXT NOT NULL,  -- 'addition', 'deletion', or 'substitution'
    location    TEXT NOT NULL,  -- location in the file (0-indexed)
    token       INTEGER,        -- addition: the token inserted
                                -- deletion: the token deleted
                                -- substitution: the token that has replaced the old token

    PRIMARY KEY (hash, type, location, token)
);

CREATE TABLE IF NOT EXISTS prediction (
    model   TEXT NOT NULL,      -- model that created the prediction
    context BLOB NOT NULL,      -- input of the prediction

    data    BLOB NOT NULL,      -- prediction data, as a numpy array

    PRIMARY KEY (model, context)
);

-- same as `mutant`, but contains syntactically-correct mutants.
CREATE TABLE IF NOT EXISTS correct_mutant (
    hash        TEXT NOT NULL,
    type        TEXT NOT NULL,
    location    TEXT NOT NULL,
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

        self.forwards_model = Model(forwards.load_model())
        self.backwards_model = Model(backwards.load_model(), backwards=True)
        self.sentence_length = recipe.sentence
        self.persistence = None

        def _predict(model_recipe, model, tuple_context):
            """
            Does prediction, consulting the database first before consulting
            the model.
            """
            context = np.array(tuple_context, np.uint8)
            stashed_prediction = self.persistence.get_prediction(
                    model_recipe=model_recipe,
                    context=context
            )
            if stashed_prediction is None:

                prediction = model.predict(context)
                self.persistence.add_prediction(
                        model_recipe=model_recipe,
                        context=context,
                        prediction=prediction)
                return prediction
            return stashed_prediction

        # Create cached prediction functions.
        @functools.lru_cache(maxsize=2**12)
        def predict_forwards(prefix):
            return _predict(forwards, self.forwards_model, prefix)

        @functools.lru_cache(maxsize=2**12)
        def predict_backwards(suffix):
            return _predict(backwards, self.backwards_model, suffix)

        self.predict_forwards = predict_forwards
        self.predict_backwards = predict_backwards

    def predict(self, filename):
        """
        Predicts at each position in the file.

        Side-effect: writes predictions to persistence.
        """

        # Get file vector for this (incorrect) file.
        with open(str(filename), 'rt', encoding='UTF-8') as script:
            tokens = tokenize_file(script)
        file_vector = vectorize_tokens(tokens)

        # Create predictions.
        for (prefix, _), (suffix, _) in self.contexts(file_vector):
            self.predict_forwards(tuple(prefix))
            self.predict_backwards(tuple(suffix))

    def clear_cache(self):
        self.predict_forwards.cache_clear()
        self.predict_backwards.cache_clear()

    def cache_info(self):
        return self.predict_forwards.cache_info(), self.predict_backwards.cache_info()

    def contexts(self, file_vector):
        """
        Yield every context (prefix, suffix) in the given file vector.
        """
        sent_forwards = Sentences(file_vector,
                                  size=self.sentence_length,
                                  backwards=False)
        sent_backwards = Sentences(file_vector,
                                   size=self.sentence_length,
                                   backwards=True)
        return zip(sent_forwards, chop_prefix(sent_backwards))

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


class SourceCode:
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
    # TODO: Fix these tests to make up for 20 token "margin"
    """
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
    """

    p = Persistence(database=':memory:')

    original_model = ModelRecipe.from_string('python-f-310-5.4.5.h5')
    alternate_model = ModelRecipe.from_string('python-b-310-5.4.5.h5')
    sentence = (0, 10, 12, 67, 32)
    context = np.array(sentence, np.uint8)
    predictions = np.array([random.random() for _ in range(100)])

    with p:
        # Initially, it should return None.
        pred = p.get_prediction(model_recipe=original_model, context=context)
        assert pred is None

        p.add_prediction(model_recipe=original_model, context=context,
                         prediction=predictions)

        # Get the new prediction.
        result = p.get_prediction(model_recipe=original_model,
                                  context=context)
        assert result is not None
        assert all(x == y for x, y in zip(predictions, result))
        # Ensure the other context is not the same.
        assert p.get_prediction(model_recipe=alternate_model,
                                context=context) is None


def serialize_context(context):
    """
    Convert context to an unsigned 8-bit numpy array.
    """
    return to_blob(np.array(context, np.uint8))


def to_blob(np_array):
    """
    Serialize a Numpy array for storage.
    """
    filelike = io.BytesIO()
    np.save(filelike, np_array)
    return filelike.getbuffer()


class Persistence:
    """
    Persist every mutation, and enough data to reconstruct every single
    prediction.
    """

    def __init__(self, database=DATABASE_FILENAME):
        self._program = None
        self._dbname = database
        self._conn = None

    @property
    def program(self):
        """
        SourceCode object that is currently being mutated.
        """
        return self._program

    @program.setter
    def program(self, new_program):
        assert isinstance(new_program, SourceCode)
        self._program = new_program

    @property
    def current_source_hash(self):
        """
        hash of the current source file being mutated.
        """
        return self._program.hash

    def add_mutant(self, mutation):
        """
        Register a new complete mutation.
        """
        return self._add_mutant(mutation, usable_mutation=True)

    def add_correct_file(self, mutation):
        """
        Records that a mutation created a syntactically-correct file.
        """
        return self._add_mutant(mutation, usable_mutation=False)

    def _add_mutant(self, mutation, *, usable_mutation=None):
        assert self._conn
        assert isinstance(mutation, Mutation)

        if usable_mutation:
            sql = r'''
                INSERT INTO mutant(hash, type, location, token)
                     VALUES (:hash, :type, :location, :token)
            '''
        else:
            sql = r'''
                INSERT INTO correct_mutant(hash, type, location, token)
                     VALUES (:hash, :type, :location, :token)
            '''

        with self._conn:
            self._conn.execute(sql, dict(hash=self.current_source_hash,
                                         type=mutation.name,
                                         location=mutation.location,
                                         token=mutation.token))

    def add_prediction(self, *, model_recipe=None, context=None,
                       prediction=None):
        """
        Add the prediction (model, context) -> prediction
        """
        assert self._conn
        assert isinstance(model_recipe, ModelRecipe)
        assert isinstance(prediction, np.ndarray)

        with self._conn:
            self._conn.execute(r'''
                INSERT INTO prediction(model, context, data)
                VALUES (:model, :context, :data)
            ''', dict(model=model_recipe.identifier,
                      context=serialize_context(context),
                      data=to_blob(prediction)))

    def get_prediction(self, *, model_recipe=None, context=None):
        """
        Try to fetch the prediction from the database.

        Returns None if the entry is not found.
        """
        assert self._conn
        assert isinstance(model_recipe, ModelRecipe)
        cur = self._conn.execute(r'''
            SELECT data
            FROM prediction
            WHERE model = :model AND context = :context
        ''', dict(model=model_recipe.identifier,
                  context=serialize_context(context)))
        result = cur.fetchall()

        if not result:
            # Prediction not found!
            return None
        else:
            # Return the precomputed prediction.
            return unblob(result[0][0])

    def __enter__(self):
        # Connect to the database
        conn = self._conn = sqlite3.connect(str(self._dbname))
        # Initialize the database.
        with conn:
            conn.executescript(SCHEMA)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.close()
        self._conn = self._program = None


def write_cookie(filename, file_hash):
    """
    Write an indicator file whose existence signfies that this script actually
    did something. For use with the Makefile
    """
    with open(filename, 'wt') as cookie:
        cookie.write(file_hash)
        cookie.write('\n')


def main():
    # Requires: corpus, model data (backwards and forwards)
    args = parser.parse_args()

    corpus = args.corpus
    model_recipe = args.model
    fold_no = model_recipe.fold
    test_set_filename = args.test_set
    limit = args.limit
    offset = args.offset

    cookie_filename = (
        '{m.corpus}.{m.fold}.{m.epoch}.cookie'.format(m=model_recipe)
    )

    # Loads the parallel models.
    sensibility = Sensibility(model_recipe)

    # Load the test set. Assume the file hashes are already in random order.
    with open(str(test_set_filename)) as test_set_file:
        test_set = tuple(line.strip() for line in test_set_file
                         if line.strip())

    # Resize the test set
    upper_bound = offset + limit if limit is not None else len(test_set)
    test_set = test_set[offset:upper_bound]

    print("Considering", len(test_set), "test files to mutate...")

    with Persistence() as persist:
        sensibility.persistence = persist

        # Mutate each file in the test set.
        for file_hash in tqdm(test_set):
            # Get the file
            try:
                _, tokens = corpus[file_hash]
            except:
                continue
            program = SourceCode(file_hash, tokens)

            if program.usable_length < 0:
                # Program is useless for evaluation
                continue

            persist.program = program
            progress = tqdm(total=args.mutations * 3, leave=False)

            for mutation_kind in Addition, Deletion, Substitution:
                failures = 0
                mutations_seen = set()

                # Clamp down the maximum number of mutations.
                max_mutations = min(args.mutations, program.usable_length)
                max_failures = max_mutations

                while failures < max_failures and len(mutations_seen) < max_mutations:
                    mutation = mutation_kind.create_random_mutation(program)
                    if mutation in mutations_seen:
                        failures += 1
                        continue
                    mutations_seen.add(mutation)

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
                        sensibility.predict(mutated_file.name)

                    persist.add_mutant(mutation)
                    progress.update(1)

                    # Update the description.
                    symbol = (
                        '+' if mutation_kind is Addition else
                        '-' if mutation_kind is Deletion else
                        '𐇼'
                    )
                    progress.set_description((
                        '{symbol}[h/{f.hits} m/{f.misses}][h/{b.hits} m/{b.misses}]'
                        ' {trial:d}({failures})'
                        ).format(f=sensibility.predict_forwards.cache_info(),
                                 b=sensibility.predict_backwards.cache_info(),
                                 trial=len(mutations_seen),
                                 **vars())
                    )

            # Close the tqdm progress bar.
            progress.close()
            # Clear the LRU cache for the new file.
            sensibility.clear_cache()
            write_cookie(cookie_filename, file_hash)


if __name__ == '__main__':
    main()
