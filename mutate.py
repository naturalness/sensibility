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
Perform the evaluation for one fold.
 7. If it is an open-class, consider it UNFIXABLE!

This evaluation:
 - not representative of actual errors
 - demonstrates theoretical efficacy
 - intended to test algorithm given a number of different scenarios

In paper, discuss how the file can still be correct, even with a different
operation (give example).
"""

import argparse
import csv
import sys
import tempfile
import random
import time
from math import inf

from tqdm import tqdm

from condensed_corpus import CondensedCorpus
from detect import (
    Fix, Fixes, Model, Agreement, check_syntax_file, check_syntax, tokenize_file, chop_prefix,
    PREFIX_LENGTH, consensus, index_of_max, rank, id_to_token
)
from model_recipe import ModelRecipe
from token_utils import Token
from training_utils import Sentences, one_hot_batch
from vectorize_tokens import vectorize_tokens
from vocabulary import vocabulary, START_TOKEN, END_TOKEN

# According to Campbell et al. 2014
MAX_MUTATIONS = 120

parser = argparse.ArgumentParser()
parser.add_argument('corpus', type=CondensedCorpus.connect_to)
parser.add_argument('model', type=ModelRecipe.from_string)
parser.add_argument('-k', '--mutations', type=int, default=MAX_MUTATIONS)
parser.add_argument('--headers', action='store_true')


def random_token_from_vocabulary():
    """
    Gets a uniformly random token from the vocabulary as a vocabulary index.
    """
    # Generate anything EXCEPT the start and the end token.
    return random.randint(vocabulary.start_token_index + 1,
                          vocabulary.end_token_index - 1)


def to_r(condition):
    return 'TRUE' if condition else 'FALSE'


class Sensibility:
    """
    A dual-intuition syntax error locator and fixer.
    """

    def __init__(self, forwards_file, backwards_file, sentence_length):
        architecture = 'model-architecture.json'
        self.forwards_model = Model.from_filenames(weights=forwards_file,
                                                   architecture=architecture,
                                                   backwards=False)
        self.backwards_model = Model.from_filenames(weights=backwards_file,
                                                    architecture=architecture,
                                                    backwards=True)
        self.sentence_length = sentence_length

    @classmethod
    def from_model_recipe(cls, model_recipe):
        forwards = model_recipe
        backwards = model_recipe.flipped()
        if backwards.forwards:
            forwards, backwards = backwards, forwards
        return cls(forwards.filename, backwards.filename,
                   model_recipe.sentence)

    def detect_and_suggest(self, filename):
        """
        Detects the location of syntax errors, and suggests fixes.
        """

        # Get file vector for this (incorrect) file.
        with open(str(filename), 'rt', encoding='UTF-8') as script:
            tokens = tokenize_file(script)
        file_vector = vectorize_tokens(tokens)

        least_agreements = []
        forwards_predictions = []
        backwards_predictions = []

        sent_forwards = Sentences(file_vector,
                                  size=self.sentence_length,
                                  backwards=False)
        sent_backwards = Sentences(file_vector,
                                   size=self.sentence_length,
                                   backwards=True)

        # Predict every context.
        contexts = enumerate(zip(chop_prefix(tokens, PREFIX_LENGTH),
                                 sent_forwards, chop_prefix(sent_backwards)))

        # Find disagreements.
        for index, (token, (prefix, x1), (suffix, x2)) in contexts:
            prefix_pred = self.forwards_model.predict(prefix)
            suffix_pred = self.backwards_model.predict(suffix)

            # Get its harmonic mean
            mean = consensus(prefix_pred, suffix_pred)
            forwards_predictions.append(index_of_max(prefix_pred))
            backwards_predictions.append(index_of_max(suffix_pred))
            paired_rankings = rank(mean)
            min_token_id, min_prob = paired_rankings[0]
            least_agreements.append(Agreement(min_prob, index + PREFIX_LENGTH))

        fixes = Fixes(tokens, offset=0)

        # For the top disagreements, synthesize fixes.
        least_agreements.sort()
        for disagreement in least_agreements[:3]:
            pos = disagreement.index

            # Assume an addition. Let's try removing some tokens.
            fixes.try_remove(pos)

            # Assume a deletion. Let's try inserting some tokens.
            fixes.try_insert(pos, id_to_token(forwards_predictions[pos]))
            fixes.try_insert(pos, id_to_token(backwards_predictions[pos]))

        results = argparse.Namespace()
        results.ranks = least_agreements
        results.fixes = fixes
        return results

    @staticmethod
    def is_okay(filename):
        with open(filename, 'rb') as source_file:
            return check_syntax_file(source_file)


class classproperty(object):
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


class RecordElapsedTime:
    def __init__(self):
        self.end = None
        self.start = None

    @property
    def value(self):
        if self.end is None:
            raise RuntimeError('Timing not yet finished')
        return self.end - self.start

    def __float__(self):
        return self.value

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()


def find_rank(location, ranks):
    """
    Find of rank of the agreement.
    """
    for i, agreement in enumerate(ranks, start=1):
        if agreement.index == location:
            return i
    return inf


def test():
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
    Saves A LOT of data to the a CSV file.
    """
    def __init__(self, model_recipe):
        self.corpus = model_recipe.corpus
        self.fold_no = model_recipe.fold
        self.epoch = model_recipe.epoch
        self._main_file = None
        self._secondary_file = None
        self._program = None

    @property
    def filename(self):
        return '{corpus}.{fold_no}.{epoch}.csv'.format_map(vars(self))

    headers = (
        'fold epoch '
        'file.hash n.tokens '
        'trial elapsed.time '
        'mutation mutation.location mutation.token '
        'fix fix.location fix.token '
        'rank.correct syntax.ok'
    ).replace(' ', ',')

    @property
    def program(self):
        assert self._program_hash
        return self._program_hash

    @program.setter
    def program(self, new_program):
        assert isinstance(new_program, SourceCode)
        self._program_hash = new_program.hash
        self._trial = 1
        self._n_tokens = len(new_program)

    def increase_trial(self):
        self._trial += 1

    def add(self, *, mutation=None, elapsed_time=None, fix=None,
            rank=None, syntax_ok=None):
        assert isinstance(mutation, Mutation)
        assert isinstance(elapsed_time, RecordElapsedTime)
        assert isinstance(rank, int) or rank == inf
        assert fix is None or isinstance(fix, Fix)
        assert isinstance(syntax_ok, bool)

        if fix is None:
            fix_name = None
            fix_location = None
            fix_token = None
        else:
            fix_name = fix.name
            fix_location = fix.location
            # The token is the token TEXT, not a string.
            fix_token = vocabulary.to_index(fix.token.value)

        if rank == inf:
            rank = self._n_tokens + 1

        self._writer.writerow((
            self.fold_no, self.epoch,
            self.program, self._n_tokens,
            self._trial, float(elapsed_time),
            mutation.name, mutation.location, mutation.token,
            fix_name, fix_location, fix_token,
            rank, to_r(syntax_ok)
        ))

    def add_correct_file(self, mutation):
        """
        Records that a mutation created a correct file.
        """
        self._secondary_writer.writerow((
            self.program, mutation.name, mutation.location, mutation.token
        ))

    def __enter__(self):
        assert self._main_file is None
        self._main_file = csv_file = open(self.filename, 'a+t', encoding='UTF-8')
        self._writer = csv.writer(self._main_file)
        self._secondary_file = open(self.corpus + '.correct.csv', 'a+t',
                                    encoding='UTF-8')
        self._writer = csv.writer(self._main_file)
        self._secondary_writer = csv.writer(self._secondary_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._main_file.close()
        self._secondary_file.close()
        self._writer = None
        self._secondary_writer = None


def main():
    # Requires: corpus, model data (backwards and forwards)
    args = parser.parse_args()
    if args.headers:
        print(Persistence.headers)
        exit()

    corpus = args.corpus
    model_recipe = args.model
    fold_no = model_recipe.fold
    sensibility = Sensibility.from_model_recipe(model_recipe)

    with Persistence(model_recipe) as persist:
        for file_hash, tokens in tqdm(corpus.files_in_fold(fold_no)):
            program = SourceCode(file_hash, tokens)

            if program.usable_length < 0:
                # Program is useless for evaluation
                continue

            persist.program = program

            progress = tqdm(total=args.mutations * 3, leave=False)
            #for mutation_kind in Addition, Deletion, Substitution:
            for mutation_kind in Addition, Deletion:
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

                    with tempfile.NamedTemporaryFile(mode='w+t', encoding='UTF-8') as mutated_file:
                        # Apply the mutatation and flush it to disk.
                        mutation.format(program, mutated_file)
                        mutated_file.flush()

                        # Try the file, reject if it compiles.
                        if sensibility.is_okay(mutated_file.name):
                            persist.add_correct_file(mutation)
                            failures += 1
                            continue

                        # Do it!
                        with RecordElapsedTime() as elapsed_time:
                            results = sensibility.detect_and_suggest(mutated_file.name)

                    # TODO: Count how many times the suggestion IS the true result.

                    # Figure out common things.
                    rank = find_rank(mutation.location, results.ranks)

                    # Will create as many entries as there are fixes, or
                    # simply create one entry if there are no fixes.
                    if results.fixes:
                        for fix in results.fixes:
                            persist.add(mutation=mutation,
                                        elapsed_time=elapsed_time,
                                        syntax_ok=True,
                                        rank=rank,
                                        fix=fix)
                    else:
                        persist.add(mutation=mutation,
                                    elapsed_time=elapsed_time,
                                    syntax_ok=False,
                                    rank=rank)
                    persist.increase_trial()

                    progress.update(1)
                    mutations_seen.add(mutation)

            progress.close()


if __name__ == '__main__':
    main()
