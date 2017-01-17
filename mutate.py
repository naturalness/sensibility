#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Do evaluation.

Requirements:

 1. Aware of fold assignments
 2. Uses random files from the validation fold
 3. Does K times mutations of each kind per file:
    * Addition
    * Deletion
    * Substitution
 4. Constrict the mutation to the inner context (make it fair...)
 5. Evaluate location of change => MRR
 6. Can it fix this change?

Reread UnnaturalCode, and my paper.

HOW DO I PERSIST THIS DATA?
The same way I persist any data.
TRY TO TAKE OVER T--sqlite3.
"""

import argparse
import sys
import tempfile
import random

from vocabulary import vocabulary
from model_recipe import ModelRecipe


# According to Campbell et al. 2014
MAX_MUTATIONS = 120

def get_random_token_from_vocabulary():
  """
  Gets a random token from a uni
  """
  

class Model:
  ...

class Mutation:
    ...

class Addition(Mutation):
    __slots__ = ('index', 'token')

    def format(self, program, file=sys.stdout):
        """
        Applies the mutation to the source code and writes it to a file.
        """
        raise NotImplementedError

    @classmethod
    def create_random_mutation(cls, program):
        insertion_point = ...
        token = ... # needs to create a new token


class Deletion(Mutation):
    __slots__ = ('index')

    def __init__(self, index):
        self.index = index

    def format(self, program, file=sys.stdout):
        """
        Applies the mutation to the source code and writes it to a file.
        """
        delete_index = self.index
        for index, token in program:
            if index == delete_index:
                continue
            file.write(str(token))

    @classmethod
    def create_random_mutation(cls, program, randint=random.randint):
        """
        Campbell et al. 2014:

        A token (lexeme) was chosen at random from the input source file and
        deleted. The file was then run through the querying and ranking
        process to determine where the first result with adjacent code
        appeared in the suggestions.

        Random in same file? Or random in corpus?

        """
        # TODO: random token function.
        victim_index = randint(0, len(program))
        return cls(victim_index)


class Substitution(Mutation):
    __slots__ = ('index', 'token')
    def format(self, program, file=sys.stdout):
        """
        Applies the mutation to the source code and writes it to a file.
        """
        raise NotImplementedError

    @classmethod
    def create_random_mutation(cls, program, get_random_token=):
        """
        A token was chosen at random and replaced with a random token found in
        the same file.
        """
        victim_index = ...
        token = ... # needs to create a new token from the vocabulary


class SourceCode(Mutation):
    """
    A source code file.
    """
    def __init__(self, file_hash, tokens):
        self._file = file_hash
        self._tokens = tuple(tokens)

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

parser = argparse.ArgumentParser()
parser.add_argument('corpus', CondensedCorpus.connect_to)
parser.add_argument('model', 'a model')



def main():
    # Requires: corpus, model data forwards, backwards.
    # Open the corpus.
    model = ...
    # Josh style!
    for file_hash, tokens in corpus.files_in_fold(fold_no):
        program = SourceCode(file_hash, tokens)
        for mutation_kind in Addition, Deletion, Substitution:
            for _ in range(MAX_MUTATIONS):
                mutation = mutation_kind.create_random_mutation(program)
                with open(...) as tempfile:
                    mutation.format(mutation, tempfile)
                    results = model.detect(tempfile.name)
                    persist(program, mutation, results)
