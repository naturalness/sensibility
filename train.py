#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2016 Eddie Antonio Santos <easantos@ualberta.ca>
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
Trains an LSTM from sentences in the vectorized corpus.
"""

# TODO: MAKE IT SAVE THE MODEL WITH ARCHITECTURE!

import argparse
from pathlib import Path

import numpy as np

from model_recipe import ModelRecipe
from vocabulary import vocabulary


# Based on White et al. 2015
SENTENCE_LENGTH = 20
SIGMOID_ACTIVATIONS = 300

# This is arbitrary, but it should be fairly small.
BATCH_SIZE = 512


def when_none_selected(**kwargs):
    """
    Print usage and die.
    """
    parser.print_help()
    exit(-1)


def when_new(vector_filename=None, backwards=None, sigmoid_activations=None,
             sentence_length=None, batch_size=None, fold=None, **kwargs):
    assert Path(vector_filename).exists()

    label = Path(vector_filename).stem
    recipe = ModelRecipe(label, backwards, sigmoid_activations,
                         sentence_length, fold)

    corpus = CondensedCorpus.connect_to(vector_filename)
    assert fold in corpus.fold_ids, (
        'Requested fold {} is not in {}'.format(fold, corpus.fold_ids)
    )
    corpus.disconnect()

    print("Compiling the model...")
    model = recipe.create_model()
    print("Done!")

    print("Creating batches with size:", batch_size)
    training_batches, eval_batches = recipe.create_batches(
        vector_filename, batch_size
    )

    print("Will train on", training_batches.samples_per_epoch, "samples")
    print("Training:", recipe.filename)
    model.fit_generator(iter(training_batches),
                        samples_per_epoch=training_batches.samples_per_epoch,
                        validation_data=iter(eval_batches),
                        nb_val_samples=eval_batches.samples_per_epoch // batch_size,
                        verbose=1,
                        pickle_safe=True,
                        nb_epoch=1)

    print("Saving model to", recipe.filename)
    model.save_weights(recipe.filename)


def when_continue(vector_filename=None, weights=None, batch_size=None,
                  **kwargs):
    assert Path(weights.filename).is_file()
    recipe = weights.next_epoch()

    print("Compiling the model and loading weights...")
    model = recipe.create_model_and_load_weights(weights.filename)
    print("Done!")

    print("Creating batches with size:", batch_size)
    training_batches, eval_batches = recipe.create_batches(
        args.vector_filename, args.batch_size
    )

    print("Will train on", training_batches.samples_per_epoch, "samples")
    print("Training:", recipe.filename)
    model.fit_generator(iter(training_batches),
                        samples_per_epoch=training_batches.samples_per_epoch,
                        validation_data=iter(eval_batches),
                        nb_val_samples=eval_batches.samples_per_epoch // batch_size,
                        verbose=1,
                        pickle_safe=True,
                        nb_epoch=1)

    print("Saving model.")
    model.save_weights(recipe.filename)


# Create the argument parser.
parser = argparse.ArgumentParser(description='Train one epoch from corpus '
                                 'of vectors, with fold assignments')
parser.set_defaults(action=when_none_selected)

subparsers = parser.add_subparsers(help='must choose one subaction')
new = subparsers.add_parser('new')
new.set_defaults(action=when_new)
extend = subparsers.add_parser('continue')
extend.set_defaults(action=when_continue)

new.add_argument('-f', '--fold', type=int, required=True,
                 help='which fold to use')
group = new.add_mutually_exclusive_group(required=True)
group.add_argument('--backwards', action='store_true')
group.add_argument('--forwards', action='store_false', dest='backwards')
new.add_argument('--sigmoid-activations', type=int, default=SIGMOID_ACTIVATIONS,
                 help='default: %d' % SIGMOID_ACTIVATIONS)
new.add_argument('--sentence-length', type=int, default=SENTENCE_LENGTH,
                 help='default: %d' % SENTENCE_LENGTH)

for subparser in new, extend:
    subparser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                           help='default: %d' % BATCH_SIZE)
    subparser.add_argument('vector_filename', metavar='vectors',
                           help='corpus of vectors, with assigned folds')

extend.add_argument('weights', type=ModelRecipe.from_string,
                    help='Loads weights and biases from a previous run')

if __name__ == '__main__':
    args = parser.parse_args()
    args.action(**vars(args))
