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

import argparse
from pathlib import Path

import numpy as np

from vocabulary import vocabulary


# Based on White et al. 2015
SENTENCE_LENGTH = 20
SIGMOID_ACTIVATIONS = 300

# This is arbitrary, but it should be fairly small.
BATCH_SIZE = 512

# {corpus}-{forward/backward}-{sigmoid}-{sentence}.{fold}.{epoch}.h5
# e.g., javascript-f-300-20.4.3.h5

# TODO: remove this, probably.
class ModelRecipe:
    """
    All the data of an existing model.
    >>> m = ModelRecipe.create('javascript-f-300-20.4.3.h5')
    >>> m.filename
    'javascript-f-300-20.4.3.h5'
    >>> m.next_epoch().filename
    'javascript-f-300-20.4.4.h5'
    """

    @classmethod
    def from_string(cls, raw_string):
        try:
            prefix, fold, epoch, _extension = raw_string.split('.')
            corpus, direction, sigmoid, sentence = prefix.split('-')

            fold = int(fold)
            epoch = int(epoch)
            sigmoid = int(sigmoid)
            sentence = int(sentence)
        except ValueError:
            raise SyntaxError(raw_string)

        if direction == 'f':
            backwards = False
        elif direction == 'b':
            backwards = True
        else:
            raise SyntaxError(raw_string)

        return cls(corpus, backwards, sigmoid, sentence, fold, epoch)

    def __init__(self, corpus, backwards, sigmoid, sentence, fold, epoch=1):
        self.corpus = corpus
        self.backwards = backwards
        self.sigmoid = sigmoid
        self.sentence = sentence
        self.fold = fold
        self.epoch = epoch

    def next_epoch(self):
        return type(self)(self.corpus, self.backwards, self.sigmoid,
                          self.sentence, self.fold, self.epoch + 1)

    def __repr__(self):
        return 'ModelRecipe.from_string({!r})'.format(self.filename)

    @property
    def d(self):
        return 'b' if self.backwards else 'f'

    @property
    def filename(self):
        return (
            '{s.corpus}-{s.d}-{s.sigmoid}-{s.sentence}.{s.fold}.{s.epoch}.h5'
        ).format(s=self)

    def create_batches(self, corpus, batch_size):
        """
        Returns iterables for training and evaluation batches, respectively.
        """
        # Get the tokens from the 9 training folds.
        training_batches = LoopBatchesEndlessly\
            .for_training(corpus, self.fold,
                          batch_size=batch_size,
                          sentence_length=self.sentence,
                          backwards=self.backwards)
        # Get the tokens from the leftover fold (evaluation).
        eval_batches = LoopBatchesEndlessly\
            .for_evaluation(corpus, self.fold,
                          batch_size=batch_size,
                          sentence_length=self.sentence,
                          backwards=self.backwards)
        return training_batches, eval_batches

    def create_model(self):
        """
        Creates and compiles the Keras model.
        """
        # Defining the model:
        model = Sequential()
        model.add(LSTM(self.sigmoid,
                       input_shape=(self.sentence, len(vocabulary))))
        model.add(Dense(len(vocabulary)))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.001),
                      metrics=['categorical_accuracy'])
        return model

    def create_model_and_load_weights(self, weights_filename):
        model = self.create_model()
        model.load_weights(weights_filename)
        return model


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

    # Load these here, because normally, they take forever...
    if args.action is not when_none_selected:
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        from keras.layers import LSTM
        from keras.optimizers import RMSprop

        from training_utils import LoopBatchesEndlessly, Sentences
        from condensed_corpus import CondensedCorpus

    args.action(**vars(args))
