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


class ModelRecipe:
    """
    All the data of an existing model.
    >>> m = ModelRecipe.from_string('javascript-f-300-20.4.3.h5')
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

    def __eq__(self, other):
        return (
            isinstance(other, ModelRecipe) and
            self.corpus == other.corpus and
            self.backwards == other.backwards and
            self.sigmoid == other.sigmoid and
            self.sentence == other.sentence and
            self.fold == other.fold and
            self.epoch == other.epoch
        )

    def __repr__(self):
        return 'ModelRecipe.from_string({!r})'.format(self.filename)

    @property
    def d(self):
        return 'b' if self.backwards else 'f'

    def flipped(self):
        """
        Returns a model recipe with the opposite direction.

        >>> m = ModelRecipe.from_string('javascript-f-300-20.4.3.h5')
        >>> m.flipped().filename
        'javascript-b-300-20.4.3.h5'
        >>> m.flipped().flipped() == m
        True
        """
        return type(self)(self.corpus, not self.backwards, self.sigmoid,
                          self.sentence, self.fold, self.epoch)

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
