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


from itertools import islice


class Sentences:
    """
    Generates samples from the given vector.

    >>> from corpus import Token
    >>> from vocabulary import vocabulary
    >>> from vectorize_tokens import vectorize_tokens
    >>> t = Token(value='var', type='Keyword', loc=None)
    >>> v = vectorize_tokens([t])
    >>> sentences = Sentences(v, size=20)
    >>> len(sentences)
    0
    >>> list(iter(sentences))
    []

    >>> v = vectorize_tokens([t] * 19)
    >>> sentences = Sentences(v, size=20)
    >>> len(sentences)
    1
    >>> x, y = next(iter(sentences))
    >>> list(x) == [0] + [86] * 19
    True
    >>> y
    99
    """

    def __init__(self, vector, *, size=None):
        # TODO: Step?
        self.vector = vector
        self.size= size
        if not isinstance(size, int):
            raise ValueError("Size must be an int.")

    def __iter__(self):
        n_sentences = len(self)

        if n_sentences <= 0:
            raise StopIteration

        sentence_len = self.size
        token_vector = self.vector

        # Fill in the vectors.
        for sentence_id in range(n_sentences):
            start = sentence_id
            end = sentence_id + sentence_len
            assert end < len(token_vector), "not: %d < %d" %(end,
                                                        len(token_vector))
            sentence = islice(token_vector, start, end)

            yield sentence, token_vector[end]

    def __len__(self):
        """
        Returns how many sentences this will produce. Can be zero!
        """
        sentences_possible = len(self.vector) - self.size
        return at_least(0, sentences_possible)


class LoopSentencesEndlessly:
    def __init__(self, corpus_filename, folds):
        assert Path(corpus_filename).exists()
        self.filename = corpus_filename
        self.folds = folds
        self.corpus = None

    def __iter__(self):
        self.corpus = corpus = CondensedCorpus.connect_to(self.filename)
        while True:
            for fold in self.folds:
                for file_hash in corpus.hashes_in_fold(fold):
                    _, tokens = corpus[file_hash]
                    yield from Sentences(tokens)

    def __del__(self):
        return
        if self.corpus is not None:
            self.corpus.disconnect()

    @classmethod
    def for_training(cls, corpus_filename, fold):
        """
        Endlessly yields (X, Y) pairs from the corpus for training.
        """
        # XXX: hardcode a lot of stuff
        # Assume there are 10 folds.
        assert 0 <= fold <= 9
        training_folds = tuple(num for num in range(10) if num != fold)
        assert len(training_folds) == 9
        return cls(corpus_filename, training_folds)

    @classmethod
    def for_evaluation(cls, corpus_filename, fold):
        """
        Endlessly yields (X, Y) pairs from the corpus for evaluation.
        """
        # XXX: hardcode a lot of stuff
        # Assume there are 10 folds.
        assert 0 <= fold <= 9
        return cls(corpus_filename, (fold,))


def at_least(value, *args):
    """
    Returns **at least** the given number.

    For Dr. Hindle's sanity.
    """
    return max(value, *args)


def count_samples_slow(filename, fold):
    corpus = CondensedCorpus.connect_to(filename)
    folds = tuple(num for num in range(10) if num != fold)
    n_samples = 0
    for n in folds:
        for file_hash in corpus.hashes_in_fold(n):
            _, tokens = corpus[file_hash]
            n_samples += len(Sentences(tokens))
    return n_samples


def dumb_one_hot_stuff():
    vocab_size = len(vocabulary)

    # Create empty one-hot vectors
    x = np.zeros((n_sentences, sentence_len, vocab_size), dtype=np.bool)
    y = np.zeros((n_sentences, vocab_size), dtype=np.bool)

    # Fill in the vectors.
    for sentence_id in range(n_sentences):
        start = sentence_id
        end = sentence_id + sentence_len
        assert end < len(token_vector), "not: %d < %d" %(end,
                                                    len(token_vector))
        sentence = islice(token_vector, start, end)

        # Fill in the one-hot matrix for X
        for i, token_id in enumerate(sentence):
            x[sentence_id, i, token_id] = 1

        # Add the last token for the one-hot vector Y.
        last_token_id = token_vector[end]
        y[sentence_id, last_token_id] = 1

    yield x, y
