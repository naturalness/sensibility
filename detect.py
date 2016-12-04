#!/usr/bin/env python
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

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from keras.models import model_from_json
from blessings import Terminal

from unvocabularize import unvocabularize
from vectorize_tokens import vectorize_tokens
from corpus import Token
from vocabulary import vocabulary
from training_utils import Sentences, one_hot_batch



THIS_DIRECTORY = Path(__file__).parent
TOKENIZE_JS_BIN = ('node', str(THIS_DIRECTORY / 'tokenize-js'))

SENTENCE_LENGTH = 20
PREFIX_LENGTH = SENTENCE_LENGTH - 1

parser = argparse.ArgumentParser()
parser.add_argument('filename', nargs='?', type=Path,
                    default=Path('/dev/stdin'))
parser.add_argument('--architecture', type=Path,
                    default=THIS_DIRECTORY / 'model-architecture.json')
parser.add_argument('--weights-forwards', type=Path,
                    default=THIS_DIRECTORY / 'javascript-tiny.5.h5')


def tokenize_file(file_obj):
    """
    >>> import tempfile
    >>> with tempfile.TemporaryFile('w+t', encoding='utf-8') as f:
    ...     f.write('$("hello");')
    ...     f.seek(0)
    ...     tokens = tokenize_file(f)
    11
    0
    >>> len(tokens)
    5
    >>> isinstance(tokens[0], Token)
    True
    """
    status = subprocess.run(TOKENIZE_JS_BIN,
                            check=True,
                            stdin=file_obj,
                            stdout=subprocess.PIPE)
    return [
        Token.from_json(raw_token)
        for raw_token in json.loads(status.stdout.decode('UTF-8'))
    ]


class Model:
    """
    >>> model = Model.from_filenames(architecture='model-architecture.json',
    ...                              weights='javascript-tiny.5.h5')
    >>> comma = vocabulary.to_index(',')
    >>> answer = model.predict([comma] * 19)
    >>> len(answer) == len(vocabulary)
    True
    >>> answer[comma] > 0.5
    True
    """
    def __init__(self, model):
        self.model = model

    def predict(self, vector):
        """
        TODO: Create predict() for entire file as a batch?
        """
        x, y = one_hot_batch([(vector, 0)], batch_size=1,
                             sentence_length=SENTENCE_LENGTH)
        return self.model.predict(x, batch_size=1)[0]

    @classmethod
    def from_filenames(cls, *, architecture=None, weights=None):
        with open(architecture) as archfile:
            model = model_from_json(archfile.read())
        model.load_weights(weights)

        return cls(model)


def rank(predictions):
    return list(sorted(enumerate(predictions),
                       key=lambda t: t[1], reverse=True))

def mean_reciprocal_rank(ranks):
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def print_top_5(model, file_vector):
    t = Terminal()
    header = "For {t.underline}{sentence_text}{t.normal}, got:"
    ranking_line = "   {prob:6.2f}% → {color}{text}{t.normal}"
    actual_line = "{t.red}Actual{t.normal}: {t.bold}{actual_text}{t.normal}"

    ranks = []

    for sentence, actual in Sentences(file_vector, size=SENTENCE_LENGTH):
        predictions = model.predict(sentence)
        paired_rankings = rank(predictions)
        ranked_vocab = list(tuple(zip(*paired_rankings))[0])
        top_5 = paired_rankings[:5]
        top_5_words = ranked_vocab[:5]

        sentence_text = unvocabularize(sentence)
        print(header.format_map(locals()))


        for token_id, weight in top_5:
            color = t.green if token_id == actual else ''
            if token_id == actual:
                found_it = True
            text = vocabulary.to_text(token_id)
            prob = weight * 100.0
            print(ranking_line.format_map(locals()))

        if actual not in top_5_words:
            actual_text = vocabulary.to_text(actual)
            print(actual_line.format_map(locals()))

        ranks.append(ranked_vocab.index(actual) + 1)

        print()

    print("MRR: ", mean_reciprocal_rank(ranks))
    print("Lowest rank", max(ranks))

# zip the three streams!
# do a element-wise multiplication on the probabilities
# rank based on highest probability.
# cases is it an addition or deletion or substitution or transposition
# transfusions are easy:
#   forwards thinks it's one over;
#   backwards thinks it's one over.

if __name__ == '__main__':
    globals().update(vars(parser.parse_args()))

    assert architecture.exists()
    assert weights_forwards.exists()

    with open(str(filename), 'rt', encoding='UTF-8') as script:
        tokens = tokenize_file(script)

    file_vector = vectorize_tokens(tokens)
    forwards = Model.from_filenames(architecture=str(architecture),
                                    weights=str(weights_forwards))

    print_top_5(forwards, file_vector)
