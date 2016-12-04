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
import sys
import tempfile

from pathlib import Path
from itertools import islice
from collections import namedtuple
from functools import total_ordering

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
CHECK_SYNTAX_BIN = (*TOKENIZE_JS_BIN, '--check-syntax')

SENTENCE_LENGTH = 20
PREFIX_LENGTH = SENTENCE_LENGTH - 1


Common = namedtuple('Common',
                    'forwards_model backwards_model file_vector tokens')


@total_ordering
class Agreement(namedtuple('BaseAgreement', 'probability index')):
    def __lt__(self, other):
        return self.probability < other.probability

    def __eq__(self, other):
        return self.probability == other.probability

    def __rmatmul__(self, other):
        return other[self.index]

    def prefix(self, other, k=5):
        i = self.index
        return other[i - k:i]

    def suffix(self, other, k=5):
        i = self.index
        return other[i + 1:i + 1 +k]


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
    def __init__(self, model, backwards=False):
        self.model = model
        self.backwards = backwards

    @property
    def forwards(self):
        return not self.backwards

    def predict(self, vector):
        """
        TODO: Create predict() for entire file as a batch?
        """
        x, y = one_hot_batch([(vector, 0)], batch_size=1,
                             sentence_length=SENTENCE_LENGTH)
        return self.model.predict(x, batch_size=1)[0]

    @classmethod
    def from_filenames(cls, *, architecture=None, weights=None, **kwargs):
        with open(architecture) as archfile:
            model = model_from_json(archfile.read())
        model.load_weights(weights)

        return cls(model, **kwargs)


def check_syntax(source):
    """
    >>> check_syntax('function name() {}')
    True
    >>> check_syntax('function name() }')
    False
    """
    with tempfile.TemporaryFile('w+t', encoding='utf-8') as source_file:
        source_file.write(source)
        source_file.seek(0)
        status = subprocess.run(CHECK_SYNTAX_BIN, stdin=source_file)
    return status.returncode == 0


def tokenize_file(file_obj):
    """
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


def rank(predictions):
    return list(sorted(enumerate(predictions),
                       key=lambda t: t[1], reverse=True))


def mean_reciprocal_rank(ranks):
    return sum(1.0 / rank for rank in ranks) / len(ranks)


def common_args(*, filename=None,
                architecture=None,
                weights_forwards=None, weights_backwards=None,
                **kwargs):
    assert architecture.exists()
    assert weights_forwards.exists()

    with open(str(filename), 'rt', encoding='UTF-8') as script:
        tokens = tokenize_file(script)

    file_vector = vectorize_tokens(tokens)
    forwards_model = Model.from_filenames(architecture=str(architecture),
                                          weights=str(weights_forwards),
                                          backwards=False)
    backwards_model = Model.from_filenames(architecture=str(architecture),
                                           weights=str(weights_backwards),
                                           backwards=True)

    return Common(forwards_model, backwards_model, file_vector, tokens)


def top_5(*, forwards=None, **kwargs):
    common = common_args(**kwargs)
    model = common.forwards_model if forwards else common.backwards_model
    print_top_5(model, common.file_vector)


def chop_prefix(sequence, prefix=SENTENCE_LENGTH):
    return islice(sequence, prefix, len(sequence))


def index_of_max(seq):
    return max(enumerate(seq), key=lambda t: t[1])[0]


def combined(**kwargs):
    """
    cases: an addition or deletion or substitution or transposition
    transpositions are easy:
      forwards thinks it's one over;
      backwards thinks it's one over.
    """
    common = common_args(**kwargs)

    t = Terminal()
    ranking_line = "   {prob:6.2f}% → {color}{text}{t.normal}"
    actual_line = "{t.red}Actual{t.normal}: {t.bold}{actual_text}{t.normal}"

    sent_forwards = Sentences(common.file_vector,
                              size=SENTENCE_LENGTH,
                              backwards=False)
    sent_backwards = Sentences(common.file_vector,
                               size=SENTENCE_LENGTH,
                               backwards=True)

    least_agreement = []
    forwards_predictions = []
    backwards_predictions = []
    ranks = []

    contexts = enumerate(zip(chop_prefix(common.tokens, PREFIX_LENGTH),
                             sent_forwards, chop_prefix(sent_backwards)))

    # Note, the index is offset from the true start; i.e., when
    # index == 0, the true index is SENTENCE_LENGTH
    for index, (token, (prefix, x1), (suffix, x2)) in contexts:
        assert x1 == x2
        actual = x1
        print(unvocabularize(prefix[-5:]),
              t.bold_underline(token.value),
              unvocabularize(suffix[:5]))

        prefix_pred = common.forwards_model.predict(prefix)
        suffix_pred = common.backwards_model.predict(suffix)

        # harmonic mean
        mean = 2 * (prefix_pred * suffix_pred) / (prefix_pred + suffix_pred)

        forwards_predictions.append(index_of_max(prefix_pred))
        backwards_predictions.append(index_of_max(suffix_pred))

        paired_rankings = rank(mean)
        ranked_vocab = list(tuple(zip(*paired_rankings))[0])
        top_5 = paired_rankings[:5]
        top_5_words = ranked_vocab[:5]

        for token_id, weight in top_5:
            color = t.green if token_id == actual else ''
            text = vocabulary.to_text(token_id)
            prob = weight * 100.0
            print(ranking_line.format_map(locals()))

        ranks.append(ranked_vocab.index(actual) + 1)
        min_token_id, min_prob = paired_rankings[0]
        least_agreement.append(Agreement(min_prob, index))

        if actual not in top_5_words:
            actual_text = vocabulary.to_text(actual)
            print(actual_line.format_map(locals()))

        print()

        if not ranks:
            print(t.red("Could not analyze file!"), file=sys.stderr)
            return

    print("MRR: ", mean_reciprocal_rank(ranks))
    print("Lowest rank:", max(ranks))
    print("Time at #1: {:.2f}%".format(
          100 * sum(1 for rank in ranks if rank == 1) / len(ranks)))
    print()

    forwards_text = [vocabulary.to_text(num) for num in forwards_predictions]
    backwards_text = [vocabulary.to_text(num) for num in backwards_predictions]

    least_agreement.sort()
    # Compensate for offset indices
    file_vector = common.file_vector[SENTENCE_LENGTH:]
    tokens_text = [tok.value for tok in common.tokens[PREFIX_LENGTH:]]
    for disagreement in least_agreement[:5]:
        print(disagreement.probability)
        prefix = ' '.join(disagreement.prefix(tokens_text))
        suffix = ' '.join(disagreement.suffix(tokens_text))

        print("   ", prefix, t.yellow(forwards_text @ disagreement), suffix)
        print("   ", prefix, t.underline(tokens_text @ disagreement), suffix)
        print("   ", prefix, t.blue(backwards_text @ disagreement), suffix)
        print()


def print_top_5(model, file_vector):
    t = Terminal()
    header = "For {t.underline}{sentence_text}{t.normal}, got:"
    ranking_line = "   {prob:6.2f}% → {color}{text}{t.normal}"
    actual_line = "{t.red}Actual{t.normal}: {t.bold}{actual_text}{t.normal}"

    ranks = []
    sentences = Sentences(file_vector, size=SENTENCE_LENGTH,
                          backwards=model.backwards)

    for sentence, actual in sentences:
        predictions = model.predict(sentence)
        paired_rankings = rank(predictions)
        ranked_vocab = list(tuple(zip(*paired_rankings))[0])
        top_5 = paired_rankings[:5]
        top_5_words = ranked_vocab[:5]

        sentence_text = unvocabularize(sentence)
        print(header.format_map(locals()))

        for token_id, weight in top_5:
            color = t.green if token_id == actual else ''
            text = vocabulary.to_text(token_id)
            prob = weight * 100.0
            print(ranking_line.format_map(locals()))

        if actual not in top_5_words:
            actual_text = vocabulary.to_text(actual)
            print(actual_line.format_map(locals()))

        ranks.append(ranked_vocab.index(actual) + 1)

        print()

    if not ranks:
        print(t.red("Could not analyze file!"), file=sys.stderr)
        return

    print("MRR: ", mean_reciprocal_rank(ranks))
    print("Lowest rank:", max(ranks))
    print("Time at #1: {:.2f}%".format(
        100 * sum(1 for rank in ranks if rank == 1) / len(ranks)
    ))


def add_common_args(parser):
    parser.add_argument('filename', nargs='?', type=Path,
                        default=Path('/dev/stdin'))
    parser.add_argument('--architecture', type=Path,
                        default=THIS_DIRECTORY / 'model-architecture.json')
    parser.add_argument('--weights-forwards', type=Path,
                        default=THIS_DIRECTORY / 'javascript-tiny.5.h5')
    parser.add_argument('--weights-backwards', type=Path,
                        default=THIS_DIRECTORY /
                        'javascript-tiny.backwards.5.h5')


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title='subcommands',
                                   description='valid subcommands')

top_5_parser = subparsers.add_parser('top-5')
group = top_5_parser.add_mutually_exclusive_group()
group.add_argument('--forwards', action='store_true', default=True)
group.add_argument('--backwards', dest='forwards', action='store_false')
add_common_args(top_5_parser)
top_5_parser.set_defaults(func=top_5)

combined_parser = subparsers.add_parser('combined')
add_common_args(combined_parser)
combined_parser.set_defaults(func=combined)

parser.set_defaults(func=None)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.func:
        args.func(**vars(args))
    else:
        parser.print_usage()
        exit(-1)
