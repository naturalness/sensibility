#!/usr/bin/env python3
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
Compute natural break between "minified" and "hand-written" JavaScript.
"""

import sys
import csv
from math import log, exp
from operator import attrgetter
from typing import NamedTuple, List, Tuple

import numpy as np
import jenkspy  # type: ignore

from sensibility.language import language, SourceSummary
from sensibility.miner.corpus import Corpus


class SourceFile:
    __slots__ = 'filehash', 'n_tokens', 'sloc'

    def __init__(self, filehash: str, summary: SourceSummary) -> None:
        self.filehash = filehash
        self.n_tokens = summary.n_tokens
        self.sloc = summary.sloc

    @property
    def ratio(self):
        return self.ntokens / self.sloc


def dump(breakpoint: float, files: List[SourceFile]) -> None:
    for sf in files:
        label = 'gen' if sf.ratio > breakpoint else 'hw'
        print(f"{label},{sf.ratio:.1f},{sf.filehash}")


if __name__ == '__main__':
    # This only makes sense for JavaScript (I think).
    language.set_language('JavaScript')
    corpus = Corpus()
    files = list(SourceFile(filehash, summary)
                 for filehash, summary in corpus.source_summaries)

    # Prepare the data for jenks natural break algorithm
    xs: np.ndarray[float] = np.zeros((len(files), 1), np.float64)
    for i, source in enumerate(files):
        xs[i] = log(source.ratio)

    print("Computing breaks...", file=sys.stderr)
    breaks: Tuple[float, ...] = jenkspy.jenks_breaks(xs, nb_class=2)
    start, break_point, end = (exp(p) for p in breaks)

    print(f'# {break_point:.1f} [{start}, {end}]', file=sys.stderr)
    dump(break_point, files)
