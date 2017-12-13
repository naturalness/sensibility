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

import logging
import sys
from math import exp, log
from operator import attrgetter
from typing import NamedTuple, Sequence, Tuple

import jenkspy  # type: ignore
import numpy as np

from sensibility.language import SourceSummary, language


class SummaryWithHash:
    __slots__ = 'filehash', 'n_tokens', 'sloc'

    def __init__(self, filehash: str, summary: SourceSummary) -> None:
        self.filehash = filehash
        if summary.n_tokens == 0:
            raise ValueError(f"Not enough tokens: {filehash}")
        assert summary.sloc > 0
        self.n_tokens = summary.n_tokens
        self.sloc = summary.sloc

    @property
    def ratio(self) -> float:
        return self.n_tokens / self.sloc


def dump(breakpoint: float, files: Sequence[SummaryWithHash]) -> None:
    for sf in files:
        label = 'gen' if sf.ratio > breakpoint else 'hw'
        print(f"{label},{sf.ratio:.1f},{sf.filehash}")


def find_break_point(files: Sequence[SummaryWithHash]) -> float:
    logger = logging.getLogger(__name__)

    # Prepare the data for jenks natural break algorithm
    logger.debug("Creating data structure")
    xs: np.ndarray[float] = np.zeros((len(files), 1), np.float64)
    for i, source in enumerate(files):
        # Log transform the ratio.
        xs[i] = log(source.ratio)

    logger.debug("Computing break")
    breaks: Tuple[float, ...] = jenkspy.jenks_breaks(xs, nb_class=2)
    start, break_point, end = (exp(p) for p in breaks)
    logger.info(f'# {break_point:.1f} [{start}, {end}]')

    return break_point
