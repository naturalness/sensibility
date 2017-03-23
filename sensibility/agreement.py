#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import warnings
from typing import Any
from functools import total_ordering


@total_ordering
class Agreement:
    ___slots__ = 'probability', 'index'

    def __init__(self, probability, index):
        self.probability = probability
        self.index = index

    def __lt__(self, other: 'Agreement') -> bool:
        return self.probability < other.probability

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Agreement):
            return self.probability == other.probability
        return False

    def __rmatmul__(self, other):
        warnings.warn("Using untyped method: Any @ Agreement")
        return other[self.index]

    def prefix(self, other, k=5):
        warnings.warn("Using untyped method: Agreement.prefix(Any)")
        i = self.index
        return other[i - k:i]

    def suffix(self, other, k=5):
        warnings.warn("Using untyped method: Agreement.suffix(Any)")
        i = self.index
        return other[i + 1:i + 1 +k]
