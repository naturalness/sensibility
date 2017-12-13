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
Paths for internal use.
"""

import warnings
from pathlib import Path

from sensibility.language import language

# Get paths for here and repository root dir.
HERE = Path(__file__).parent
REPOSITORY_ROOT = HERE.parent
assert (REPOSITORY_ROOT / '.git').exists()

# Directories for storing data and models.
DATA_DIR = REPOSITORY_ROOT / 'data'
MODEL_DIR = REPOSITORY_ROOT / 'models'

EVALUATION_DIR = REPOSITORY_ROOT / 'evaluation'


def get_evaluation_dir() -> Path:
    return EVALUATION_DIR / language.id


def get_sources_path() -> Path:
    return get_evaluation_dir() / 'sources.sqlite3'


def get_vectors_path() -> Path:
    return get_evaluation_dir() / 'vectors.sqlite3'


def get_partitions_path() -> Path:
    return get_evaluation_dir() / 'partitions'


def get_validation_set_path(partition: int) -> Path:
    return get_partitions_path() / str(partition) / 'validation'


def get_training_set_path(partition: int) -> Path:
    return get_partitions_path() / str(partition) / 'training'


def get_test_set_path(partition: int) -> Path:
    return get_partitions_path() / str(partition) / 'test'


def get_mistakes_path() -> Path:
    return get_evaluation_dir() / 'mistakes.sqlite3'


def get_lstm_path(direction: str, partition: int) -> Path:
    warnings.warn(f"Use models in {REPOSITORY_ROOT/'models'!s}", DeprecationWarning)
    return get_evaluation_dir() / 'models' / f'{language.id}-{direction}{partition}.hdf5'


def get_cache_path() -> Path:
    warnings.warn(f"Don't ever use this again", DeprecationWarning)
    return get_evaluation_dir() / 'models' / f'{language.id}-predictions.sqlite3'
