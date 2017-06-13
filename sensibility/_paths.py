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

from sensibility.language import language
from pathlib import Path


# Get paths for here and repository root dir.
HERE = Path(__file__).parent
REPOSITORY_ROOT = HERE.parent
assert (REPOSITORY_ROOT / '.git').exists()

# Directories for storing data and models.
DATA_DIR = REPOSITORY_ROOT / 'data'
MODEL_DIR = REPOSITORY_ROOT / 'models'

EVALUATION_DIR = REPOSITORY_ROOT / 'evaluation'

# Paths to specific databases.
SOURCES_PATH = DATA_DIR / 'javascript-sources.sqlite3'
VECTORS_PATH = DATA_DIR / 'javascript-vectors.sqlite3'
MUTATIONS_PATH = DATA_DIR / 'javascript-mutations.sqlite3'
PREDICTIONS_PATH = DATA_DIR / 'javascript-predictions.sqlite3'

def get_partitions_path(language=language) -> Path:
    return EVALUATION_DIR / language.id / 'partitions'

def get_validation_set_path(partition: int, language=language) -> Path:
    return get_partitions_path(language) / str(partition) / 'validation'

def get_training_set_path(partition: int, language=language) -> Path:
    return get_partitions_path(language) / str(partition) / 'training'

def get_test_set_path(partition: int, language=language) -> Path:
    return get_partitions_path(language) / str(partition) / 'test'
