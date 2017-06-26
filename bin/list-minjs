#!/bin/sh

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

# Prints the filehash and name of JavaScript sources suffixed with:
#   .min.js or -min.js
#
# Usage:
#       list-minjs > minjs.csv

set -e

sqlite3 -separator "$(printf "\t")" ./evaluation/javascript-sources.sqlite3 <<'SQL'
    SELECT hash, path FROM repository_source
     WHERE lower(path) GLOB '*[.-]min.js';
SQL
