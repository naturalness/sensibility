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


"""
Takes a vector representing a file with the vocabulary and spits out a
(hopefully) syntactically valid replica of said file.

Pipe this into a JavaScript formatter like js-beautify.

e.g.,

 $ npm install -g js-beautify
 $ ./unvocabularize.py corpus.sqlite3 1 | js-beautify
 /*<start>*/
 var $anyIdentifier = $anyIdentifier("any-string");
 $anyIdentifier().$anyIdentifier(function ($anyIdentifier) {
    $anyIdentifier.$anyIdentifier(`template-head${$anyIdentifier}template-tail`);
 }); /*<end>*/
"""

from vocabulary import vocabulary


def unvocabularize(vector):
    """
    >>> unvocabularize((0, 86, 5, 31, 99))
    '/*<start>*/ var $anyIdentifier ; /*<end>*/'
    """

    return ' '.join(vocabulary.to_text(element) for element in vector)


if __name__ == '__main__':
    import sys
    from condensed_corpus import CondensedCorpus
    _, filename, query = sys.argv

    # Attempt to convert the thing into an index.
    try:
        query = int(query)
    except ValueError:
        pass

    corpus = CondensedCorpus.connect_to(filename)
    file_hash, vector = corpus[query]
    print(unvocabularize(vector))
