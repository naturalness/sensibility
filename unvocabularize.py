#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Takes a vector representing a file with the vocabulary and spits out a
(hopefully) syntactically valid replica of said file. 
"""

from vocabulary import vocabulary

def unvocabularize(vector):
    """
    >>> unvocabularize((0, 86, 5, 31, 99))
    '/*<start>*/ var $anyIdentifier ; /*<end>*/'
    """
    
    def generate():
        for element in vector:
            yield vocabulary.to_text(element)
    return ' '.join(generate())


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
