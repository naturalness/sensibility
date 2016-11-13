#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Creates matrix-corpus.sqlite3 from a full corpus stored as an SQLite database.

The file contains one-hot encoded matrices.
"""

import logging

import io
import sys
import sqlite3

import numpy as np
from tqdm import tqdm

from corpus import Corpus
from vectorize_tokens import vectorize_tokens
from vocabulary import vocabulary

DESTINATION = 'matrix-corpus.sqlite3'
SCHEMA = """
CREATE TABLE IF NOT EXISTS source_matrix(
    hash TEXT PRIMARY KEY,
    np_array BLOB NOT NULL,     -- the numpy array, as a blob.
    n_tokens INTEGER NOT NULL   -- the amount of tokens, (excluding start/end)
);
"""


def insert(conn, hash_, tokens, n_vocab=len(vocabulary)):
    dimensions = (2 + len(tokens), n_vocab)
    array = np.zeros(dimensions, dtype=np.bool_)
    for t, index in enumerate(vectorize_tokens(tokens)):
        array[t, index] = 1

    filelike = io.BytesIO()
    np.save(filelike, array)

    conn.execute("""
        INSERT INTO source_matrix(hash, np_array, n_tokens)
             VALUES (?, ?, ?)
     """, (hash_, filelike.getbuffer(), len(tokens)))
    conn.commit()


def main():
    _, filename = sys.argv

    corpus = Corpus.connect_to(filename)

    # Create the schema
    conn = sqlite3.connect(DESTINATION)
    conn.executescript(SCHEMA)
    conn.commit()

    files = corpus.iterate(with_hash=True, skip_empty=True)
    for i, result in enumerate(tqdm(files, total=len(corpus))):
        hash_, tokens = result
        insert(conn, hash_, tokens)


if __name__ == '__main__':
    main()
