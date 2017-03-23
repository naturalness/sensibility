#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sqlite3
import functools
import array
from pathlib import Path
from typing import Sequence, Tuple, Union, TextIO, Iterable, cast

import numpy as np

from .model import Model
from ._paths import PREDICTIONS_PATH, MODEL_DIR
from .sentences import Sentence, T, forward_sentences, backward_sentences
from .tokenize_js import tokenize_file
from .vocabulary import Vind
from .vectorize_tokens import vectorize_tokens


# A type that neatly summarizes the double contexts.
Contexts = Iterable[Tuple[Sentence[Vind], Sentence[Vind]]]


class Predictions:
    """
    Stores predictions.
    """

    SCHEMA = r"""
    PRAGMA encoding = "UTF-8";

    CREATE TABLE IF NOT EXISTS prediction (
        model   TEXT NOT NULL,      -- model that created the prediction
        context BLOB NOT NULL,      -- input of the prediction, as an array
        vector  BLOB NOT NULL,      -- prediction data, as an array

        PRIMARY KEY (model, context)
    );
    """

    def __init__(self, fold: int, filename: Path=PREDICTIONS_PATH) -> None:
        assert 0 <= fold < 5
        forwards_path = MODEL_DIR / f"javascript-f{fold}.hdf5"
        backwards_path = MODEL_DIR / f"javascript-b{fold}.hdf5"
        self.forwards_model = Model.from_filename(forwards_path)
        self.backwards_model = Model.from_filename(backwards_path,
                                                   backwards=True)
        # XXX: Hard code the context length!
        self.context_length = 20
        self._conn = self._connect(filename)

        forwards = f'f{fold}'
        backwards = f'b{fold}'

        def _predict(name: str, model: Model,
                     tuple_context: Sequence[Vind]) -> array.array:
            """
            Does prediction, consulting the database first before consulting
            the model.
            """
            context = array.array('B', tuple_context).tobytes()
            try:
                return self.get_prediction(name, context)
            except KeyError:
                prediction = model.predict(tuple_context)
                self.add_prediction(name, context, prediction)
                return prediction

        # > SELECT MAX(n_tokens) FROM vectorized_source;
        # 1809948
        # ceil(log(1809948, 2)) -- this will be like... 2 GiB PER CACHE.
        cache_size = 2**21

        # Create prediction functions with in-memory caching.
        @functools.lru_cache(maxsize=cache_size)
        def predict_forwards(prefix: Tuple[Vind, ...]) -> array.array:
            return _predict(forwards, self.forwards_model, prefix)

        @functools.lru_cache(maxsize=cache_size)
        def predict_backwards(suffix: Tuple[Vind, ...]) -> array.array:
            return _predict(backwards, self.backwards_model, suffix)

        self.predict_forwards = predict_forwards
        self.predict_backwards = predict_backwards

    def predict(self, filename: Union[Path, str]) -> None:
        """
        Predicts at each position in the file, writing predictions to
        database.
        """

        # Get file vector for this (incorrect) file.
        with open(str(filename), 'rt', encoding='UTF-8') as script:
            tokens = tokenize_file(cast(TextIO, script))
        file_vector = vectorize_tokens(tokens)

        # Calculate and store predictions.
        for (prefix, _), (suffix, _) in self.contexts(file_vector):
            self.predict_forwards(tuple(prefix))
            self.predict_backwards(tuple(suffix))

    def clear_cache(self):
        self.predict_forwards.cache_clear()
        self.predict_backwards.cache_clear()

    def contexts(self, file_vector: Sequence[Vind]) -> Contexts:
        """
        Yield every context (prefix, suffix) in the given file vector.
        """
        sent_forwards = forward_sentences(file_vector,
                                          context=self.context_length)
        sent_backwards = backward_sentences(file_vector,
                                            context=self.context_length)
        return zip(sent_forwards, sent_backwards)

    def add_prediction(self, name: str, context: bytes,
                       prediction: np.ndarray) -> None:
        """
        Add the prediction (model, context) -> prediction
        """
        assert self._conn

        vector: bytes = array.array('f', prediction).tobytes()
        with self._conn:
            self._conn.execute('BEGIN')
            self._conn.execute(r'''
                INSERT INTO prediction(model, context, vector)
                VALUES (:model, :context, :vector)
            ''', dict(model=name, context=context, vector=vector))

    def get_prediction(self, name: str, context: bytes) -> array.array:
        """
        Try to fetch the prediction from the database.

        Returns None if the entry is not found.
        """
        assert self._conn
        cur = self._conn.execute(r'''
            SELECT vector
            FROM prediction
            WHERE model = :model AND context = :context
        ''', dict(model=name, context=context))
        result = cur.fetchall()

        if not result:
            # Prediction not found!
            raise KeyError(f'{name}/{context!r}')
        else:
            # Return the precomputed prediction.
            output = array.array('f')
            output.frombytes(result[0][0])
            return output

    def _connect(self, filename: Path) -> sqlite3.Connection:
        # Connect to the database
        conn = sqlite3.connect(str(filename))
        # Initialize the database.
        with conn:
            conn.executescript(self.SCHEMA)
            # Some speed optimizations:
            # http://codificar.com.br/blog/sqlite-optimization-faq/
            conn.executescript(r'''
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = normal;
            ''')
        return conn
