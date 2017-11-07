Evaluation
==========

Files need for and created by the empirical evaluation.

Directory structure:

    .
    └── {language}/
        ├── partitions/
        │   └── {partition-number}/
        │       ├── test
        │       ├── training
        │       └── validation
        ├── results/
        │   └── fix-(dual|left|right)-{configuration-id}.sqlite3
        ├── mistakes.sqlite3
        ├── sources.sqlite3
        └── vectors.sqlite3

Partition files
===============

The corpus is partitioned into several different sets (currently 5), for
the purposes of evaluating how consistent the results are across
different data.

For each partition, there is a directory corresponding to its number,
that contains three flat files: training, validation, and test. These
files are a set of file hashes (see sources), each separated by
newlines.

Results files
=============
