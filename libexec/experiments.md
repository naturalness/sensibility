So I'll have one script check the directory for models:

   make -j --keep-going java-experiments

This will:

  - for each training configuration, create a **modelset**.
  - for each modelset, do an empirical evaluation.

A modelset is a directory containing models in the following format:

   {dir}{partition}-{validation}.hdf5

There will be no symlinks, because that confuses me.

   {modelset}:
      train-lstm -o $@ --hidden-layers --context --

For the evaluation, create a result database:

   {results}: {modelset}
      evaluate $< -o $@

Concatenate the database:

    master.results: {*results}
       combine-results $< -o $@

Let the evaluation deal with ALL the different types of evaluations:

   - different models
   - different fixers

Or maybe... have a concatenation of configurations? And only use the
configurations required. For example, the models require LESS
configuration parameters than the evaluations

Finally, concatenate the database:
   master.results


Filenames:
   {config sha}.models/
      MANIFEST.json
      {direction}{partition}
   {config sha}.sqlite3

Configuration naming
--------------------

   Just a SHA-256 of the entire configuration

Manifest
--------

Dump the FULL configuration in a transpose TSV format.

Configurations
==============

Partitions
 ~ 0
 ~ 1
 ~ 2
 ~ 3
 ~ 4

Fix
 ~ forwards
 ~ backwards
 ~ duel
