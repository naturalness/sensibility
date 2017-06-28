***********
Sensibility
***********

Finds and fixes syntax errors. Currently in development, and in less of
a working stage.

    **NOTE**: The cmput680_ tag has a working proof-of-concept. Clone the
    `old repo`_ at the ``cmput680`` tag:

    ::

        git clone --branch=cmput680 https://github.com/eddieantonio/training-grammar-guru.git

    Then following its install and usage sections.

.. _old repo: https://github.com/eddieantonio/training-grammar-guru
.. _cmput680: https://github.com/eddieantonio/training-grammar-guru/tree/cmput680


Requirements
============

Sensibility requires:

 * Python 3.6.
 * The JavaScript back-end requires Node.JS >= 6.0 and ZeroMQ_
   (macOS: ``brew install zeromq``; Ubuntu: ``apt install libzmq-dev``)

Researchers that wish to mine more repositories or languages will require:

 * A running Redis_ server.

Researchers wishing to evaluate the current methods will require:

 * GNU shuf (macOS: `brew install coreutils`)
 * GNU parallel (macOS `brew install parallel`; Ubuntu: `apt install parallel`)

.. _Redis: https://redis.io/
.. _ZeroMQ: http://zeromq.org/


Install
=======

Activate a virtualenv, if that's your thing. Then,

::

    pip install -e .

> **TODO**: This section is out-dated! Consult the author.

Download the `model data`_ and copy the ``*.h5`` and ``*.json`` files into the
directory you'll run the tool.

.. _model data: https://archive.org/details/lstm-javascript-tiny


Usage
=====

> **TODO**: This section is out-dated! Consult the author.

To suggest a fix for a file:

::

    $ bin/sensibility my-incorrect-file.js
    my-incorrect-file.js:1:1: try inserting a '{'
        if (name)
                  ^
                  {

To dump the model's token-by-token consensus about the file:

::

    $ bin/sensibility --dump my-incorrect-file.js


Development
===========

To run the scripts in bin, do this::

    pip install -e .

To run the tests, install tox_ using Pip, then run tox.

.. _tox: https://tox.readthedocs.io/en/latest/


Mining repositories
-------------------

 1. You must create a GitHub OAuth token and save it as `.token` in the
    repository root.
 2. Run `redis-server` on localhost on the default port.
 3. Use `bin/get-repo-list` to get a list of the top ~10000 repos::

     bin/get-repo-list javascript | sort -u > javascript-repos.txt

 4. Use `bin/enqueue-repo` to enqueue repos to download::

     bin/enqueue-repo < javascript-repos.txt

 5. Start one or more downloaders. These will dequeue from Redis and download
    sources::

     bin/download

The following diagram the data flow, starting from the name of the language you
wish to train, all the way to the mutant evaluation.

.. image:: https://raw.githubusercontent.com/naturalness/sensibility/master/docs/dependencies.png
    :width: 100%
    :align: center


License
=======

Copyright 2016, 2017 Eddie Antonio Santos easantos@ualberta.ca

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
