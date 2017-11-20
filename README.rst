***********
Sensibility
***********

.. image:: https://travis-ci.org/naturalness/sensibility.svg?branch=master
    :target: https://travis-ci.org/naturalness/sensibility

Finds and fixes syntax errors. In development.

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

For fixing Java files:

* Java 8 SE

For fixing JavaScript files:

* Node.JS >= 6.0 and ZeroMQ_
  (macOS: ``brew install zeromq``; Ubuntu: ``apt install libzmq-dev``)

Researchers that wish to mine more repositories or languages will require:

* A running Redis_ server.

Researchers wishing to evaluate the current methods will require:

* UnnaturalCode_

.. _Redis: https://redis.io/
.. _ZeroMQ: http://zeromq.org/
.. _UnnaturalCode: https://github.com/naturalness/unnaturalcode/tree/eddie-eval


Install
=======

Activate a virtualenv, if that's your thing. Then,

::

    pip install -e .

Usage
-----

Once installed, there's one entry point to all the scripts and utilities included::

   sensibility <SUBCOMMAND>

Most subcommands require the specification of a language,
with the ``-l <LANGUAGE>`` option before the subcommand.

For example, to train Java models::

   sensibility -l java train-list --help


Development
===========

The following diagram visualizes the data flow.
Subcommands to ``sensibility`` are in black rectangles; white ovals are products.
Please contact the author to obtain the mistake database,
which is only applicable when evaluating the Java models.

.. image:: https://raw.githubusercontent.com/naturalness/sensibility/master/docs/dependencies.png
    :width: 100%
    :align: center


Tests
-----

To run the tests, install tox_ using Pip, then run tox.

.. _tox: https://tox.readthedocs.io/en/latest/


Mining repositories
-------------------

1. You must create a GitHub OAuth token and save it as ``.token`` in the
   repository root.
2. Run ``redis-server`` on localhost on the default port.
3. Use ``sensibility mine find-repos`` to get a list of the top ~10k repos::

    sensibility mine find-repos javascript | sort -u > javascript-repos.txt

4. Use ``bin/enqueue-repo`` to enqueue repos to download::

    sensibility mine enqueue-repo < javascript-repos.txt

5. Start one or more downloaders. These will dequeue a repo from the running Redis server and download sources::

    sensibility mine download


Evaluation
----------

Type ``make experiments`` to train all of the models and evaluate each one.
See ``libexec/experiments`` for more details.


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
