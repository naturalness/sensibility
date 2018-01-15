***********
Sensibility
***********

.. image:: https://travis-ci.org/naturalness/sensibility.svg?branch=master
    :target: https://travis-ci.org/naturalness/sensibility

Finds and fixes syntax errors. This is experimental software.

    **NOTE**: Please use the saner2018_ tag if you're looking to replicate the
    results from our SANER 2018 paper. Find the replication data on `archive.org`_
    See Citation_ for how to cite this paper.

    ::

        git clone --branch=saner2018 https://github.com/eddieantonio/training-grammar-guru.git

.. _saner2018: https://github.com/naturalness/sensibility/tree/saner2018
.. _`archive.org`: https://archive.org/details/sensibility-saner2018


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


Citation
========

If you use Sensibility in academic works, please use the following citation:

    @inproceedings{santos2018, 
        author={Santos, Eddie Antonio and
                Campbell, Joshua Charles and
                Patel, Dhvani and
                Hindle, Abram and
                Amaral, Jos{\'e} Nelson}, 
        booktitle={2018 {IEEE} 25th International Conference on Software Analysis, Evolution and Reengineering ({SANER})}, 
        title={Syntax and {Sensibility}: Using Language Models to Detect and Correct Syntax Errors},
        year={2018}, 
        month={Mar}}

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
