miner
=====

Does some ad hoc source code mining using the GitHub API.

Requirements
------------

 - Python 3.5+
 - Redis 3.x

Install
-------

You must create a GitHub OAuth token and save it as `.token` in the
repository root.

Usage
-----

Run `redis-server` on localhost on the default port; then consult with
the documentation in the loose Python files. :/

To run files in `bin/` you must have `sensibility` installed locally:

	pip install -e .

License
-------

Copyright © 2016 Eddie Antonio Santos. Apache 2.0 licensed.
