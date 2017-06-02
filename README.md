miner
=====

Does some ad hoc source code mining using the GitHub API.

Requirements
------------

 - Python 3.6+
 - Redis 3.x
 - GNU shuf (macOS: `brew install coreutils`)
 - GNU parallel (macOS `brew install parallel`; Ubuntu: `apt install parallel`)

Install
-------

You must create a GitHub OAuth token and save it as `.token` in the
repository root.

Create a `virtualenv` (optional), then:

    pip install -r requirements.txt
    pip install -e .

Usage
-----

Run `redis-server` on localhost on the default port; then consult with
the documentation in the scripts in `bin/`.

License
-------

Copyright © 2016, 2017 Eddie Antonio Santos. Apache 2.0 licensed.
