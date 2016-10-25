miner
=====

Does some ad hoc source code mining using the GitHub API.


Requirements
------------

 - Python 3.5+
 - Redis 3.x
 - Node.JS 4+

Install
-------

You must create a GitHub OAuth token and save it as `.token` in the
repository root.

```sh
# For the NodeJS script (required for Python scripts!)
cd parse-js/
npm install

# For the Python scripts
virtualenv -p $(which python3) venv
source venv/bin/activate
pip install -r requirements.txt
```

Usage
-----

Run `redis-server` on localhost on the default port; then:

```sh
# Find big list of repos.
python search_worker.py
# Download repos, one-by-one. Can be launched multiple times.
python download_worker.py
# Parse files using Node.JS
python parse_worker.py
```

License
-------

Copyright © 2016 Eddie Antonio Santos. Apache 2.0 licensed.
