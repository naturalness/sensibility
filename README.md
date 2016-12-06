Grammar Guru
============

Detects and fixes syntax errors. Currently works with JavaScript.

Install
-------

Requires Python 3.5.

Activate a virtualenv, if that's your thing. Then,

    pip install -r requirements.txt

Download the [model data] and copy the `*.h5` and `*.json` files into
the directory you'll run the tool.

[model data]: https://archive.org/details/lstm-javascript-tiny


Usage
-----

To suggest a fix for a file:

    $ ./detect.py suggest my-incorrect-file.js
    my-incorrect-file.js:1:1: try inserting a '{'
        if (name)
                  ^
                  {

To dump the model's token-by-token consensus about the file:

    $ ./detect.py dump my-incorrect-file.js


License
-------

Copyright 2016 Eddie Antonio Santos <easantos@ualberta.ca>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
