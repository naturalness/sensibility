#!/bin/bash

# Wraps the invocation of index.js such that it automatically NPM installs if
# there are 
# Auto npm install before running index.js

original_dir=$(pwd)
script_dir=$(dirname "$0")

cd "$script_dir" || exit -1
if [[ node_modules -ot package.json ]]; then
    npm install > /dev/null
fi
cd "$original_dir" || exit -1
    
exec node "$script_dir/index.js" "$@"
