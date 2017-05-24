#!/bin/bash

# Prints hashes of files that have been downloaded, but have not been parsed.

DATABASE=${1:-python-sources.sqlite3}
QUERY=$(cat <<-SQL
SELECT hash FROM source_file
EXCEPT SELECT hash from source_summary
EXCEPT SELECT hash FROM failure
SQL
)

sqlite3 "$DATABASE" "$QUERY"
