#!/bin/sh

# Concatenates a bunch of CSV files with headers.

head -n1 "$1"

for file in "$@"; do
    tail -n+2 "$file"
done
