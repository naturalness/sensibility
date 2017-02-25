#!/bin/sh

head -n1 "$1"

for file in "$@"; do
    tail -n+2 "$file"
done
