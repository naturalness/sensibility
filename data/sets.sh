#!/bin/bash

set -eux

get_fold() {
    local fold="$1"
    sqlite3 javascript-vectors.sqlite3 "SELECT hash FROM fold_assignment WHERE fold = ${fold}"
}

get_train() {
    for i in {0..4} ; do
        get_fold "$i" > "joshua/train.$i"
    done
}

get_validate() {
    for i in {5..9} ; do
        local fold="$((i - 5))"
        get_fold "$i" > "joshua/validate.$fold"
    done
}

get_validate
