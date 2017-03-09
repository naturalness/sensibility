#!/bin/bash

# An ad hoc file that gives some (maybe) imporant descriptive statistics.

minified_files() {
    grep -c 'minified' "$HOME/Documents/Log at 2017-02-15 09.27.31.txt"
}

sum_folds() {
    local query
    read -r -d '' query <<'SQL'
        SELECT fold, SUM(n_tokens)
          FROM vectorized_source JOIN fold_assignment USING (hash)
         GROUP BY fold;
SQL
    echo "> $query"
    sqlite3 javascript.sqlite3 "$query"
}

avg_folds() {
    local query
    read -r -d '' query <<SQL
        SELECT AVG(tokens) FROM (
            SELECT SUM(n_tokens) tokens
              FROM vectorized_source JOIN fold_assignment USING (hash)
             GROUP BY fold
        );
SQL

    echo "> $query"
    sqlite3 javascript.sqlite3 "$query"
}

minified_files
# 10767

sum_folds
# 0|10148370
# 1|11757743
# 2|10929836
# 3|10283616
# 4|10783852
# 5|14807363
# 6|10253839
# 7|10026583
# 8|11170749
# 9|10446591

avg_folds
# 11060854.2
