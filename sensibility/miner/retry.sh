#!/bin/sh

# Retry downloaded files

set +ex

QUEUE=q:download
ERRORS=$QUEUE:errors

TEMPSET=$QUEUE:temp

redis-cli SADD "$TEMPSET" $(redis-cli LRANGE "$ERRORS" 0 -1)
redis-cli DEL "$ERRORS"
redis-cli LPUSH "$QUEUE"  $(redis-cli SMEMBERS "$TEMPSET")
