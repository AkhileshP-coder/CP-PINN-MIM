#!/bin/sh
# usage: ./runjobs.sh jobs.txt   (two runs at a time; each line: variant seed epochs [weight=multiplier ...])
cd "$(dirname "$0")"; mkdir -p results
cat "$1" | xargs -P2 -L1 sh -c 'python3 bench.py "$@" > "results/log_$(echo "$*" | tr " =" "__").txt" 2>&1' sh
