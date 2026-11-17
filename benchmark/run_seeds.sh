#!/bin/bash
# Five-seed runs of the CP-PINN (Section 3.2, Fig. 7, Table 7): seeds 1234, 1334, 2024, 1434, 1534, 2000 epochs each, two runs at a time.
# Writes results/cp_s<seed>_e2000.json (final metrics, loss and test R2 of every epoch). The other models are run with the commands in jobs_all.txt
# (python3 bench.py <variant> <seed> <epochs> [weight=multiplier ...]). Run from this folder: ./run_seeds.sh
cd "$(dirname "$0")"
mkdir -p results
for pair in "1234 1334" "2024 1434" "1534"; do
  for s in $pair; do python3 bench.py cp $s 2000 > results/log_cp_$s.txt 2>&1 & done
  wait
done
