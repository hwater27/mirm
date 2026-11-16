#!/usr/bin/env bash
# Simulation study: smoke / pilot / heavy, or 'from' (heavy design with items from fitted empirical runs).
# 'from' takes every generator's items from the runs in FROM_RUNS (the PISA Reading, Science and Mathematics runs):
# MiRM from their per-item MiRM, 2PL / 3PL / LPE / Spline from their full-data fits of the same arm.
# Usage: ./synthetic.sh from    (SEED=27 WORKERS=30 ANALYSIS_WORKERS=10 DEVICE=cpu SELECT_TOL=1e-7 ./synthetic.sh heavy)
set -euo pipefail

PROFILE=${1:-from}
WORKERS=${WORKERS:-30}
ANALYSIS_WORKERS=${ANALYSIS_WORKERS:-10}               # worker processes of the analysis (CPU)
THREADS=${THREADS:-1}
DEVICE=${DEVICE:-cpu}
SELECT_TOL=${SELECT_TOL:-1e-7}                       # early-stopping tolerance of every selection round
SEED=${SEED:-27}                                     # global seed (initialisation, splits' folds, sampling)
FROM_RUNS="../v4_8/runs/application/20261020_125753_pisa_2015_R ../v4_8/runs/application/20261019_133510_pisa_2015_S ../v4_8/runs/application/20261019_133728_pisa_2015_M"

from_args=()
if [[ "$PROFILE" == "from" ]]; then
  for r in $FROM_RUNS; do
    [ -f "$r/run_meta.json" ] && [ -f "$r/jobs.pt" ] || { echo "missing or unfinished run: $r" >&2; exit 1; }
  done
  from_args=(--from-runs $FROM_RUNS)
fi

python -u -m mirm.run.simulation \
  --data-fname "$PROFILE" \
  --num-workers "$WORKERS" \
  --analysis-workers "$ANALYSIS_WORKERS" \
  --threads-per-worker "$THREADS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --select-early-stop-tol "$SELECT_TOL" \
  --tag "_$PROFILE" \
  "${from_args[@]}"
