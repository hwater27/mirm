#!/usr/bin/env bash
# PISA 2015 full bank (R + S + M items together): MiRM per-item only, full data only (no competitors, no
# prediction), PRM and no-PRM fitted at once on two workers; saved as two runs (_full_prm, _full_noprm) for pisa_ndim.sh.
# no-PRM switches every PRM off (selection rounds and refit).
# Usage: ./pisa_full.sh        (SEED=27 PREFIX=pisa_2015_v1 DEVICE=cpu SELECT_TOL=1e-7 ./pisa_full.sh)
set -euo pipefail

PREFIX=${PREFIX:-pisa_2015_v1}
WORKERS=${WORKERS:-2}
ANALYSIS_WORKERS=${ANALYSIS_WORKERS:-10}               # worker processes of the analysis (CPU)
THREADS=${THREADS:-1}
DEVICE=${DEVICE:-cuda}
SELECT_TOL=${SELECT_TOL:-1e-7}                       # early-stopping tolerance of every selection round
SEED=${SEED:-27}                                     # global seed (initialisation, splits' folds, sampling)
DATA_DIR=${DATA_DIR:-../../../data/processed}

python -u -m mirm.run.application \
  --data-dir "$DATA_DIR" \
  --data-name pisa_2015 \
  --data-fname "${PREFIX}.pt" \
  --arms "MiRM per-item" \
  --test-seeds \
  --prm-conditions prm no-prm \
  --select-early-stop-tol "$SELECT_TOL" \
  --num-workers "$WORKERS" \
  --analysis-workers "$ANALYSIS_WORKERS" \
  --threads-per-worker "$THREADS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --tag "_full"
