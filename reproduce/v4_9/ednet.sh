#!/usr/bin/env bash
# EdNet application (keyed data; intercept audit on), restricted to the TOP most answered items of the file.
# Usage: ./ednet.sh   (SEED=27 ITEMS=1000 TOP=100 WORKERS=2 DEVICE=cpu SELECT_TOL=1e-7 ./ednet.sh)
set -euo pipefail

ITEMS=${ITEMS:-1000}
TOP=${TOP:-100}
WORKERS=${WORKERS:-2}
ANALYSIS_WORKERS=${ANALYSIS_WORKERS:-10}               # worker processes of the analysis (CPU)
THREADS=${THREADS:-1}
DEVICE=${DEVICE:-cuda}
SELECT_TOL=${SELECT_TOL:-1e-7}                       # early-stopping tolerance of every selection round
SEED=${SEED:-27}                                     # global seed (initialisation, splits' folds, sampling)
DATA_DIR=${DATA_DIR:-../../../data/processed}

python -u -m mirm.run.application \
  --data-dir "$DATA_DIR" \
  --data-name ednet \
  --data-fname "ednet_${ITEMS}.pt" \
  --top-items "$TOP" \
  --prm \
  --check-intercept \
  --select-early-stop-tol "$SELECT_TOL" \
  --num-workers "$WORKERS" \
  --analysis-workers "$ANALYSIS_WORKERS" \
  --threads-per-worker "$THREADS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --tag "_${ITEMS}_top${TOP}"
