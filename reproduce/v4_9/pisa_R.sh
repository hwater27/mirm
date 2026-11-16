#!/usr/bin/env bash
# PISA 2015 application: Reading (R) with PRM, all arms, full data + prediction splits.
# Usage: ./pisa_R.sh        (SEED=27 PREFIX=pisa_2015_v1 WORKERS=2 TEST_SEEDS="1 3 8" DEVICE=cpu SELECT_TOL=1e-7 ./pisa_R.sh)
set -euo pipefail

PREFIX=${PREFIX:-pisa_2015_v1}
WORKERS=${WORKERS:-2}
ANALYSIS_WORKERS=${ANALYSIS_WORKERS:-10}               # worker processes of the analysis (CPU)
THREADS=${THREADS:-1}
DEVICE=${DEVICE:-cuda}
SELECT_TOL=${SELECT_TOL:-1e-7}                       # early-stopping tolerance of every selection round
SEED=${SEED:-27}                                     # global seed (initialisation, splits' folds, sampling)
TEST_SEEDS=${TEST_SEEDS:-"1 3 8 0 6 4 9"}
DATA_DIR=${DATA_DIR:-../../../data/processed}

python -u -m mirm.run.application \
  --data-dir "$DATA_DIR" \
  --data-name pisa_2015 \
  --data-fname "${PREFIX}_R.pt" \
  --prm \
  --test-seeds $TEST_SEEDS \
  --select-early-stop-tol "$SELECT_TOL" \
  --num-workers "$WORKERS" \
  --analysis-workers "$ANALYSIS_WORKERS" \
  --threads-per-worker "$THREADS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --tag "_R"
