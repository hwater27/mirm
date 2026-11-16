#!/usr/bin/env bash
# Preprocess PISA 2015 (R / S / M / full) and EdNet KT1 into data/processed.
# Output names follow the run scripts: pisa_2015/${PREFIX}_{R,S,M}.pt, pisa_2015/${PREFIX}.pt (full), ednet/ednet_${EDNET_TOP_K_ITEMS}.pt.
# Paths are overridable: RAW=... OUT=... PREFIX=pisa_2015_v1 SCOPE=cba ./preprocess.sh
set -euo pipefail

RAW=${RAW:-../../../data/raw}
OUT=${OUT:-../../../data/processed}
PREFIX=${PREFIX:-pisa_2015_v1}                        # PISA file prefix read by pisa_{R,S,M,full}.sh
SCOPE=${SCOPE:-cba}                                   # cba | pba | union | intersection
PISA_SRC=${PISA_SRC:-$RAW/pisa_2015}                  # directory holding cy6_ms_cmb_stu_cog (.sav/.txt) + format file
EDNET_KT1=${EDNET_KT1:-$RAW/ednet/KT1}                # directory of per-student KT1 csv files
EDNET_QUESTIONS=${EDNET_QUESTIONS:-$RAW/ednet/contents/questions.csv}
EDNET_TOP_K_ITEMS=${EDNET_TOP_K_ITEMS:-1000}

for DOMAIN in R S M full; do
  if [[ "$DOMAIN" == "full" ]]; then NAME="${PREFIX}.pt"; else NAME="${PREFIX}_${DOMAIN}.pt"; fi
  python -c "
from mirm.data.preprocessors import preprocess
preprocess(
    'pisa_2015',
    input_data_path='$PISA_SRC',
    output_path='$OUT/pisa_2015/$NAME',
    domain='$DOMAIN',
    administration_scope='$SCOPE',
)
"
done

python -c "
from mirm.data.preprocessors import preprocess
preprocess(
    'ednet',
    kt1_path='$EDNET_KT1',
    questions_path='$EDNET_QUESTIONS',
    output_path='$OUT/ednet/ednet_${EDNET_TOP_K_ITEMS}.pt',
    first_response_only=True,
    top_k_items=$EDNET_TOP_K_ITEMS,
    min_students_per_item=200,
    min_items_per_student=20,
)
"
