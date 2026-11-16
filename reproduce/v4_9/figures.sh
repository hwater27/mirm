#!/usr/bin/env bash
# Cross-run figures: (c,d) selection consistency (EdNet, Reading, Science, Math) and the three-item dip showcase.
# Usage: EDNET_RUN=... R_RUN=... S_RUN=... M_RUN=... ./figures.sh     (EDNET_RUN=blank:100 for an empty panel)
#        (SORT=zigzag|config|index: zigzag = configurations by c+d, alternating direction within each c+d, so
#         consecutive configurations are lattice neighbours; config = c ascending within each c+d)
set -euo pipefail

: "${R_RUN:?set R_RUN to the Reading run directory}"
: "${S_RUN:?set S_RUN to the Science run directory}"
: "${M_RUN:?set M_RUN to the Mathematics run directory}"
EDNET_RUN=${EDNET_RUN:-blank:100}
SORT=${SORT:-zigzag}
OUT=${OUT:-./runs/figures/$(date +%Y%m%d_%H%M%S)}

for d in "$R_RUN" "$S_RUN" "$M_RUN"; do
  [ -f "$d/tables/selection.csv" ] || { echo "missing selection table: $d" >&2; exit 1; }
done
if [[ "$EDNET_RUN" != blank:* && ! -f "$EDNET_RUN/tables/selection.csv" ]]; then
  echo "EdNet selection table not found ($EDNET_RUN) -> blank panel" >&2
  EDNET_RUN=blank:100
fi
mkdir -p "$OUT"

export R_RUN S_RUN M_RUN EDNET_RUN SORT OUT
python -u - <<'PYEOF'
import os
from pathlib import Path
from mirm.run.utils.artifacts import cd_consistency, dip_showcase

E = os.environ
out = Path(E["OUT"])
cd_consistency([("EdNet", E["EDNET_RUN"]), ("PISA Reading", E["R_RUN"]), ("PISA Science", E["S_RUN"]), ("PISA Math", E["M_RUN"])],
               out / "cd_consistency.png", sort=E["SORT"])
dip_showcase({"R": Path(E["R_RUN"]), "S": Path(E["S_RUN"]), "M": Path(E["M_RUN"])}, out / "dip_showcase.png")
print(out)
PYEOF
