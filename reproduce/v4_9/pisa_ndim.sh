#!/usr/bin/env bash
# PRM ablation across finished PISA runs.
# Usage: R_RUN=... S_RUN=... M_RUN=... FULL_PRM=... FULL_NOPRM=... ./pisa_ndim.sh
#        (ELEV=25 AZIM=-52 MAX_POINTS=20000: 3D view and the number of students drawn in the cloud)
#        (FLIP_NOPRM=auto|1|0: reflect the no-PRM fit theta -> -theta (alpha -> -alpha); auto reflects it when its ICCs
#         match the PRM fit's ICCs better reflected)
#        (LOW zoom: half-width LOW_HALF, centred between the common start and the first near-crossing of the two paths,
#         at most LOW_SHIFT * LOW_HALF from the start; LOW_RADIUS: inset radius)
#        (HIGH zoom at the high-theta turn: HIGH_INSET=1|0; inset next to it on HIGH_SIDE=below|above|right|left|auto,
#         HIGH_CENTER=x,y (figure coordinates) overrides; HIGH_RADIUS: inset radius; HIGH_CIRCLE: dashed marker radius;
#         HIGH_HALF: zoom half-width in domain coordinates)
set -euo pipefail

: "${R_RUN:?set R_RUN to the Reading run directory}"
: "${S_RUN:?set S_RUN to the Science run directory}"
: "${M_RUN:?set M_RUN to the Mathematics run directory}"
: "${FULL_PRM:?set FULL_PRM to the full-bank PRM run directory}"
: "${FULL_NOPRM:?set FULL_NOPRM to the full-bank no-PRM run directory}"
OUT=${OUT:-./runs/ablation/pisa_2015_$(date +%Y%m%d_%H%M%S)}
ELEV=${ELEV:-25}
AZIM=${AZIM:--52}
MAX_POINTS=${MAX_POINTS:-20000}
FLIP_NOPRM=${FLIP_NOPRM:-auto}
LOW_HALF=${LOW_HALF:-2.2}
LOW_SHIFT=${LOW_SHIFT:-0.75}
LOW_RADIUS=${LOW_RADIUS:-0.14}
HIGH_INSET=${HIGH_INSET:-1}
HIGH_SIDE=${HIGH_SIDE:-below}
HIGH_CENTER=${HIGH_CENTER:-auto}
HIGH_RADIUS=${HIGH_RADIUS:-0.10}
HIGH_CIRCLE=${HIGH_CIRCLE:-0.055}
HIGH_HALF=${HIGH_HALF:-1.1}

for d in "$R_RUN" "$S_RUN" "$M_RUN" "$FULL_PRM" "$FULL_NOPRM"; do
  [ -f "$d/run_meta.json" ] || { echo "missing run: $d" >&2; exit 1; }
done

export R_RUN S_RUN M_RUN FULL_PRM FULL_NOPRM OUT ELEV AZIM MAX_POINTS FLIP_NOPRM \
       LOW_HALF LOW_SHIFT LOW_RADIUS HIGH_INSET HIGH_SIDE HIGH_CENTER HIGH_RADIUS HIGH_CIRCLE HIGH_HALF
python -u - <<'PYEOF'
import os
from pathlib import Path
from mirm.run.utils.artifacts import prm_ablation

E = os.environ
insets = {"low_half": float(E["LOW_HALF"]), "low_shift": float(E["LOW_SHIFT"]), "low_radius": float(E["LOW_RADIUS"]),
          "high": E["HIGH_INSET"] == "1", "high_side": E["HIGH_SIDE"], "high_radius": float(E["HIGH_RADIUS"]),
          "high_circle": float(E["HIGH_CIRCLE"]), "high_half": float(E["HIGH_HALF"]),
          "high_center": None if E["HIGH_CENTER"] == "auto" else tuple(float(v) for v in E["HIGH_CENTER"].split(","))}
prm_ablation(
    anchors={"R": Path(E["R_RUN"]), "S": Path(E["S_RUN"]), "M": Path(E["M_RUN"])},
    full_prm=Path(E["FULL_PRM"]),
    full_noprm=Path(E["FULL_NOPRM"]),
    out_path=Path(E["OUT"]),
    max_points=int(E["MAX_POINTS"]),
    elev=float(E["ELEV"]),
    azim=float(E["AZIM"]),
    flip_noprm={"1": True, "0": False}.get(E["FLIP_NOPRM"], "auto"),
    insets=insets,
)
print(E["OUT"])
PYEOF
