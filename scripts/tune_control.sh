#!/bin/bash
# Sweep --exhaustiveness on the control ligand alone, and report what each setting
# recovers. Run inside the GPU allocation, with the unidock env active.
#
#   bash scripts/tune_control.sh
#
# WHY. Section 8.5(a)'s control failed: Uni-Dock reproduced the Ca501 contact
# (2.378 A) but not beta1-Asn224 (4.830 A against a crystal 2.63 A). The first
# hypothesis - that Uni-Dock's default --energy_range 3 was pruning poses - was
# tested and REFUTED: widening it to 10 took the pose count from 2 to 3, and all
# three sat within 0.11 kcal/mol and 0.86 A of each other. The search had converged,
# not been filtered.
#
# The actual signal is the score. With the SAME scoring function, smina found -6.91
# at exhaustiveness 16 while Uni-Dock found -6.30. A worse optimum from an identical
# function is a search-depth problem, so exhaustiveness is the lever.
#
# What to look for in the output below, per setting:
#   affinity approaching smina's -6.91   -> the search is now finding that basin
#   pose count rising toward 20          -> enough poses for a geometry filter to work
#   a pose with donor_dist <= 3.5        -> the Asn224 contact is recoverable
# The smallest setting that delivers all three is the one to use for the full run.

set -o pipefail

RECEPTOR="${RECEPTOR:-docking/receptor.pdbqt}"
LIGAND="${LIGAND:-docking/ligands_pdbqt/CONTROL_crystal.pdbqt}"
BOX="--center_x 2.300 --center_y 114.144 --center_z 40.194 --size_x 16.957 --size_y 18.471 --size_z 21.167"
WORK="${WORK:-/tmp/ctrl_sweep}"
LEVELS="${LEVELS:-16 64 128 256 512}"

command -v unidock >/dev/null || { echo "FATAL: unidock not on PATH"; exit 1; }
[ -f "$LIGAND" ] || { echo "FATAL: $LIGAND missing"; exit 1; }
[ -f "$RECEPTOR" ] || { echo "FATAL: $RECEPTOR missing"; exit 1; }

mkdir -p "$WORK"
realpath "$LIGAND" > "$WORK/one.txt"

echo "control ligand : $LIGAND"
echo "reference      : smina at exhaustiveness 16 gave affinity -6.91, 20 poses,"
echo "                 top pose 2.72 A to Ca501 and 2.89 A to Asn224 O (RMSD 0.63 A)"
echo "crystal        : 2.62 A to Ca501, 2.63 A to Asn224 O"

for ex in $LEVELS; do
  out="$WORK/ex$ex"
  rm -rf "$out"; mkdir -p "$out"
  printf '\n========== exhaustiveness %s ==========\n' "$ex"
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  unidock --receptor "$RECEPTOR" --ligand_index "$WORK/one.txt" --dir "$out" \
          --scoring vina --exhaustiveness "$ex" --num_modes 20 \
          --energy_range 10 --min_rmsd 0.5 --seed 42 $BOX \
          > "$out/unidock.log" 2>&1
  status=$?
  elapsed=$(( $(date +%s) - start ))
  if [ $status -ne 0 ]; then
    echo "  FAILED (exit $status); see $out/unidock.log"
    continue
  fi
  pose_file="$out/CONTROL_crystal_out.pdbqt"
  printf '  wall            %ss\n' "$elapsed"
  printf '  poses           %s\n' "$(grep -c '^MODEL' "$pose_file" 2>/dev/null || echo 0)"
  printf '  best affinity   %s\n' \
    "$(grep -m1 'VINA RESULT' "$pose_file" 2>/dev/null | awk '{print $4}')"
  echo "  --- geometry per pose (ca_dist / donor_dist, cutoffs 3.2 / 3.5) ---"
  python3 scripts/pose_geometry.py --poses <(python3 - "$pose_file" <<'PY'
import sys, pathlib
sys.path.insert(0, "scripts")
from poses_to_sdf import poses_from_pdbqt
from rdkit import Chem
w = Chem.SDWriter("/dev/stdout")
for m in poses_from_pdbqt(sys.argv[1]):
    w.write(m)
w.close()
PY
) --out-csv "$out/geometry.csv" >/dev/null 2>&1 \
    && awk -F, 'NR>1 {printf "    pose %-3s ca %-7s donor %-7s passes %s\n", $2,$4,$6,$9}' \
         "$out/geometry.csv" \
    || echo "    (geometry step failed; run pose_geometry.py manually on $pose_file)"
done

cat <<'NOTE'

=========================================================================
Reading this: the smallest exhaustiveness whose output contains a pose with
ca <= 3.2 AND donor <= 3.5 is the setting for the full run. If no setting
produces one, the problem is not search depth, and section 8.5(a) applies -
do not rank anything by docking; fall back to similarity/QSAR triage and say
so on the poster.

Cost scaling for the full run: the 7,763-ligand batch took ~15 min on 8 GPUs at
exhaustiveness 16, i.e. ~0.93 GPU-seconds per ligand. Multiply by the factor
between 16 and the chosen level, and expect better than linear because batching
amortizes the fixed overhead that dominated these single-ligand runs.
NOTE
