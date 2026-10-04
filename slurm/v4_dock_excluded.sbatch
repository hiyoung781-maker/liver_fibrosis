#!/bin/bash
#SBATCH -J avb1_v4_excluded
#SBATCH -p 8gpu
#SBATCH --comment pytorch
#SBATCH --time 10:00:00
#SBATCH --nodes 1
#SBATCH -o logs/v4_excluded_%j.out
#SBATCH -e logs/v4_excluded_%j.err
#
# MEASURE THE FALSE-NEGATIVE RATE OF THE PRE-DOCKING ADMET WINDOW (section 5.5).
#
# v4 moved the ADMET window ahead of docking, taking 13,767 survivors down to 334
# docked ligands. The saving is real (40x of the docking budget) and the window is
# pre-registered -- but the 13,433 rejects were never scored against the receptor,
# so the campaign cannot state how many of them would have cleared the section 5.3
# binding-mode gate and the affinity cut. That rate is what this job measures.
#
#   mkdir -p logs && sbatch slurm/v4_dock_excluded.sbatch
#
# TIMING, from the measured v4 rate of 335 ligands / 25 min on 2 GPUs (9 GPU-s per
# ligand): 13,433 ligands on 8 A100s is about 4.2 h of docking, preceded by about
# 50 min of 3D embedding (prepare_ligands reports ~30 min for 7,767). The 10 h wall
# clock leaves headroom for the embedding and for A100s running slower per-ligand
# than the TITAN Xp does on a smaller batch.
#
# WHY THE GATE ORDER IS REVERSED HERE. The main run gated binding mode first
# (2,442 poses -> 44) and affinity second (-> 24). This job applies affinity first
# and runs PLIP only on what survives. The conjunction is commutative, so the
# final set is identical -- but PLIP is CPU-bound and ~98,000 poses of it would
# cost more than the docking. Filtering on the affinity cut first leaves a few
# thousand poses to analyse instead.

module load compilers/cuda/12.4
export PATH="$HOME/bin:$PATH"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "=============================================================="
echo "job      : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "started  : $(date)"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# The A100 is sm_80, which the stock Uni-Dock build targets. tools/unidock-sm61 is
# the LOCAL TITAN Xp build (sm_61) and its RUNPATH points at a local conda env, so
# it must NOT be put on PATH here -- use the cluster's own unidock.
if ! command -v unidock >/dev/null 2>&1; then
    echo "FAIL: unidock not on PATH. docs/KBDS_RUNBOOK_v2.md section 1 records" \
         "that the cluster carries unidock/unidocktools/Meeko already; if that" \
         "has changed, install it before resubmitting. Do NOT fall back to" \
         "tools/unidock-sm61 -- that binary is sm_61 and will not run here." >&2
    exit 1
fi
command -v unidock

eval "$(conda shell.bash hook)"

EXCLUDED="${EXCLUDED:-data/v4_TL-B/admet_excluded.smi}"
SDF_DIR="${SDF_DIR:-docking/v4_excluded/sdf}"
PDBQT_DIR="${PDBQT_DIR:-docking/v4_excluded/pdbqt}"
POSES="${POSES:-docking/v4_excluded/poses}"
RECEPTOR="${RECEPTOR:-docking/receptor.pdbqt}"
GPUS="${GPUS:-8}"

test -s "$EXCLUDED" || { echo "FAIL: $EXCLUDED missing. Run scripts/build_admet_excluded.py" >&2; exit 1; }
test -s "$RECEPTOR" || { echo "FAIL: receptor $RECEPTOR missing" >&2; exit 1; }
echo "ligands to dock: $(grep -vc '^#' "$EXCLUDED")"

# ---- 1. 3D embedding (CPU) ------------------------------------------------
# Resumable by hand: prepare_ligands writes shard SDFs, so a job that dies here
# can be resubmitted after deleting only the last (possibly truncated) shard.
if [ -z "$(ls -A "$SDF_DIR" 2>/dev/null)" ]; then
    conda activate admet
    echo; echo "== embedding $EXCLUDED -> $SDF_DIR =="
    python scripts/prepare_ligands.py --survivors "$EXCLUDED" --out-dir "$SDF_DIR"
    conda deactivate
else
    echo "== $SDF_DIR already populated, skipping embedding =="
fi

# ---- 2. Meeko SDF -> PDBQT (CPU) -----------------------------------------
# Meeko lives in the `admet` env locally, not `reinvent`: section 8.5's PLIP run
# failed 2,442/2,442 poses because meeko was absent from `reinvent`. Keep the two
# steps in the env that actually carries the dependency.
if [ -z "$(ls -A "$PDBQT_DIR" 2>/dev/null)" ]; then
    conda activate admet
    echo; echo "== Meeko $SDF_DIR -> $PDBQT_DIR =="
    python scripts/ligands_to_pdbqt.py --sdf-dir "$SDF_DIR" --out-dir "$PDBQT_DIR"
    conda deactivate
else
    echo "== $PDBQT_DIR already populated, skipping Meeko =="
fi

find "$PDBQT_DIR" -name '*.pdbqt' | sort > docking/v4_excluded/ligands.txt
echo "pdbqt prepared: $(wc -l < docking/v4_excluded/ligands.txt)"

# ---- 3. Docking (8 GPUs) --------------------------------------------------
# --resume re-queues anything missing from $POSES, so a requeued job continues
# rather than restarting the 4 h run. It deliberately re-docks the most recently
# modified output, which may have been truncated mid-write.
conda activate reinvent
echo; echo "== Uni-Dock: $(wc -l < docking/v4_excluded/ligands.txt) ligands on $GPUS GPUs =="
python scripts/dock_unidock.py \
    --ligand-index docking/v4_excluded/ligands.txt \
    --receptor "$RECEPTOR" \
    --out-dir "$POSES" \
    --shard-dir docking/v4_excluded/shards \
    --gpus "$GPUS" --resume

echo
echo "docked: $(find "$POSES" -name '*_out.pdbqt' | wc -l)"
echo "finished : $(date)"
echo
echo "NEXT, on a CPU partition (PLIP has no GPU implementation):"
echo "  scripts/collect_affinities.py --poses $POSES --out results/v4_excluded/affinity.csv"
echo "  then PLIP only the poses that clear the affinity cut -- see this script's header."
