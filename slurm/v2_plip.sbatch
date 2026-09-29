#!/bin/bash
#SBATCH -J avb1_plip_v2
#SBATCH -p cpu64
#SBATCH --time 06:00:00
#SBATCH --nodes 1
#SBATCH -o logs/plip_v2_%j.out
#SBATCH -e logs/plip_v2_%j.err
#
# PLIP over every docked pose of both arms (~53,000). See scripts/plip_batch.py.
#
#   mkdir -p logs && sbatch slurm/v2_plip.sbatch
#
# cpu64, NOT 8gpu. PLIP has no GPU implementation, and the 8gpu partition bills
# 8 node-hours per wall hour whether or not a GPU is touched.
#
# AND NOT THE LOGIN NODE. Running this with --jobs 64 on bdata-login01 had every
# worker's plip killed with SIGINT (exit -2, KeyboardInterrupt inside numpy's
# import). That is the scheduler's policy, not a docking or chemistry failure.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# LD_LIBRARY_PATH is exported for autodock_gpu elsewhere in this project and
# MUST NOT reach here: _openbabel.so is built against the old C++ ABI, and
# prepending the conda lib directory makes the loader find a libopenbabel.so
# built against the new one, so the old-ABI std::string symbol goes missing
# (undefined symbol: _ZN9OpenBabel8OBPlugin7DisplayERSsPKcS3_). Nothing here
# runs autodock_gpu.
unset LD_LIBRARY_PATH

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-unidock}"

RECEPTOR="${RECEPTOR:-docking/v2/receptor_h.pdb}"
JOBS="${JOBS:-${SLURM_CPUS_ON_NODE:-64}}"

echo "=============================================================="
echo "job      : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "partition: ${SLURM_JOB_PARTITION:-?}"
echo "jobs     : $JOBS"
echo "started  : $(date)"
echo "=============================================================="

echo
echo "== tools =="
command -v plip   || { echo "FAIL: plip not on PATH" >&2; exit 1; }
command -v obabel || { echo "FAIL: obabel not on PATH" >&2; exit 1; }
plip -h   >/dev/null || { echo "FAIL: plip present but not runnable" >&2; exit 1; }
obabel -V || { echo "FAIL: obabel present but not runnable" >&2; exit 1; }

for ARM in TL-A-prime TL-C; do
    echo
    echo "== PLIP: $ARM =="
    python scripts/plip_batch.py \
        --pose-dir "docking/v2/unidock/${ARM}" \
        --receptor "$RECEPTOR" \
        --out-csv  "results/v2_${ARM}/plip.csv" \
        --jobs     "$JOBS"
done

echo
echo "finished : $(date)"
