#!/bin/bash
#SBATCH -J avb1_plip_v2
#SBATCH -p cpu64-only
#SBATCH --time 06:00:00
#SBATCH --nodes 1
#SBATCH -o logs/plip_v2_%j.out
#SBATCH -e logs/plip_v2_%j.err
#
# PLIP over every docked pose of both arms (~53,000). See scripts/plip_batch.py.
#
#   [bdata-login01]$ cd ~/liver_fibrosis && mkdir -p logs
#   [bdata-login01]$ sbatch slurm/v2_plip.sbatch
#
# SUBMIT FROM THE LOGIN NODE, from inside the repository. The user guide
# (p.19) gives `[bdata-login01]$ sbatch [script]`, and the script path is
# relative to where you submit from.
#
# PARTITION cpu64-only, not 8gpu. The guide's queue table (p.18) names the CPU
# queues cpu32-only and cpu64-only -- "cpu64" does not exist. cpu64-only bills
# 1 node-hour per wall hour against 8gpu's 8, and PLIP has no GPU
# implementation, so a GPU queue would burn the allocation for nothing.
# All queues are exclusive-node, so --exclusive is not needed.
#
# DO NOT RUN THIS DIRECTLY ON A NODE. Run as a bare command with --jobs 64 on
# bdata-login01, every worker's plip was killed with SIGINT (exit -2,
# KeyboardInterrupt inside numpy's import). That is the scheduler's policy,
# not a docking or chemistry failure.

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
