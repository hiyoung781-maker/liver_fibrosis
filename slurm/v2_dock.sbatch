#!/bin/bash
#SBATCH -J avb1_dock_v2
#SBATCH -p 8gpu
#SBATCH --comment pytorch
#SBATCH --time 08:00:00
#SBATCH --nodes 1
#SBATCH -o logs/dock_v2_%j.out
#SBATCH -e logs/dock_v2_%j.err
#
# AutoDock-GPU batch run over the ~9,726 v2 ligands (both arms), 8 shards / 8xA100.
# See scripts/dock_autodock_gpu.py and results/v2_EXECUTION_CHECKLIST.md's Task 9
# AMENDMENT section for the --filelist/--devnum design this script drives.
#
#   mkdir -p logs && sbatch slurm/v2_dock.sbatch
#
# The module system only affects the shell that loads it -- these two lines MUST
# run before anything else, or CUDA is silently absent and a docking failure looks
# indistinguishable from an environment failure.
module load compilers/cuda/12.4
export PATH="$HOME/bin:$PATH"

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "=============================================================="
echo "job      : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "partition: ${SLURM_JOB_PARTITION:-?}"
echo "started  : $(date)"
echo "=============================================================="

echo
echo "== GPU as the compute node sees it =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# $HOME/bin/autodock_gpu is a symlink to the built bin/autodock_gpu_64wi -- the
# Makefile names the binary after its work-item count, so plain "autodock_gpu"
# does not exist on PATH without this symlink.
echo
echo "== autodock_gpu on PATH? =="
if ! command -v autodock_gpu >/dev/null 2>&1; then
    echo "FAIL: autodock_gpu not found on PATH (checked \$HOME/bin and the rest of" \
         "PATH). Expected a symlink at \$HOME/bin/autodock_gpu -> bin/autodock_gpu_64wi." \
         "See results/v2_EXECUTION_CHECKLIST.md, Task 9." >&2
    exit 1
fi
command -v autodock_gpu
autodock_gpu --help | head -5 || true

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-reinvent}"

LIGANDS="${LIGANDS:-docking/v2/ligands_pdbqt/ligands.txt}"
MAPS="${MAPS:-docking/v2/maps}"
OUT="${OUT:-docking/v2/dlg}"
GPUS="${GPUS:-8}"

mkdir -p "$OUT"

echo
echo "== AutoDock-GPU batch run: $LIGANDS -> $OUT ($GPUS shards) =="
echo "REMINDER: the --filelist layout written by write_filelist() is UNVERIFIED" \
     "against the real binary (see the Task 9 AMENDMENT note in" \
     "results/v2_EXECUTION_CHECKLIST.md). Confirm it on a two-ligand test before" \
     "trusting this run's output at scale."
python scripts/dock_autodock_gpu.py --ligands "$LIGANDS" --maps "$MAPS" \
    --out "$OUT" --gpus "$GPUS"

echo
echo "finished : $(date)"
