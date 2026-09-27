#!/bin/bash
#SBATCH -J reinvent_smoke
#SBATCH -p debug-1gpu
#SBATCH --comment pytorch
#SBATCH --time 00:30:00
#SBATCH --nodes 1
#SBATCH -o logs/smoke_%j.out
#SBATCH -e logs/smoke_%j.err
#
# First job that actually touches a GPU. It runs the whole stack end to end on the
# real curated data - transfer learning for one epoch, then sampling - so that a
# failure here is a setup failure and not a science failure.
#
#   mkdir -p logs && sbatch slurm/00_smoke.sbatch
#
# debug-1gpu is capped at 8 hours and is the right queue for exactly this.

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

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-reinvent}"

echo
echo "== torch sees the GPU? =="
python - <<'PY'
import sys, torch
print(f"torch {torch.__version__}  built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("FAIL: torch.cuda.is_available() is False - driver/CUDA build mismatch")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"device: {name}  sm_{cap[0]}{cap[1]}")
x = torch.randn(4096, 4096, device="cuda")
print(f"matmul on GPU OK, trace={float((x @ x.T).diagonal().sum()):.3e}")
PY

WORK="$(mktemp -d "${SLURM_SUBMIT_DIR:-$PWD}/logs/smoke.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
echo
echo "== scratch: $WORK =="

# --- transfer learning, 1 epoch on the real TL-A split ------------------------
# Field names are exactly those in REINVENT4 v4.5.11's TL SectionParameters, which
# uses pydantic extra="forbid": an unknown key is a hard error. In particular there
# is NO isomeric_smiles field in this release - and it would be wrong anyway, since
# the de novo prior's vocabulary has no stereochemistry tokens.
cat > "$WORK/tl_smoke.toml" <<TOML
run_type = "transfer_learning"
device = "cuda:0"

[parameters]
num_epochs = 1
save_every_n_epochs = 1
batch_size = 16
sample_batch_size = 100
num_refs = 0
input_model_file = "REINVENT4/priors/reinvent.prior"
smiles_file = "data/folds/actives_core_fold1_train.smi"
validation_smiles_file = "data/folds/actives_core_fold1_valid.smi"
output_model_file = "$WORK/smoke_focused.prior"
standardize_smiles = true
randomize_all_smiles = true
TOML

echo
# Fold 1 of the TL-A diagnostic folds. Any fold would do here - this job proves the
# stack runs, it is not the diagnostic pass itself (that is 5 folds per arm, D3).
echo "== 1. transfer_learning (1 epoch, TL-A fold 1 train split) =="
reinvent -l "$WORK/tl_smoke.log" "$WORK/tl_smoke.toml"
grep -iE "read .* input smiles|epoch|loss" "$WORK/tl_smoke.log" | head -8 || true

[ -s "$WORK/smoke_focused.prior" ] || { echo "FAIL: no focused prior written"; exit 1; }
echo "focused prior written: $(du -h "$WORK/smoke_focused.prior" | cut -f1)"

# --- sampling from the focused prior -----------------------------------------
cat > "$WORK/sample_smoke.toml" <<TOML
run_type = "sampling"
device = "cuda:0"

[parameters]
model_file = "$WORK/smoke_focused.prior"
output_file = "$WORK/smoke_samples.csv"
num_smiles = 200
unique_molecules = true
TOML

echo
echo "== 2. sampling (200 molecules) =="
reinvent -l "$WORK/sample_smoke.log" "$WORK/sample_smoke.toml"

echo
echo "== 3. did the generator produce usable chemistry? =="
python - "$WORK/smoke_samples.csv" <<'PY'
import csv, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

rows = list(csv.DictReader(open(sys.argv[1])))
col = next(c for c in rows[0] if "smiles" in c.lower())
smis = [r[col] for r in rows if r[col]]
mols = [Chem.MolFromSmiles(s) for s in smis]
valid = [m for m in mols if m is not None]
acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
n_acid = sum(1 for m in valid if m.HasSubstructMatch(acid))

print(f"sampled          : {len(smis)}")
print(f"valid            : {len(valid)} ({100*len(valid)/max(len(smis),1):.0f}%)")
print(f"carboxylic acid  : {n_acid} ({100*n_acid/max(len(valid),1):.0f}%)")
if valid:
    mw = sorted(Descriptors.MolWt(m) for m in valid)
    print(f"MW median        : {mw[len(mw)//2]:.0f}")
    print("examples         : " + " | ".join(Chem.MolToSmiles(m) for m in valid[:3]))

if len(valid) / max(len(smis), 1) < 0.8:
    sys.exit("FAIL: validity below 80% - the stack is not healthy")
print("\nSMOKE TEST PASSED")
PY

echo
echo "finished : $(date)"
