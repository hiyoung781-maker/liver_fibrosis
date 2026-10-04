#!/bin/bash
# Dock the 13,433 molecules the pre-docking ADMET window rejected, on gpu-8-002.
#
# WHY: v4 moved the ADMET window ahead of docking, taking 13,767 survivors down to
# 334 docked ligands. The saving is real (40x of the docking budget) and the window
# is pre-registered -- but those 13,433 rejects were never scored against the
# receptor, so the campaign cannot state how many would have cleared the section 5.3
# binding-mode gate and the affinity cut. That false-negative rate is what this run
# measures.
#
# THIS IS NOT AN SBATCH SCRIPT, DELIBERATELY. On K-BDS, ~/kbds/kbds_gpu/*.sh are
# ALLOCATION HOLDERS -- each sbatch's an infinite sleep loop to hold a node, and real
# work happens interactively inside the held allocation. See docs/KBDS_GPU_DOCKING.md
# ("Workflow this fits"): the deliverable for a docking run is a script run inside a
# held allocation, not a batch job. An #SBATCH-headed script would queue a SECOND job
# behind the one already holding the node.
#
# HOW TO RUN
#
#   # 1. LOGIN NODE -- the compute node cannot resolve github.com at all.
#   cd /home01/$USER/liver_fibrosis && git pull
#
#   # 2. Hold a node, then enter it.
#   sbatch ~/kbds/kbds_gpu/gpu-8-002.sh      # or however the holder is named
#   ssh gpu-8-002                            # or: srun --jobid=<id> --pty bash
#
#   # 3. ON THE NODE. nohup, because this runs ~5 h and an ssh drop would kill it.
#   cd /home01/$USER/liver_fibrosis
#   nohup bash scripts/kbds_dock_excluded.sh > logs/v4_excluded.log 2>&1 &
#   tail -f logs/v4_excluded.log
#
# TIMING, from the measured v4 rate of 335 ligands / 25 min on 2 GPUs (9 GPU-s per
# ligand): 13,433 ligands on 8 A100s is about 4.2 h of docking, preceded by about
# 50 min of 3D embedding (prepare_ligands reports ~30 min for 7,767). Call it 5 h.
# Every stage is skipped if its output directory is already populated, and the
# docking step passes --resume, so re-running this script after an interruption
# continues rather than restarting.
#
# WHY THE GATE ORDER IS REVERSED. The main run gated binding mode first (2,442 poses
# -> 44) and affinity second (-> 24). The follow-up here applies the affinity cut
# first and runs PLIP only on what survives. The conjunction is commutative, so the
# final set is identical -- but PLIP is CPU-bound and ~98,000 poses of it would cost
# more than the docking itself.

set -uo pipefail       # NOT -e: see the conda note below.

cd "$(dirname "$0")/.."
mkdir -p logs

echo "=============================================================="
echo "host    : $(hostname)"
echo "started : $(date)"
echo "=============================================================="

# `module load` is a NO-OP on this cluster's compute nodes (measured: cmake fails on
# an unset env(COMPILER_VER), no nvcc from any CUDA module). The conda unidock
# package ships its own CUDA runtime, which is the whole reason conda is used here --
# so there is nothing to load and nothing to check for.
#
# Two miniconda installs exist: ~/kbds/kbds_gpu/*.sh source
# /apps/application/miniconda3, while the compute node's CONDA_EXE points at
# /apps/application/miniconda3-new. Use whichever carries the `unidock` env.
CONDA_SH=""
for candidate in /apps/application/miniconda3/etc/profile.d/conda.sh \
                 /apps/application/miniconda3-new/etc/profile.d/conda.sh \
                 "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    [ -f "$candidate" ] || continue
    # shellcheck disable=SC1090
    . "$candidate"
    if conda env list | grep -qE '^unidock\s'; then CONDA_SH="$candidate"; break; fi
done
if [ -z "$CONDA_SH" ]; then
    echo "FAIL: no conda install carries a 'unidock' env. Create it ON THE LOGIN" \
         "NODE (the compute node has no internet) per docs/KBDS_GPU_DOCKING.md:" >&2
    echo "  conda create -y -n unidock --override-channels -c conda-forge \\" >&2
    echo "      python=3.11 'unidock=1.1.3=cuda118*' rdkit openbabel meeko numpy pandas" >&2
    exit 1
fi
echo "conda   : $CONDA_SH"

# `set -e` is deliberately off: under this cluster's own conda.sh an earlier setup
# script aborted with NO OUTPUT AT ALL when combined with -e (recorded in
# docs/KBDS_GPU_DOCKING.md's opening note). Every step below therefore checks its
# own exit status explicitly.
conda activate unidock || { echo "FAIL: conda activate unidock" >&2; exit 1; }

# One env for all three stages. The unidock package set carries rdkit (embedding),
# meeko (PDBQT), and numpy -- which is every dependency the three scripts import.
# Locally these live in two different envs; here they do not, and section 8.5's
# 2,442/2,442 PLIP failure came from running a script in the env that lacked meeko.
python - <<'PY' || exit 1
import sys
missing = []
for mod in ("rdkit", "numpy", "meeko"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit(f"FAIL: env 'unidock' lacks {', '.join(missing)}")
print("deps    : rdkit, numpy, meeko all import")
PY

command -v unidock >/dev/null 2>&1 || { echo "FAIL: unidock not on PATH" >&2; exit 1; }
echo "unidock : $(command -v unidock)"
# tools/unidock-sm61 is the LOCAL TITAN Xp build (sm_61) and its RUNPATH points at a
# local conda env. The A100 is sm_80. Never put it on PATH here.
case "${PATH}" in
    *unidock-sm61*) echo "FAIL: tools/unidock-sm61 is on PATH. That is the local" \
                         "sm_61 build and will not run on an A100. Remove it." >&2
                    exit 1;;
esac

# Do NOT pre-export CUDA_VISIBLE_DEVICES: dock_unidock.py sets it per shard
# (CUDA_VISIBLE_DEVICES=<shard index>) and would overwrite an outer pin, silently
# sending every shard to whichever physical device index it chose. If some GPUs on
# this node are busy with another job, lower GPUS instead.
echo
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

EXCLUDED="${EXCLUDED:-data/v4_TL-B/admet_excluded.smi}"
WORK="${WORK:-docking/v4_excluded}"
RECEPTOR="${RECEPTOR:-docking/receptor.pdbqt}"
GPUS="${GPUS:-8}"

test -s "$EXCLUDED" || { echo "FAIL: $EXCLUDED missing. Run scripts/build_admet_excluded.py" >&2; exit 1; }
test -s "$RECEPTOR" || { echo "FAIL: receptor $RECEPTOR missing" >&2; exit 1; }
echo
echo "ligands to dock: $(grep -vc '^#' "$EXCLUDED")   GPUs: $GPUS"

# ---- 1. 3D embedding (CPU) ------------------------------------------------
if [ -z "$(ls -A "$WORK/sdf" 2>/dev/null)" ]; then
    echo; echo "== embedding $EXCLUDED -> $WORK/sdf =="
    python scripts/prepare_ligands.py --survivors "$EXCLUDED" --out-dir "$WORK/sdf" \
        || { echo "FAIL: prepare_ligands" >&2; exit 1; }
else
    echo "== $WORK/sdf already populated, skipping embedding =="
fi

# ---- 2. Meeko SDF -> PDBQT (CPU) -----------------------------------------
if [ -z "$(ls -A "$WORK/pdbqt" 2>/dev/null)" ]; then
    echo; echo "== Meeko $WORK/sdf -> $WORK/pdbqt =="
    python scripts/ligands_to_pdbqt.py --sdf-dir "$WORK/sdf" --out-dir "$WORK/pdbqt" \
        || { echo "FAIL: ligands_to_pdbqt" >&2; exit 1; }
else
    echo "== $WORK/pdbqt already populated, skipping Meeko =="
fi

# ligands_to_pdbqt.py writes its own index; prefer it over a fresh find so the
# docking input is exactly what the converter claims it produced. Fall back to find
# only if a hand-edited pdbqt dir has no index.
if [ -s "$WORK/pdbqt/ligands.txt" ]; then
    cp "$WORK/pdbqt/ligands.txt" "$WORK/ligands.txt"
else
    find "$WORK/pdbqt" -name '*.pdbqt' | sort > "$WORK/ligands.txt"
fi
echo "pdbqt prepared: $(wc -l < "$WORK/ligands.txt")"

# prepare_ligands.py adds a CONTROL_crystal shard (the 8W30 crystal ligand) on its
# own. Docking it here is not redundant with the recorded 0.63 A redocking result --
# docs/KBDS_GPU_DOCKING.md asks for exactly this as a PLUMBING check on the cluster
# build and the receptor prep, not a revalidation of the method. If CONTROL_crystal
# comes back with a nonsense affinity, the problem is the build or the PDBQT, and
# every other ligand's score is suspect for a reason that looks chemical.
grep -q 'CONTROL_crystal' "$WORK/ligands.txt" \
    && echo "control : CONTROL_crystal is in the docking set (plumbing check)" \
    || echo "control : WARNING - no CONTROL_crystal in the set; the build goes unchecked"

# ---- 3. Docking (8 GPUs) --------------------------------------------------
# --resume re-queues anything missing from the output dir, so re-running this script
# continues the 4 h run. It deliberately re-docks the most recently modified output,
# which may have been truncated mid-write.
echo; echo "== Uni-Dock: $(wc -l < "$WORK/ligands.txt") ligands on $GPUS GPUs =="
python scripts/dock_unidock.py \
    --ligand-index "$WORK/ligands.txt" \
    --receptor "$RECEPTOR" \
    --out-dir "$WORK/poses" \
    --shard-dir "$WORK/shards" \
    --gpus "$GPUS" --resume \
    || { echo "FAIL: dock_unidock (re-run this script to resume)" >&2; exit 1; }

echo
echo "docked  : $(find "$WORK/poses" -name '*_out.pdbqt' | wc -l)"
echo "finished: $(date)"
echo
echo "NEXT, on a CPU allocation (PLIP has no GPU implementation): apply the affinity"
echo "cut to $WORK/poses first, then PLIP only what survives -- see this script's"
echo "header for why that order. Only the summary CSV needs to come back; the pose"
echo "directory is ~350 MB and does not belong in git."
