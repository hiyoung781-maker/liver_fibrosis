#!/bin/bash
# K-BDS runner for the TL-A' production sweep (spec §4.2, Task 5 step 5).
#
# No SLURM/sbatch is used on K-BDS in this project (see scripts/run_d3.sh,
# scripts/setup_kbds.sh) - GPU jobs run directly on the allocated GPU node,
# backgrounded with nohup so the session can disconnect safely.
#
# Runs `MODE=production-sweep bash scripts/run_d3.sh --arm A-prime` exactly as
# the task brief names it. run_d3.sh's --arm/ARM= selector (added after review)
# restricts phases 1 and 2 to the core/TL-A arm, so this run does NOT touch the
# core_B/TL-B diagnostic fold or TL-B production at all - v2 discards TL-B
# outright, and running it here would burn real GPU allocation for nothing on a
# node-hour-billed partition.
#
#   ssh bdata-gpuNN                       # allocated GPU node
#   cd ~/liver_fibrosis
#   bash scripts/kbds_submit_tl_sweep.sh  # launches sweep + report in background
#
# What this produces:
#   priors/focused_A_prime.prior(.N.chkpt for N in 20,40,...,180)  - checkpoints
#   logs/d3/<timestamp>/prod_core/tl.log                            - training log,
#       must show BOTH `train/loss` and `valid/nll` series (wandb run
#       "tl-a-prime-sweep") - if valid/nll is missing, save_every_n_epochs did not
#       take effect and the sweep must be re-run, not patched after the fact.
#   results/v2_tl_sweep.csv - epoch, acid_pct, unique_scaffolds, max_nn_tanimoto,
#       tpsa_median, qed_median for epoch 0, 20, 40, ..., 200 (11 checkpoints)
#
# Gate (pre-registered, scripts/tl_sweep_report.py select_epoch): highest epoch
# with epoch >= 100 and unique_scaffolds >= 700 (of 1000 sampled). If none
# qualifies, select_epoch returns None and TL-A' is DROPPED as an arm - record
# that in results/v2_tl_decision.md and proceed with TL-C alone in Task 6.
# Memorization (max_nn_tanimoto) is reported, never a disqualifier.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG="logs/tl_a_prime_sweep_$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs priors results

echo "launching TL-A' production sweep in background, log: $LOG"
nohup bash -c '
  set -euo pipefail
  # --arm A-prime restricts run_d3.sh to the core/TL-A arm only: no core_B
  # diagnostic fold, no TL-B production train.
  MODE=production-sweep bash scripts/run_d3.sh --arm A-prime
  # Phase 2 writes checkpoints as priors/focused_A.prior(.N.chkpt) for the core
  # arm - that IS the A-prime sweep output, so aggregate it directly.
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME:-reinvent}"
  python scripts/tl_sweep_report.py --checkpoints priors/focused_A.prior \
      --n-samples 1000 --final-epoch 200 --out results/v2_tl_sweep.csv
  cp priors/focused_A.prior priors/focused_A_prime.prior
' > "$LOG" 2>&1 &

echo "PID $!  -  tail -f $LOG"
