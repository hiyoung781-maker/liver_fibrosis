#!/bin/bash
# K-BDS runner for Task 6: the two RL arms (spec section 4.2/4.3).
#
# Conditional on Task 5's pre-registered gate (scripts/tl_sweep_report.py
# select_epoch): TL-A' only runs if priors/focused_A_prime.prior exists,
# which only happens if the Task 5 TL sweep ran and select_epoch() returned
# an epoch (not None). If Task 5 hasn't run yet, or select_epoch returned
# None, TL-A' is dropped and this launches TL-C alone - that is the
# pre-registered outcome, not a fallback improvised here.
#
#   ssh bdata-gpuNN                 # allocated GPU node
#   cd ~/liver_fibrosis
#   bash scripts/kbds_submit_rl.sh  # launches whichever arm(s) are ready, backgrounded
#
# What each arm produces (spec section 4.2 diagnostics, every 100 steps):
#   results/v2_TL-<arm>/rl_final.chkpt, rl_1.csv, ...  - NN-Tanimoto distribution,
#   acid pass rate, unique scaffolds, alert match rate, SAScore.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p logs/v2_TL-A-prime logs/v2_TL-C results/v2_TL-A-prime results/v2_TL-C

run_arm () {
  local arm="$1" tag="$2" config="$3"
  local log="logs/${tag}/rl_$(date +%Y%m%d-%H%M%S).log"
  echo "launching RL arm ${arm} (${config}) in background, log: ${log}"
  nohup bash -c "
    set -euo pipefail
    eval \"\$(conda shell.bash hook)\"
    conda activate \"\${ENV_NAME:-reinvent}\"
    reinvent -l logs/${tag}/rl.log ${config}
  " > "${log}" 2>&1 &
  echo "PID $!  -  tail -f ${log}"
}

if [[ -f priors/focused_A_prime.prior ]]; then
  echo "priors/focused_A_prime.prior found: TL-A' is IN. Running both arms."
  run_arm "A-prime" "v2_TL-A-prime" "configs/rl_A_prime.toml"
  run_arm "C"       "v2_TL-C"       "configs/rl_C.toml"
else
  echo "priors/focused_A_prime.prior missing: Task 5 TL sweep has not produced" >&2
  echo "a qualifying checkpoint yet (either not run, or select_epoch() returned" >&2
  echo "None). Per the pre-registered rule, TL-A' is DROPPED. Running TL-C only." >&2
  run_arm "C" "v2_TL-C" "configs/rl_C.toml"
fi
