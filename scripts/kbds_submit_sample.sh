#!/bin/bash
# K-BDS runner for Task 7: arm-scoped sampling + post-processing (spec section 6).
#
# Conditional on Task 6 having produced each arm's checkpoint. Only launches
# an arm whose results/v2_<arm>/rl_final.chkpt exists - if TL-A' was dropped
# by Task 5/6's pre-registered gate, its checkpoint never exists and this
# script correctly skips it rather than failing.
#
#   ssh bdata-gpuNN
#   cd ~/liver_fibrosis
#   bash scripts/kbds_submit_sample.sh   # launches whichever arm(s) are ready, backgrounded
#
# Each arm: `reinvent sampling` (20,000 molecules, ~GPU minutes) followed by
# build_library.py --arm and build_survivors.py --arm (post-processing,
# ~0.14 s/molecule of tautomer normalization -> ~47 min per arm on CPU).
# Produces:
#   data/v2_<arm>/library.smi    (~19-20k rows, 2 columns)
#   data/v2_<arm>/survivors.smi

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p logs/v2_TL-A-prime logs/v2_TL-C data/v2_TL-A-prime data/v2_TL-C

run_arm () {
  local arm="$1" tag="$2" config="$3"
  local chkpt="results/${tag}/rl_final.chkpt"
  if [[ ! -f "$chkpt" ]]; then
    echo "skip ${arm}: ${chkpt} not found (Task 6 hasn't produced it, or the" >&2
    echo "arm was dropped by the pre-registered TL sweep gate)" >&2
    return 0
  fi
  local log="logs/${tag}/sample_$(date +%Y%m%d-%H%M%S).log"
  echo "launching sampling + post-processing for arm ${arm} (${config}) in background, log: ${log}"
  nohup bash -c "
    set -euo pipefail
    eval \"\$(conda shell.bash hook)\"
    conda activate \"\${ENV_NAME:-reinvent}\"
    reinvent -l logs/${tag}/sample.log ${config}
    python scripts/build_library.py --arm ${arm}
    python scripts/build_survivors.py --arm ${arm}
  " > "${log}" 2>&1 &
  echo "PID $!  -  tail -f ${log}"
}

run_arm "TL-A-prime" "v2_TL-A-prime" "configs/sample_TL-A-prime.toml"
run_arm "TL-C"       "v2_TL-C"       "configs/sample_TL-C.toml"
