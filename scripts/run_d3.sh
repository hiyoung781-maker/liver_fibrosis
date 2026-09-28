#!/bin/bash
# Blueprint section 4, D3: choose the transfer-learning epoch count on scaffold folds,
# train the production priors on the full sets, then run the acceptance checks.
#
#   bash scripts/run_d3.sh                 # offline wandb, GPU 1
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_d3.sh --no-wandb
#   bash scripts/run_d3.sh --mode production-sweep --print-config   # render TL config only
#   MODE=production-sweep bash scripts/run_d3.sh --arm A-prime      # TL-A' only, skips TL-B entirely
#
# --mode {diagnostic,production-sweep} picks the (EPOCHS, SAVEFREQ) pair used when
# rendering configs/tl.toml.in: diagnostic is 20/1, production-sweep is 200/20.
# production-sweep exists so the production TL run gets a validation NLL at every
# checkpoint (REINVENT4 computes it only on save epochs) instead of just at the end.
# --print-config renders the resulting TL config to stdout and exits without running
# anything, so it can be diffed or asserted on in tests.
#
# --arm/ARM= {both,core,core_B} (aliases A/A-prime/TL-A -> core, B/TL-B -> core_B)
# restricts phases 1 and 2 to one arm. Default is "both" (unchanged historical
# behaviour: 5-fold diagnostics + full production for core AND core_B), so existing
# callers are unaffected. Use --arm core (or A-prime) for a TL-A'-only production
# sweep - without it, MODE=production-sweep also runs the core_B diagnostic fold and
# a full TL-B production train, which v2 discards outright and which is real,
# non-trivial GPU allocation burn on a node-hour-billed partition, not just wasted
# wall-clock.
#
# Three phases, and phases 1 and 2 must not be confused:
#   1. diagnostic - 5 folds x 2 arms, trains on a fold's train split and validates on
#      its held-out scaffolds, purely to find the epoch where validation NLL turns up
#   2. production - one run per arm on the FULL set with no validation file, using the
#      epoch count phase 1 found. This is the prior that actually generates. Holding
#      6 of 29 molecules back permanently would discard 21% of the public non-RGD
#      chemotype.
#   3. acceptance - sample from both focused priors and the untouched prior
#
# Diagnostics run to 20 epochs, not the blueprint's 10: REINVENT4 warns when the best
# epoch is the last one, which means the minimum was never reached. Opening the budget
# lets the minimum fall inside the range so there is a knee to read.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

N_FOLDS=5
SAMPLE_N="${SAMPLE_N:-1000}"
USE_WANDB=1
MODE="${MODE:-diagnostic}"
ARM="${ARM:-both}"
PRINT_CONFIG=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-wandb) USE_WANDB=0 ;;
    --mode) MODE="$2"; shift ;;
    --mode=*) MODE="${1#*=}" ;;
    --arm) ARM="$2"; shift ;;
    --arm=*) ARM="${1#*=}" ;;
    --print-config) PRINT_CONFIG=1 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# Arm selector: default "both" runs core (TL-A/A-prime) and core_B (TL-B), exactly
# the historical behaviour, so existing callers are unaffected. --arm/ARM= restricts
# phases 1 and 2 to a single arm - added so a production-sweep run for TL-A' does
# not also burn GPU allocation on the TL-B diagnostic fold and full TL-B production,
# which v2 discards outright (see results/v2_EXECUTION_CHECKLIST.md, Task 5).
case "$ARM" in
  both)                 ARMS=(core core_B) ;;
  core|A|A-prime|TL-A)  ARMS=(core) ;;
  core_B|B|TL-B)        ARMS=(core_B) ;;
  *) echo "unknown arm: $ARM (expected both|core|core_B, aliases A/A-prime/TL-A, B/TL-B)" >&2; exit 2 ;;
esac

# Mode selector: diagnostic finds the epoch where validation NLL turns up (fine
# checkpoint granularity); production-sweep is the 200-epoch production run, whose
# checkpoints must land every 20 epochs - REINVENT4 computes validation NLL only on
# save epochs, so a coarse save_every_n_epochs here means the sweep curve has no
# validation NLL to choose an epoch from at all.
case "$MODE" in
  diagnostic)       EPOCHS=20;  SAVEFREQ=1  ;;
  production-sweep) EPOCHS=200; SAVEFREQ=20 ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
DIAG_EPOCHS="${DIAG_EPOCHS:-$EPOCHS}"

render() {  # smiles validation output epochs savefreq tb_logdir -> stdout
  sed -e "s|{SMILES}|$1|" -e "s|{VALIDATION}|$2|" -e "s|{OUTPUT}|$3|" \
      -e "s|{EPOCHS}|$4|" -e "s|{SAVEFREQ}|$5|" -e "s|{TB_LOGDIR}|$6|" \
      configs/tl.toml.in
}

if [ "$PRINT_CONFIG" = 1 ]; then
  render "data/actives_core.smi" "" "priors/focused_A.prior" \
         "$EPOCHS" "$SAVEFREQ" "logs/d3/tb"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# Offline by default: nothing leaves this machine until `wandb sync` is run by hand.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-reinvent}"

RUN="logs/d3/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN" priors
echo "=============================================================="
echo "D3  run dir : $RUN"
echo "    GPU     : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo "    wandb   : $([ $USE_WANDB = 1 ] && echo "$WANDB_MODE" || echo disabled)"
echo "=============================================================="

count() { grep -vc '^#' "$1"; }

# ---------------------------------------------------------------- phase 1
declare -A BEST_EPOCH
for arm in "${ARMS[@]}"; do
  label="TL-A"; [ "$arm" = core_B ] && label="TL-B"
  echo
  echo "== phase 1: $label diagnostic, $N_FOLDS scaffold folds, up to $DIAG_EPOCHS epochs =="
  epochs=()
  for i in $(seq 1 $N_FOLDS); do
    tr="data/folds/actives_${arm}_fold${i}_train.smi"
    va="data/folds/actives_${arm}_fold${i}_valid.smi"
    d="$RUN/diag_${arm}_fold${i}"; mkdir -p "$d"
    render "$tr" "validation_smiles_file = \"$va\"" "$d/focused.prior" \
           "$DIAG_EPOCHS" 1 "$d/tb" > "$d/tl.toml"
    reinvent -l "$d/tl.log" "$d/tl.toml" > "$d/stdout.txt" 2>&1 || {
      echo "  fold $i FAILED - see $d/tl.log"; tail -5 "$d/tl.log"; exit 1; }
    best=$(grep -oP 'was at epoch \K\d+' "$d/tl.log" | tail -1 || true)
    warn=$(grep -c 'No clear minimum' "$d/tl.log" || true)
    printf "  fold %d: best epoch %-3s %s\n" "$i" "${best:-?}" \
      "$([ "${warn:-0}" -gt 0 ] && echo '(no clear minimum - budget too small)')"
    [ -n "$best" ] && epochs+=("$best")
    if [ $USE_WANDB = 1 ]; then
      python3 scripts/wandb_sync.py --arm "$label" --job diagnostic --fold "$i" \
        --epochs "$DIAG_EPOCHS" --n-train "$(count "$tr")" --n-valid "$(count "$va")" \
        --tb-logdir "$d/tb" --log "$d/tl.log" || echo "  (wandb sync failed, continuing)"
    fi
    # 20 checkpoints per run at ~22 MB each; the log has what we needed from them
    find "$d" -name 'focused.prior*' -delete
  done
  if [ ${#epochs[@]} -eq 0 ]; then
    echo "  no fold reported a best epoch; falling back to the blueprint's 10"
    BEST_EPOCH[$arm]=10
  else
    read -r mean lo hi <<< "$(printf '%s\n' "${epochs[@]}" | awk '
      {s+=$1; v[NR]=$1; if(NR==1||$1<m)m=$1; if($1>M)M=$1}
      END{printf "%d %d %d", (s/NR)+0.5, m, M}')"
    BEST_EPOCH[$arm]=$mean
    echo "  -> best epoch across folds: mean $mean, range $lo-$hi  (report mean and range, never one fold)"
  fi
done

# ---------------------------------------------------------------- phase 2
echo
echo "== phase 2: production priors, full sets, no validation file =="
for arm in "${ARMS[@]}"; do
  label="TL-A"; [ "$arm" = core_B ] && label="TL-B"
  out="priors/focused_$([ "$arm" = core ] && echo A || echo B).prior"
  d="$RUN/prod_${arm}"; mkdir -p "$d"
  e=${BEST_EPOCH[$arm]}
  prod_epochs="$e"; prod_savefreq="$e"; run_name=""
  if [ "$MODE" = "production-sweep" ] && [ "$arm" = core ]; then
    # The production sweep for TL-A ("a-prime"): fixed 200 epochs, checkpointing
    # every 20 so validation NLL actually lands on every checkpoint, not just the
    # last one (see the --mode block above).
    prod_epochs="$EPOCHS"; prod_savefreq="$SAVEFREQ"; run_name="tl-a-prime-sweep"
  fi
  render "data/actives_${arm}.smi" "" "$out" "$prod_epochs" "$prod_savefreq" "$d/tb" > "$d/tl.toml"
  reinvent -l "$d/tl.log" "$d/tl.toml" > "$d/stdout.txt" 2>&1 || {
    echo "  $label FAILED - see $d/tl.log"; tail -5 "$d/tl.log"; exit 1; }
  echo "  $label: $(count "data/actives_${arm}.smi") molecules, $prod_epochs epochs (save every $prod_savefreq) -> $out"
  if [ $USE_WANDB = 1 ]; then
    python3 scripts/wandb_sync.py --arm "$label" --job production --epochs "$prod_epochs" \
      --n-train "$(count "data/actives_${arm}.smi")" --tb-logdir "$d/tb" \
      --log "$d/tl.log" ${run_name:+--run-name "$run_name"} || echo "  (wandb sync failed, continuing)"
  fi
done

# ---------------------------------------------------------------- phase 3
echo
echo "== phase 3: D3 acceptance checks =="
python3 scripts/d3_report.py \
  --priors "TL-A=priors/focused_A.prior" "TL-B=priors/focused_B.prior" \
  --n "$SAMPLE_N" $([ $USE_WANDB = 1 ] && echo --wandb) | tee "$RUN/d3_report.txt"

echo
echo "D3 complete. Report: $RUN/d3_report.txt"
[ $USE_WANDB = 1 ] && [ "$WANDB_MODE" = offline ] && \
  echo "wandb runs are local only. Upload with:  wandb sync wandb/offline-run-*"
