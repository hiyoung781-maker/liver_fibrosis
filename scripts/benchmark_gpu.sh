#!/bin/bash
# Measure what a REINVENT4 run actually costs on this machine, so the choice between
# a local TITAN Xp and a queued A100 rests on numbers rather than on FLOPS ratios.
#
#   bash scripts/benchmark_gpu.sh
#   CUDA_VISIBLE_DEVICES=1 RL_STEP_SECONDS=1.35 bash scripts/benchmark_gpu.sh
#
# Why the answer is not obvious: the de novo prior is a 3-layer LSTM with hidden size
# 512, about 5.8M parameters. Generation is autoregressive over up to 256 timesteps,
# so it is bound by kernel-launch latency rather than by FLOPS, and every generated
# molecule is then scored by RDKit on a single CPU thread. Both halves are timed
# separately below.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-reinvent}"

WORK="$(mktemp -d ./logs/bench.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

echo "=============================================================="
echo "host    : $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "CPU     : $(nproc) threads"
echo "=============================================================="

RL_STEP_SECONDS="${RL_STEP_SECONDS:-0}" python - "$WORK" <<'INNER'
import csv, os, statistics, subprocess, sys, time
from pathlib import Path

work = Path(sys.argv[1])
import torch
print(f"\ntorch {torch.__version__}  CUDA build {torch.version.cuda}")
print(f"arch list: {torch.cuda.get_arch_list()}")
if not torch.cuda.is_available():
    sys.exit("FAIL: no CUDA device visible")
props = torch.cuda.get_device_properties(0)
arch = f"sm_{props.major}{props.minor}"
print(f"device: {props.name}  {arch}  {props.total_memory/1024**3:.0f} GB")
if arch not in torch.cuda.get_arch_list():
    print(f"  no {arch} cubin in this build; CUDA runs the sm_{props.major}0 one "
          f"(binary compatibility holds upward within a major version)")

def sample(n):
    cfg = work / f"sample_{n}.toml"
    cfg.write_text(
        'run_type = "sampling"\ndevice = "cuda:0"\n\n[parameters]\n'
        'model_file = "REINVENT4/priors/reinvent.prior"\n'
        f'output_file = "{work}/out_{n}.csv"\n'
        f'num_smiles = {n}\nunique_molecules = true\n'
    )
    t0 = time.perf_counter()
    subprocess.run(["reinvent", "-l", str(work / f"s{n}.log"), str(cfg)],
                   check=True, capture_output=True)
    return time.perf_counter() - t0

print("\n== generation (sampling from the prior) ==")
sample(200)                                   # warm the CUDA context
points = [(n, sample(n)) for n in (2000, 10000)]
for n, dt in points:
    print(f"  {n:6d} molecules  {dt:7.1f} s")

# Two points, two unknowns: t = startup + n/rate. Averaging the two apparent rates
# instead would fold ~19 s of process startup into the throughput and understate it
# by more than an order of magnitude.
(n1, t1), (n2, t2) = points
rate = (n2 - n1) / (t2 - t1)
startup = t1 - n1 / rate
print(f"  -> {startup:.1f} s startup per `reinvent` call, {rate:.0f} molecules/s once running")

print("\n== scoring (RDKit, CPU) ==")
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

smis = [v for r in csv.DictReader(open(work / "out_10000.csv"))
        for k, v in r.items() if "smiles" in k.lower() and v][:5000]
refs = [l.split()[0] for l in open("data/similarity_refs.smi")
        if not l.startswith("#") and l.strip()]
fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
ref_fps = [fpgen.GetCountFingerprint(Chem.MolFromSmiles(s)) for s in refs]
acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")

t0, scored = time.perf_counter(), 0
for s in smis:
    m = Chem.MolFromSmiles(s)
    if m is None:
        continue
    fp = fpgen.GetCountFingerprint(m)
    [DataStructs.TanimotoSimilarity(fp, r) for r in ref_fps]
    m.HasSubstructMatch(acid)
    Descriptors.TPSA(m); Crippen.MolLogP(m); Descriptors.NumRotatableBonds(m)
    Descriptors.MolWt(m); QED.qed(m); MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    scored += 1
score_rate = scored / (time.perf_counter() - t0)
print(f"  {scored} molecules, 1 thread, the Stage-2 component set: {score_rate:.0f} mol/s")
print(f"  ({os.cpu_count()} threads on this host, but REINVENT scores single-threaded)")

print("\n== projected wall time for the blueprint ==")
step = float(os.environ.get("RL_STEP_SECONDS") or 0)
if not step:
    print("  RL step time is not extrapolated from the numbers above - a step interleaves")
    print("  generation, scoring, the diversity filter and a backward pass. Measure it:")
    print("    W=$(mktemp -d ./logs/rlbench.XXXXXX)")
    print("    sed -e \"s|{PREFIX}|$W/summary|; s|{CHKPT}|$W/b.chkpt|; s|{STEPS}|20|\" \\")
    print("        configs/rl_benchmark.toml.in > \"$W/rl20.toml\"")
    print("    /usr/bin/time -f 'WALL %e s' reinvent -l \"$W/rl.log\" \"$W/rl20.toml\"")
    print(f"  then re-run with RL_STEP_SECONDS=(WALL - {startup:.0f}) / 20")
    print(f"\n  sampling 30,000 molecules: {startup + 30000/rate:.0f} s")
else:
    plan = [("TL  10 epochs x 29 molecules", startup),
            ("RL stage 1  200 steps", startup + 200 * step),
            ("RL stage 2  400 steps", startup + 400 * step),
            ("sampling  30,000 molecules", startup + 30000 / rate),
            ("scoring the benchmark panel", startup)]
    total = sum(t for _, t in plan)
    for label, t in plan:
        print(f"  {label:34s} {t/60:6.1f} min")
    print(f"  {'TOTAL (one arm, end to end)':34s} {total/60:6.1f} min")
    print(f"  {'three arms (TL-A, TL-B, TL-C)':34s} {3*total/60:6.1f} min")
INNER
