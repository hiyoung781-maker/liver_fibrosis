#!/usr/bin/env python3
"""How do the D3 acceptance metrics move as transfer learning runs longer?

The fold diagnostic optimises validation NLL, but NLL is not what this project
wants. NLL keeps falling because the model reproduces the curated actives ever more
faithfully - and those actives sit at TPSA ~145 and QED ~0.40, which is precisely the
space section 0 is trying to escape. Two points already showed the trade:

    epoch 19 -> carboxylic acid 17.0%, TPSA 85.7, QED 0.542, 908 scaffolds/1000
    epoch 37 -> carboxylic acid 26.3%, TPSA 90.5, QED 0.531, 886 scaffolds/1000

So this sweeps the metrics that actually decide, rather than the likelihood. One
training run to 200 epochs writes a checkpoint every 20 (REINVENT4 names them
`{output}.{epoch}.chkpt`), and each checkpoint is sampled. That costs one training
run plus eleven samplings instead of ten more fold runs.

    python3 scripts/epoch_sweep.py --max-epochs 200 --every 20 --n 1000 [--wandb]
    python3 scripts/epoch_sweep.py --nll-probe        # does validation NLL ever turn up?
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import d3_report  # noqa: E402  - sample() and describe() are the same code D3 used

from rdkit import Chem  # noqa: E402


def render(smiles: str, validation: str, output: str, epochs: int,
           savefreq: int, tb: str) -> str:
    tpl = (ROOT / "configs" / "tl.toml.in").read_text()
    return (tpl.replace("{SMILES}", smiles).replace("{VALIDATION}", validation)
               .replace("{OUTPUT}", output).replace("{EPOCHS}", str(epochs))
               .replace("{SAVEFREQ}", str(savefreq)).replace("{TB_LOGDIR}", tb))


def train(cfg_text: str, work: Path, tag: str) -> Path:
    cfg = work / f"{tag}.toml"
    cfg.write_text(cfg_text)
    log = work / f"{tag}.log"
    subprocess.run(["reinvent", "-l", str(log), str(cfg)], check=True, capture_output=True)
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--set", default="data/actives_core.smi", help="TL-A's full set")
    ap.add_argument("--refs", default="data/actives_core.smi")
    ap.add_argument("--nll-probe", action="store_true",
                    help="also run one fold to --max-epochs with validation")
    ap.add_argument("--probe-fold", default="data/folds/actives_core_fold3")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    out_root = ROOT / "logs" / "sweep"
    out_root.mkdir(parents=True, exist_ok=True)
    work = out_root / subprocess.run(["date", "+%Y%m%d-%H%M%S"],
                                     capture_output=True, text=True).stdout.strip()
    work.mkdir()
    print(f"work dir: {work}\n")

    ref_smiles = [l.split("\t")[0] for l in (ROOT / args.refs).open()
                  if not l.startswith("#") and l.strip()]
    refs = [d3_report._FPGEN.GetCountFingerprint(Chem.MolFromSmiles(s)) for s in ref_smiles]

    # ---------------------------------------------------------------- train once
    prior = work / "sweep.prior"
    print(f"training TL-A on the full set to {args.max_epochs} epochs "
          f"(checkpoint every {args.every}) ...", flush=True)
    train(render(args.set, "", str(prior), args.max_epochs, args.every, str(work / "tb")),
          work, "train")

    checkpoints = [(int(m.group(1)), p) for p in work.glob("sweep.prior.*.chkpt")
                   if (m := re.search(r"\.(\d+)\.chkpt$", p.name))]
    checkpoints.sort()
    print(f"  {len(checkpoints)} checkpoints: {[e for e, _ in checkpoints]}\n")

    # ---------------------------------------------------------------- sample each
    rows = []
    with tempfile.TemporaryDirectory(dir=out_root) as tmp:
        sdir = Path(tmp)
        base = d3_report.describe(
            d3_report.sample(ROOT / "REINVENT4/priors/reinvent.prior", args.n, sdir, "base"),
            refs)
        rows.append((0, base))
        print(f"  epoch {0:4d} (untrained prior): "
              f"acid {base['carboxylic_acid_pct']:5.1f}%  scaffolds {base['unique_scaffolds']:4d}")
        for epoch, path in checkpoints:
            r = d3_report.describe(d3_report.sample(path, args.n, sdir, f"e{epoch}"), refs)
            rows.append((epoch, r))
            print(f"  epoch {epoch:4d}: acid {r['carboxylic_acid_pct']:5.1f}%  "
                  f"arg {r['arg_mimic_pct']:5.1f}%  scaffolds {r['unique_scaffolds']:4d}  "
                  f"TPSA {r['tpsa_median']:5.1f}  QED {r['qed_median']:.3f}  "
                  f"NN {r['nn_tanimoto_mean']:.3f}", flush=True)

    keys = {"carboxylic_acid_pct": 1, "arg_mimic_pct": 1, "unique_scaffolds": 0,
            "tpsa_median": 1, "qed_median": 3, "nn_tanimoto_mean": 3, "mw_median": 1}
    table = ["| epoch | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for epoch, r in rows:
        table.append(f"| {epoch} | " +
                     " | ".join(f"{r.get(k, float('nan')):.{p}f}" for k, p in keys.items()) + " |")
    report = "\n".join(table)
    print("\n" + report)
    (work / "sweep.md").write_text(report + "\n")

    # ---------------------------------------------------------------- NLL probe
    if args.nll_probe:
        print(f"\nNLL probe: one fold to {args.max_epochs} epochs with validation ...",
              flush=True)
        log = train(render(f"{args.probe_fold}_train.smi",
                           f'validation_smiles_file = "{args.probe_fold}_valid.smi"',
                           str(work / "probe.prior"), args.max_epochs, args.every,
                           str(work / "tb_probe")), work, "probe")
        text = log.read_text(errors="replace")
        m = re.search(r"Best validation loss \(([\d.]+)\) was at epoch (\d+)", text)
        warned = "No clear minimum found in validation loss" in text
        if m:
            print(f"  best validation NLL {m.group(1)} at epoch {m.group(2)}"
                  f"{'  (STILL DESCENDING at the budget limit)' if warned else '  (interior minimum)'}")
            if warned:
                print("  -> validation NLL does not turn up in this range. With "
                      "randomize_all_smiles the\n     model sees fresh strings every epoch, so "
                      "the NLL stopping rule does not bite\n     here; the sweep above is the "
                      "criterion that does.")
        for p in work.glob("probe.prior*"):
            p.unlink()

    for p in work.glob("sweep.prior.*.chkpt"):
        p.unlink()

    if args.wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "avb1-denovo"),
                         group="D3-TL-A", job_type="epoch-sweep", name="epoch-sweep",
                         config={"max_epochs": args.max_epochs, "every": args.every,
                                 "n_samples": args.n, "set": args.set}, reinit=True)
        for epoch, r in rows:
            run.log({f"sweep/{k}": v for k, v in r.items()}, step=epoch)
        run.finish()
        print(f"\n  wandb [{os.environ.get('WANDB_MODE', 'online')}]: sweep logged")

    print(f"\nreport: {work}/sweep.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
