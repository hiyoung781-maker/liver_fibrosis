#!/usr/bin/env python3
"""Mirror one REINVENT4 transfer-learning run into Weights & Biases.

REINVENT4 is left unmodified: it already writes its NLL curves and valid-SMILES
fraction to TensorBoard, and wandb ingests that directly with sync_tensorboard.
Editing the vendored v4.5.11 tree would blur the version pin that the whole K-BDS
environment rests on (see TRANSFER.md).

Offline by default. Nothing leaves this machine until someone runs `wandb sync`,
which matters because the generated molecules are the project's unpublished output.
Override with WANDB_MODE=online.

    python3 scripts/wandb_sync.py --arm TL-A --job diagnostic --fold 3 \
        --epochs 20 --n-train 23 --tb-logdir logs/d3/... --log logs/d3/....log
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

PROJECT = os.environ.get("WANDB_PROJECT", "avb1-denovo")


def parse_tl_log(path: Path) -> dict:
    """Pull the figures REINVENT4 reports only to its text log.

    learning.py logs the best epoch itself, and warns when the best epoch is the
    last one - which means the minimum has not been reached and the epoch budget
    was too small. That warning is the one thing the curve alone will not tell you.
    """
    out: dict[str, object] = {}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    m = re.search(r"Best validation loss \(([\d.]+)\) was at epoch (\d+)", text)
    if m:
        out["best_validation_nll"] = float(m.group(1))
        out["best_epoch"] = int(m.group(2))
    out["no_clear_minimum"] = "No clear minimum found in validation loss" in text
    m = re.search(r"Read (\d+) input SMILES", text)
    if m:
        out["smiles_read"] = int(m.group(1))
    return out


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="TL-A or TL-B")
    ap.add_argument("--job", required=True, choices=["diagnostic", "production"])
    ap.add_argument("--fold", type=int, default=0, help="0 for production runs")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--n-valid", type=int, default=0)
    ap.add_argument("--tb-logdir", required=True)
    ap.add_argument("--log", required=True, help="the reinvent -l log file")
    args = ap.parse_args()

    import wandb

    name = (f"{args.arm}-fold{args.fold}" if args.job == "diagnostic"
            else f"{args.arm}-production")
    summary = parse_tl_log(Path(args.log))

    run = wandb.init(
        project=PROJECT,
        group=f"D3-{args.arm}",
        job_type=args.job,
        name=name,
        sync_tensorboard=True,
        config={
            "arm": args.arm,
            "job": args.job,
            "fold": args.fold,
            "num_epochs": args.epochs,
            "n_train": args.n_train,
            "n_valid": args.n_valid,
            "batch_size": 32,
            "randomize_all_smiles": True,
            "prior": "REINVENT4 v4.5.11 reinvent.prior",
            "split": "scaffold-disjoint 5-fold, seed 20260925",
            "commit": git_commit(),
        },
        reinit=True,
    )
    for key, value in summary.items():
        run.summary[key] = value

    # sync_tensorboard reads the event files when the run closes, so the directory
    # has to still be there - run this after reinvent exits, before any cleanup.
    tb = Path(args.tb_logdir)
    if not tb.exists():
        print(f"  warning: {tb} does not exist; no curves will be attached")
    run.finish()

    printed = ", ".join(f"{k}={v}" for k, v in summary.items()) or "no summary parsed"
    print(f"  wandb [{os.environ.get('WANDB_MODE', 'online')}] {name}: {printed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
