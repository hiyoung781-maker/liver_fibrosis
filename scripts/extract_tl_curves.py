"""Transfer-learning curves out of the TensorBoard event files, into a CSV.

Two environments, one each: tensorboard is installed where REINVENT runs and
matplotlib is not; matplotlib is installed where ADMET-AI runs and tensorboard
is not. Extracting to a text artifact decouples the figure from either, and a
reviewer can check the curve without a TensorBoard install.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

__all__ = ["extract"]

RUNS = {"A_Mean NLL loss_Training Loss": "training",
        "A_Mean NLL loss_Validation Loss": "validation"}


def extract(root: Path, pattern: str = "diag_core_B_fold*") -> list[dict]:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator)
    out: list[dict] = []
    for fold in sorted(root.glob(pattern)):
        name = fold.name.split("_")[-1]
        for sub, kind in RUNS.items():
            path = fold / "tb" / sub
            if not path.is_dir():
                continue
            acc = EventAccumulator(str(path))
            acc.Reload()
            tags = acc.Tags().get("scalars") or []
            if not tags:
                continue
            for event in acc.Scalars(tags[0]):
                out.append({"fold": name, "kind": kind,
                            "epoch": event.step, "nll": event.value})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("logs/d3/20261003-182445"))
    p.add_argument("--out", type=Path,
                   default=Path("results/v4_TL-B/tl_curves.csv"))
    args = p.parse_args(argv)

    rows = extract(args.root)
    if not rows:
        print(f"{args.root}: no scalar runs found", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "kind", "epoch", "nll"])
        writer.writeheader()
        writer.writerows(rows)
    folds = sorted({r["fold"] for r in rows})
    print(f"{args.out}  {len(rows)} rows, {len(folds)} folds: {', '.join(folds)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
