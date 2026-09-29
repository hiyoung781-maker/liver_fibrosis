"""How many poses each ligand actually got (spec §8.5b's own premise).

Section 8.5b's strategy is to generate MANY poses and select on geometry,
because the Vina scoring function has no metal coordination term and cannot
see the interaction that matters most here. That strategy needs poses to
select among, so the number per ligand is a result the lead report has to
carry, not an implementation detail.

The measurement is prompted by the redocking control: Uni-Dock returned 2
poses (neutral) and 4 (anion) at --num_modes 20 --energy_range 10.0
--min_rmsd 0.5, where AutoDock-GPU returned 20. Harmless there, since every
returned pose WAS the crystal pose, but it says the geometric selection has
little room to work. This module measures how much room it actually had
across the docked library.

A FILE IS NOT A POSE. Uni-Dock exits 0 having written an output file with no
MODEL block, and counting files reports that as success -- the same shape of
failure this project has now hit repeatedly, where a plausible-looking number
masked a real break. Ligands with zero poses are counted and named.

Deliberately no clustering parameters are touched here. The redocking gate
has already returned a verdict; changing --min_rmsd or --energy_range now
and re-measuring would be choosing parameters after seeing the answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

__all__ = ["census", "summarize"]


def _labels_and_poses(path: Path) -> tuple[int, float | None]:
    from dock_unidock import parse_unidock_pdbqt

    poses = parse_unidock_pdbqt(path.read_text())
    affinities = [p["affinity"] for p in poses if p["affinity"] is not None]
    return len(poses), (min(affinities) if affinities else None)


def census(pose_dir: str, pattern: str = "*_out.pdbqt") -> list[dict]:
    """One row per Uni-Dock output file: label, pose count, best affinity.

    The label is the filename with the `_out` suffix removed, which is how
    Uni-Dock names its outputs after the input ligand.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    rows = []
    for path in sorted(Path(pose_dir).glob(pattern)):
        poses, best = _labels_and_poses(path)
        rows.append({"label": path.stem.removesuffix("_out"),
                     "poses": poses, "best_affinity": best})
    return rows


def summarize(rows: list[dict]) -> dict:
    """Distribution of poses per ligand, and the ligands that got none."""
    counts = [row["poses"] for row in rows]
    no_poses = [row["label"] for row in rows if row["poses"] == 0]
    if not rows:
        return {"ligands": 0, "no_poses": [], "min_poses": None,
                "median_poses": None, "max_poses": None, "mean_poses": None,
                "histogram": {}}
    return {
        "ligands": len(rows),
        "no_poses": no_poses,
        "min_poses": min(counts),
        "median_poses": int(statistics.median(counts)),
        "max_poses": max(counts),
        "mean_poses": round(statistics.fmean(counts), 2),
        "histogram": dict(sorted(Counter(counts).items())),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pose-dir", required=True,
                    help="Uni-Dock --dir output containing *_out.pdbqt files.")
    p.add_argument("--expected", type=int, default=None,
                    help="Ligand count this directory should hold. A mismatch "
                         "is reported and exits nonzero.")
    p.add_argument("--out-csv", type=Path, default=None,
                    help="Per-ligand rows (label, poses, best_affinity).")
    p.add_argument("--out-json", type=Path, default=None,
                    help="The summary, for the lead report.")
    args = p.parse_args(argv)

    rows = census(args.pose_dir)
    summary = summarize(rows)

    print(f"{args.pose_dir}")
    print(f"  ligands        {summary['ligands']}")
    print(f"  poses/ligand   min {summary['min_poses']}  "
          f"median {summary['median_poses']}  max {summary['max_poses']}  "
          f"mean {summary['mean_poses']}")
    print(f"  histogram      {summary['histogram']}")
    print(f"  no poses       {len(summary['no_poses'])}")
    for label in summary["no_poses"][:20]:
        print(f"    {label}")
    if len(summary["no_poses"]) > 20:
        print(f"    ... and {len(summary['no_poses']) - 20} more")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["label", "poses", "best_affinity"])
            writer.writeheader()
            writer.writerows(rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n")

    status = 0
    if args.expected is not None and summary["ligands"] != args.expected:
        sys.stderr.write(f"expected {args.expected} ligands, found "
                         f"{summary['ligands']}\n")
        status = 1
    if summary["no_poses"]:
        sys.stderr.write(f"{len(summary['no_poses'])} ligand(s) produced no "
                         "poses; they are docking failures, not filter "
                         "failures, and must not be counted as either.\n")
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
