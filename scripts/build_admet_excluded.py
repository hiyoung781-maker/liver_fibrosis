"""The §5.5 ADMET window's rejects, as a docking input set.

§5.5 moves the ADMET window BEFORE docking, so 13,767 survivors become 334
docked ligands. That ordering buys a 40x docking saving, but it also means the
13,433 rejects were never scored against the receptor - so the campaign cannot
say whether any of them would have cleared the §5.3 binding-mode gate and the
affinity cut. The rejection is pre-registered and defensible; its FALSE-NEGATIVE
RATE is simply unmeasured.

This writes that unmeasured set out so it can be docked and the rate measured.
Labels are preserved from survivors.smi rather than reassigned: a reject's
identity must be the same string before and after this split, or the measured
false negatives cannot be traced back to which ADMET axis rejected them.
"""

from __future__ import annotations

import argparse
import os
import sys

from novelty import load_smi


def docked_labels(pdbqt_dir: str) -> set[str]:
    """Labels already prepared for docking, from the ligand pdbqt filenames."""
    if not os.path.isdir(pdbqt_dir):
        raise FileNotFoundError(pdbqt_dir)
    return {f[:-6] for f in os.listdir(pdbqt_dir) if f.endswith(".pdbqt")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--survivors", default="data/v4_TL-B/survivors.smi")
    parser.add_argument("--docked-dir", default="docking/v4/pdbqt")
    parser.add_argument("--out", default="data/v4_TL-B/admet_excluded.smi")
    args = parser.parse_args(argv)

    done = docked_labels(args.docked_dir)
    rows = load_smi(args.survivors)
    # load_smi yields (smiles, label); a survivor without a label would make the
    # reject untraceable, so fail loudly instead of inventing one.
    missing = [i for i, (_, label) in enumerate(rows) if not label]
    if missing:
        raise ValueError(f"{args.survivors}: {len(missing)} rows carry no label")

    kept = [(s, l) for s, l in rows if l not in done]
    lines = [
        "# The ADMET window's rejects - molecules the pre-docking ADMET gate",
        "# (section 5.5) removed, which were therefore never docked.",
        f"# from: {args.survivors} ({len(rows)} survivors)",
        f"# minus already-docked labels in {args.docked_dir} ({len(done)})",
        f"# excluded set: {len(kept)}",
        "# Labels are carried over unchanged from survivors.smi.",
    ]
    lines += [f"{s}\t{l}" for s, l in kept]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"{args.survivors} -> {args.out}")
    print(f"  survivors      {len(rows)}")
    print(f"  already docked {len(done & {l for _, l in rows})}")
    print(f"  excluded       {len(kept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
