"""Run Uni-Dock over the section 8.2 survivors, reproducing the validated box.

Uni-Dock v1.1.3's `--scoring` defaults to `vina`, which is why the redocking result
recorded with smina (0.63 A RMSD) still characterizes this method: same scoring
function, different implementation and hardware. What differs is the search
(GPU-parallel Monte Carlo), so the crystal ligand is still worth one run here - as a
check on the build and the box, not as a revalidation.

THE BOX IS THE PART THAT CAN SILENTLY DIVERGE. smina was given
`--autobox_ligand ligand_ref.sdf --autobox_add 6`; Uni-Dock has no equivalent
(`--autobox` works only with --score_only/--local_only), so the six numbers must be
computed here to match. box_from_ligand reproduces smina's rule exactly: the box
spans [min - pad, max + pad] per axis, so size = extent + 2*pad. Getting this wrong
would move the search space without any error message, and the geometry filter would
report a pass rate for a different question than the one section 8.5 asks.

Search parameters are pinned to the validated run: exhaustiveness 16, num_modes 20,
seed 42. Section 8.5b's whole design is to generate many poses and select on
geometry, because the scoring function has no metal term - so `--search_mode fast`
and a lower exhaustiveness are not tempting trade-offs here, they attack the
compensation for a known gap.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import numpy as np
from rdkit import Chem, RDLogger

__all__ = ["DEFAULTS", "box_from_ligand", "build_command"]

RDLogger.DisableLog("rdApp.*")

# Pinned to the smina redocking run that section 8.5(a) validated.
DEFAULTS = {
    "autobox_add": 6.0,
    "exhaustiveness": 16,
    "num_modes": 20,
    "seed": 42,
    "scoring": "vina",
}


def box_from_ligand(sdf_path: str, padding: float = DEFAULTS["autobox_add"]) -> dict:
    """smina's --autobox_ligand rule: bounding box of the ligand, padded each side.

    Returns center_x/y/z and size_x/y/z, the six values Uni-Dock needs.
    """
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        raise ValueError(f"{sdf_path}: could not be read as a molecule")
    xyz = mol.GetConformer().GetPositions()
    lo, hi = xyz.min(axis=0) - padding, xyz.max(axis=0) + padding
    center, size = (lo + hi) / 2.0, hi - lo
    return {
        "center_x": float(center[0]), "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(size[0]), "size_y": float(size[1]),
        "size_z": float(size[2]),
    }


def build_command(receptor: str, ligand_index: str, out_dir: str, box: dict,
                  exhaustiveness: int = DEFAULTS["exhaustiveness"],
                  num_modes: int = DEFAULTS["num_modes"],
                  seed: int = DEFAULTS["seed"],
                  scoring: str = DEFAULTS["scoring"],
                  search_mode: str | None = None) -> list[str]:
    """The Uni-Dock argv. Separated from execution so a test can read it."""
    cmd = [
        "unidock",
        "--receptor", receptor,
        "--ligand_index", ligand_index,
        "--dir", out_dir,
        "--scoring", scoring,
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--seed", str(seed),
    ]
    for key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"):
        cmd += [f"--{key}", f"{box[key]:.3f}"]
    if search_mode:
        # Mutually exclusive with explicit exhaustiveness in spirit: search_mode
        # overrides it with a preset. Recorded, not recommended - see the docstring.
        cmd += ["--search_mode", search_mode]
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receptor", default="docking/receptor.pdbqt")
    parser.add_argument("--ligand-index", default="docking/ligands_pdbqt/ligands.txt")
    parser.add_argument("--out-dir", default="docking/poses")
    parser.add_argument("--autobox-ligand", default="docking/ligand_ref.sdf")
    parser.add_argument("--autobox-add", type=float,
                        default=DEFAULTS["autobox_add"])
    parser.add_argument("--exhaustiveness", type=int,
                        default=DEFAULTS["exhaustiveness"])
    parser.add_argument("--num-modes", type=int, default=DEFAULTS["num_modes"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--search-mode", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command and the box, run nothing")
    args = parser.parse_args(argv)

    box = box_from_ligand(args.autobox_ligand, args.autobox_add)
    print("box (reproducing smina --autobox_ligand --autobox_add "
          f"{args.autobox_add:g}):")
    for key, value in box.items():
        print(f"  {key:9s} {value:9.3f}")

    cmd = build_command(args.receptor, args.ligand_index, args.out_dir, box,
                        args.exhaustiveness, args.num_modes, args.seed,
                        search_mode=args.search_mode)
    print("\ncommand:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return 0
    if shutil.which("unidock") is None:
        print("\nunidock not on PATH - activate the unidock env", file=sys.stderr)
        return 1

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    print("\nrunning...", flush=True)
    result = subprocess.run(cmd)
    print(f"exit status: {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
