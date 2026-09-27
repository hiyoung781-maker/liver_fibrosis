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

MULTI-GPU. Uni-Dock uses one GPU per process: a single invocation pinned GPU 0 at
100% and 45 GB while the node's other seven sat at 3 MiB. The 8gpu partition charges
8 node-hours per wall hour no matter how many GPUs the job touches, so running one
process per GPU is 8x faster at identical allocation cost. --gpus N splits the ligand
index N ways and launches N processes, each with CUDA_VISIBLE_DEVICES set to its own
device; they share one --dir because Uni-Dock names outputs after each ligand, so
there are no collisions.

The split is round-robin rather than contiguous. Uni-Dock groups ligands internally
by size (small/medium/large) and batches by GPU memory, so a shard that happened to
collect all the large ligands would finish long after the others; interleaving gives
every shard the same size mixture.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import numpy as np
from rdkit import Chem, RDLogger

__all__ = ["DEFAULTS", "box_from_ligand", "build_command", "split_index",
           "run_multi_gpu"]

RDLogger.DisableLog("rdApp.*")

# Pinned to the smina redocking run that section 8.5(a) validated.
# Pinned to the smina redocking run that section 8.5(a) validated, EXCEPT for the two
# pose-retention knobs, which are widened from Uni-Dock's defaults (energy_range 3,
# min_rmsd 1) because those defaults return fewer poses than section 8.5b's design
# needs - it selects on geometry precisely because the scoring function has no metal
# term, so a handful of energy-ranked poses defeats it.
#
# These two knobs are NOT the explanation for the control failure, though. That was
# the first hypothesis and it was tested and refuted: widening energy_range 3 -> 10
# took the crystal ligand from 2 poses to 3, and all three sat within 0.11 kcal/mol
# and 0.86 A of each other. The search had converged, not been filtered.
#
# The live diagnosis is search depth. With the SAME scoring function, smina found
# -6.91 at exhaustiveness 16 while Uni-Dock found -6.30, and Uni-Dock's pose
# reproduced Ca501 (2.378 A) but not beta1-Asn224 (4.830 A against a crystal 2.63 A).
# A worse optimum from an identical function is a search problem, so exhaustiveness is
# the lever - see scripts/tune_control.sh, which sweeps it on the control alone. Until
# that sweep picks a level, EXHAUSTIVENESS BELOW IS THE UNVALIDATED VALUE.
DEFAULTS = {
    "autobox_add": 6.0,
    "exhaustiveness": 16,
    "num_modes": 20,
    "seed": 42,
    "scoring": "vina",
    "energy_range": 10.0,
    "min_rmsd": 0.5,
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
                  energy_range: float = DEFAULTS["energy_range"],
                  min_rmsd: float = DEFAULTS["min_rmsd"],
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
        "--energy_range", str(energy_range),
        "--min_rmsd", str(min_rmsd),
        "--seed", str(seed),
    ]
    for key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"):
        cmd += [f"--{key}", f"{box[key]:.3f}"]
    if search_mode:
        # Mutually exclusive with explicit exhaustiveness in spirit: search_mode
        # overrides it with a preset. Recorded, not recommended - see the docstring.
        cmd += ["--search_mode", search_mode]
    return cmd


def split_index(index_path: str, shards: int, out_dir: str) -> list[str]:
    """Split a --ligand_index file into `shards` files, round-robin. Returns paths.

    Round-robin, not contiguous: Uni-Dock groups ligands by size internally, so a
    contiguous shard that collected the large ligands would straggle. Interleaving
    gives every shard the same size mixture.

    A shard that would be empty is not written, so `shards` larger than the ligand
    count does not produce processes with nothing to do.
    """
    from pathlib import Path

    lines = [line.strip() for line in Path(index_path).read_text().splitlines()
             if line.strip()]
    if not lines:
        raise ValueError(f"{index_path}: no ligand paths")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard in range(shards):
        chunk = lines[shard::shards]
        if not chunk:
            continue
        path = out / f"ligands_gpu{shard}.txt"
        path.write_text("\n".join(chunk) + "\n")
        paths.append(str(path))
    return paths


def run_multi_gpu(receptor: str, index_path: str, out_dir: str, box: dict,
                  gpus: int, shard_dir: str, **kwargs) -> dict:
    """One Uni-Dock process per GPU, each pinned with CUDA_VISIBLE_DEVICES.

    Returns per-shard exit status. A nonzero status anywhere is reported rather than
    raised, because the other shards' poses are still valid output - the caller
    decides whether a partial library is usable.
    """
    import os

    shards = split_index(index_path, gpus, shard_dir)
    os.makedirs(out_dir, exist_ok=True)

    processes = []
    for device, shard in enumerate(shards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
        cmd = build_command(receptor, shard, out_dir, box, **kwargs)
        log = os.path.join(out_dir, f"unidock_gpu{device}.log")
        handle = open(log, "w")
        print(f"  GPU {device}: {sum(1 for _ in open(shard))} ligands -> {log}",
              flush=True)
        processes.append((device, shard, log, handle,
                          subprocess.Popen(cmd, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT)))

    results = {}
    for device, shard, log, handle, process in processes:
        status = process.wait()
        handle.close()
        results[device] = {"shard": shard, "log": log, "status": status}
        print(f"  GPU {device}: exit {status}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # The HYDROGENATED PDBQT, written by scripts/prepare_receptor_pdbqt.py. Measured:
    # Uni-Dock on the raw PDB scored -5.787 with the carboxylate 4.59 A off the
    # calcium; on a PDBQT built without hydrogens, -6.31 with the Asn224 contact lost;
    # on the hydrogenated PDBQT smina recovers -6.909 with both contacts, and smina and
    # Uni-Dock agree to within 0.1 kcal/mol given the same file.
    parser.add_argument("--receptor", default="docking/receptor.pdbqt")
    parser.add_argument("--ligand-index", default="docking/ligands_pdbqt/ligands.txt")
    parser.add_argument("--out-dir", default="docking/poses")
    parser.add_argument("--autobox-ligand", default="docking/ligand_ref.sdf")
    parser.add_argument("--autobox-add", type=float,
                        default=DEFAULTS["autobox_add"])
    parser.add_argument("--exhaustiveness", type=int,
                        default=DEFAULTS["exhaustiveness"])
    parser.add_argument("--num-modes", type=int, default=DEFAULTS["num_modes"])
    parser.add_argument("--energy-range", type=float,
                        default=DEFAULTS["energy_range"],
                        help="kcal/mol window of poses to keep. Uni-Dock's default of "
                             "3 left 2 poses for the crystal ligand where smina gave 20")
    parser.add_argument("--min-rmsd", type=float, default=DEFAULTS["min_rmsd"],
                        help="minimum RMSD between retained poses; Uni-Dock's default "
                             "of 1 clusters away poses the geometry filter needs")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--search-mode", default=None)
    parser.add_argument("--gpus", type=int, default=1,
                        help="run this many Uni-Dock processes, one per GPU. "
                             "Uni-Dock uses a single GPU per process, and the 8gpu "
                             "partition charges the same whether 1 or 8 are busy")
    parser.add_argument("--shard-dir", default="docking/ligand_shards")
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
                        energy_range=args.energy_range, min_rmsd=args.min_rmsd,
                        search_mode=args.search_mode)
    print("\ncommand:")
    print("  " + " ".join(cmd))

    if args.gpus > 1:
        print(f"\nmulti-GPU: {args.gpus} processes, one per device")

    if args.dry_run:
        return 0
    if shutil.which("unidock") is None:
        print("\nunidock not on PATH - activate the unidock env", file=sys.stderr)
        return 1

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    if args.gpus > 1:
        print("\nrunning...", flush=True)
        results = run_multi_gpu(
            args.receptor, args.ligand_index, args.out_dir, box, args.gpus,
            args.shard_dir, exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes, seed=args.seed,
            energy_range=args.energy_range, min_rmsd=args.min_rmsd,
            search_mode=args.search_mode)
        failed = [d for d, r in results.items() if r["status"] != 0]
        if failed:
            print(f"\nFAILED on GPU(s) {failed} - their logs are in {args.out_dir}."
                  "\nPoses from the other shards are still valid; decide whether a"
                  "\npartial library is usable before running the geometry filter.",
                  file=sys.stderr)
            return 1
        print("\nall shards finished")
        return 0

    print("\nrunning...", flush=True)
    result = subprocess.run(cmd)
    print(f"exit status: {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
