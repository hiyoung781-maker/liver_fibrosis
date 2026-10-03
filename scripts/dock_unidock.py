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
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

__all__ = ["DEFAULTS", "box_from_ligand", "build_command", "split_index",
           "parse_unidock_pdbqt", "run_multi_gpu", "filter_done"]

OUT_SUFFIX = "_out.pdbqt"

RDLogger.DisableLog("rdApp.*")

# `REMARK VINA RESULT:` carries three numbers; the first is the affinity in
# kcal/mol, the other two are Vina's own lower- and upper-bound RMSD from the
# best mode and are not used here.
#
# smina writes the SAME Vina score under a different remark -- `REMARK
# minimizedAffinity`, one number -- so both spellings are accepted. This is a
# naming difference, not a scoring one: smina is Vina 1.1.2's scoring function
# in a different implementation, which is why section 8.5(a)'s smina redocking
# (0.63 A) and Uni-Dock's (0.66 A) characterise each other. Accepting both
# keeps every downstream reader engine-agnostic, the same property this
# function was written to give validate_redock.
_VINA_RESULT = re.compile(
    r"REMARK\s+(?:VINA RESULT:|minimizedAffinity)\s*(-?\d+\.?\d*)")

# A score below this is not a weak or a strong binding estimate, it is a broken
# one, and it is read as a MISSING score rather than as a very good one.
#
# WHERE THE NUMBER COMES FROM, measured over all 52,809 poses of both arms. The
# strongest physical pose in the campaign is -10.340 kcal/mol and the weakest is
# +8.354. Below -15 there are 251 poses reaching -219.017, and every one of the
# 126 ligands holding them carries Meeko's macrocycle glue atoms (`CG0`/`G0`):
# pseudo-atoms inserted to open a ring, which Vina has no parameters for. The
# two distributions do not touch -- nothing at all lies between -10.340 and
# -21.9 -- so any floor in that gap gives the same answer and -15.0 sits in the
# middle of it. smina confirms the mechanism independently by refusing to parse
# those ligands at all ("CG0 is not a valid AutoDock type").
#
# IT COSTS THE SCORE, NEVER THE POSE, the same rule an absent energy remark
# follows. Dropping the pose would shrink the denominator of every rate
# downstream; dropping only the number lets a ligand still pass on its other
# poses, and lets one whose only passing pose is broken fail honestly for having
# no usable score.
AFFINITY_FLOOR = -15.0


def parse_unidock_pdbqt(text: str) -> list[dict]:
    """Poses from a Uni-Dock output PDBQT, shaped like parse_dlg's output.

    The redocking gate (spec 7.4) was written against AutoDock-GPU's .dlg.
    Reverting to Uni-Dock -- that section's own pre-registered consequence --
    means the same gate must read Uni-Dock output, so this returns the same
    {rank, affinity, pdbqt_block} records and validate_redock stays engine-
    agnostic.

    A pose whose energy remark is missing keeps the pose with affinity None,
    the same rule parse_dlg follows: a v1 regression put score extraction
    inside the structure-parsing try and took the pose count to zero. An
    absent score must cost the score, never the pose.
    """
    poses: list[dict] = []
    block: list[str] = []
    affinity = None
    inside = False
    for line in text.splitlines():
        if line.startswith("MODEL"):
            block, affinity, inside = [line], None, True
            continue
        if not inside:
            continue
        # THE FIRST ENERGY REMARK IN THE BLOCK IS THIS ENGINE'S, and a later
        # one is metadata it inherited. An engine writes its own remarks
        # straight after `MODEL n` and then copies whatever the input ligand
        # carried, so a pose re-docked from another run's output pose holds two:
        # smina's `minimizedAffinity` for THIS receptor first, then the
        # `VINA RESULT` of the run the ligand came from. Assigning on every
        # match let the inherited one win, and the alphaVbeta6 side then
        # reported alphaVbeta1's scores -- every selectivity value came out
        # exactly 0.000, which is how it was caught.
        if affinity is None:
            match = _VINA_RESULT.search(line)
            if match:
                affinity = float(match.group(1))
                if affinity < AFFINITY_FLOOR:
                    affinity = None
        block.append(line)
        if line.startswith("ENDMDL"):
            poses.append({"rank": len(poses) + 1, "affinity": affinity,
                          "pdbqt_block": "\n".join(block)})
            inside = False
    return poses

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


def filter_done(index_path: str, out_dir: str) -> dict:
    """The entries of `index_path` that `out_dir` holds no finished pose file for.

    UNI-DOCK HAS NO --resume, and the first pass is an 11-hour GPU run (62 on
    CPU): an interruption at hour ten would otherwise restart from zero. Because
    Uni-Dock names every output after its ligand (`<label>_out.pdbqt`), resuming
    is an index filter rather than engine state -- drop the ligands that already
    have an output, re-run the rest.

    THE MOST RECENTLY MODIFIED OUTPUT IS RE-QUEUED rather than counted as done.
    If the process died mid-write, that file is the one that was truncated, and a
    truncated PDBQT does not announce itself: parse_unidock_pdbqt would read the
    models that reached disk and report a ligand with fewer poses, which is
    indistinguishable from a ligand whose search genuinely found few. That
    distinction is not cosmetic here -- section 4.1's retention curve and section
    4.6's criterion (c) both read the pose count per ligand, so one silently
    short ligand biases the smina-switch decision. Re-docking one ligand costs
    about eight seconds and Uni-Dock overwrites by name, so nothing needs
    cleaning up first.

    Matching is on the EXACT label, never a prefix: `gen_01` and `gen_010` are
    different ligands, and a prefix test would retire both on one output.

    A missing or empty `out_dir` is a fresh start, not an error, so passing
    --resume to a first run is a no-op and the flag can always be present.
    """
    lines = [line.strip() for line in Path(index_path).read_text().splitlines()
             if line.strip()]
    if not lines:
        raise ValueError(f"{index_path}: no ligand paths")

    outputs = sorted(Path(out_dir).glob(f"*{OUT_SUFFIX}")) if Path(out_dir).is_dir() else []
    done = {path.name[: -len(OUT_SUFFIX)] for path in outputs}

    requeued = None
    if outputs:
        newest = max(outputs, key=lambda path: path.stat().st_mtime)
        requeued = newest.name[: -len(OUT_SUFFIX)]
        done.discard(requeued)

    remaining = [line for line in lines if Path(line).stem not in done]
    return {"remaining": remaining, "done": sorted(done), "requeued": requeued,
            "total": len(lines)}


def split_index(index_path: str, shards: int, out_dir: str) -> list[str]:
    """Split a --ligand_index file into `shards` files, round-robin. Returns paths.

    Round-robin, not contiguous: Uni-Dock groups ligands by size internally, so a
    contiguous shard that collected the large ligands would straggle. Interleaving
    gives every shard the same size mixture.

    A shard that would be empty is not written, so `shards` larger than the ligand
    count does not produce processes with nothing to do.
    """
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
    parser.add_argument("--resume", action="store_true",
                        help="dock only the ligands --out-dir has no pose file "
                             "for. Uni-Dock has no --resume of its own; this is "
                             "the index filter that stands in for one, because "
                             "outputs are named after their ligand. The most "
                             "recently modified output is re-docked in case the "
                             "interruption truncated it. Harmless on a fresh "
                             "out-dir, so it can always be passed")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command and the box, run nothing")
    args = parser.parse_args(argv)

    box = box_from_ligand(args.autobox_ligand, args.autobox_add)
    print("box (reproducing smina --autobox_ligand --autobox_add "
          f"{args.autobox_add:g}):")
    for key, value in box.items():
        print(f"  {key:9s} {value:9.3f}")

    # BEFORE the command is built, so what gets printed is what would run. The
    # filtered index is written to disk rather than held in memory: it is the
    # record of which ligands this restart is responsible for, and `--resume
    # --dry-run` is the way to read the counts before committing eleven hours.
    ligand_index = args.ligand_index
    if args.resume:
        state = filter_done(ligand_index, args.out_dir)
        print(f"\nresume: {state['total']} in index, "
              f"{len(state['done'])} already done")
        if state["requeued"]:
            print(f"        re-queued in case it was truncated: "
                  f"{state['requeued']}{OUT_SUFFIX}")
        if not state["remaining"]:
            print("        nothing left to dock")
            return 0
        os.makedirs(args.shard_dir, exist_ok=True)
        ligand_index = os.path.join(args.shard_dir, "resume_index.txt")
        Path(ligand_index).write_text("\n".join(state["remaining"]) + "\n")
        print(f"        {len(state['remaining'])} to dock -> {ligand_index}")

    cmd = build_command(args.receptor, ligand_index, args.out_dir, box,
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

    os.makedirs(args.out_dir, exist_ok=True)

    if args.gpus > 1:
        print("\nrunning...", flush=True)
        results = run_multi_gpu(
            args.receptor, ligand_index, args.out_dir, box, args.gpus,
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
