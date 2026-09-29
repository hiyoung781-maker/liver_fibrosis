"""Dock the section 3.4 benchmark panel with section 8.5's exact protocol.

Section 8.7's hard filter is "binding affinity better than PLN-1474", and PLN-1474 had
never been docked - section 7 scored it with the section 5.1 objective, and the docking
run covered only the 7,763 survivors plus the crystal control. So the filter had no
reference value.

EVERYTHING must match the production run or the comparison is meaningless: the same
engine (Uni-Dock, --scoring vina), the same hydrogenated receptor PDBQT, the same box,
the same exhaustiveness / num_modes / energy_range / min_rmsd / seed. Those values are
imported from dock_unidock rather than restated here, so the two cannot drift apart.
A cross-engine comparison would not do either - smina and Uni-Dock differ by about
0.16 kcal/mol on the same receptor, which is the width of a hard filter's margin.

The panel goes through the same preparation as the generated ligands: flattened SMILES,
one ETKDG conformer, MMFF relaxation, Meeko PDBQT. Labels are prefixed `PANEL_` so they
can never collide with a `gen_NNNNN` candidate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__all__ = ["PANEL_PREFIX", "prepare_panel_ligands"]

RDLogger.DisableLog("rdApp.*")

PANEL_PREFIX = "PANEL_"


def prepare_panel_ligands(out_dir: str, panel_smi: str = "data/benchmark_panel.smi",
                          deprotonate: bool = True) -> dict:
    """Embed and convert the panel to PDBQT, returning counts and the index path.

    Uses the same embed() and prepare_one() the generated ligands went through, so a
    difference in the docking result cannot come from a difference in ligand prep.

    That invariant is why `deprotonate` defaults to True. The redocking gate
    adopted the anion (spec 7.4, both Uni-Dock arms passing), the generated
    library is prepared deprotonated, and the section 9.1 affinity filter is a
    PLN-1474 RELATIVE comparison -- a reference in a different charge state
    from the candidates is not a comparison at all. Pass deprotonate=False
    only if the campaign adopts the neutral state, which the neutral arm's
    0.66 A also permits; then the library must switch with it.

    A panel member with no carboxylic acid is written unchanged rather than
    dropped: the panel is scored, never trained on, and losing a reference
    would silently shrink the comparison.
    """
    from deprotonate_acids import deprotonate as deprotonate_mol
    from ligands_to_pdbqt import prepare_one
    from novelty import load_smi
    from prepare_ligands import embed

    os.makedirs(out_dir, exist_ok=True)
    written, failures = [], []
    for smiles, name in load_smi(panel_smi):
        label = f"{PANEL_PREFIX}{name}"
        mol = embed(smiles)
        if mol is None:
            failures.append(label)
            continue
        if deprotonate:
            try:
                mol = deprotonate_mol(mol)
            except ValueError:
                pass  # no carboxylic acid; keep the reference as it is
        mol.SetProp("_Name", label)
        text = prepare_one(mol)
        if text is None:
            failures.append(label)
            continue
        path = os.path.join(out_dir, f"{label}.pdbqt")
        with open(path, "w") as handle:
            handle.write(text)
        written.append(os.path.abspath(path))

    index = os.path.join(out_dir, "ligands.txt")
    with open(index, "w") as handle:
        handle.write("\n".join(written) + ("\n" if written else ""))
    return {"written": len(written), "failures": failures, "index": index}


def main(argv: list[str] | None = None) -> int:
    from dock_unidock import DEFAULTS, box_from_ligand, build_command

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", default="data/benchmark_panel.smi")
    parser.add_argument("--no-deprotonate", action="store_true",
                        help="Prepare the panel as neutral acids. The default is "
                             "deprotonated, matching the anion state the redocking "
                             "gate adopted and the state the generated library is "
                             "prepared in; the section 9.1 affinity filter only "
                             "compares if reference and candidates match.")
    parser.add_argument("--receptor", default="docking/receptor.pdbqt")
    parser.add_argument("--autobox-ligand", default="docking/ligand_ref.sdf")
    parser.add_argument("--ligand-dir", default="docking/panel_pdbqt")
    parser.add_argument("--pose-dir", default="docking/panel_poses")
    parser.add_argument("--out-sdf", default="docking/panel_poses.sdf")
    parser.add_argument("--out-csv", default="results/panel_geometry.csv")
    parser.add_argument("--gpu", default="0",
                        help="CUDA_VISIBLE_DEVICES for this run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    print("preparing panel ligands (same pipeline as the generated set)")
    prep = prepare_panel_ligands(args.ligand_dir, args.panel,
                                 deprotonate=not args.no_deprotonate)
    print(f"  written  {prep['written']}")
    if prep["failures"]:
        print(f"  FAILED   {', '.join(prep['failures'])}")
        print("  A panel compound that cannot be prepared cannot serve as the hard")
        print("  filter's reference; fix it before proceeding.")
        return 1

    box = box_from_ligand(args.autobox_ligand, DEFAULTS["autobox_add"])
    cmd = build_command(args.receptor, prep["index"], args.pose_dir, box)
    print("\nprotocol (imported from dock_unidock, identical to the production run):")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return 0
    if shutil.which("unidock") is None:
        print("\nunidock not on PATH - activate the unidock env", file=sys.stderr)
        return 1

    os.makedirs(args.pose_dir, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    print(f"\nrunning on GPU {args.gpu} ...", flush=True)
    status = subprocess.run(cmd, env=env).returncode
    if status != 0:
        print(f"unidock exited {status}", file=sys.stderr)
        return status

    from pose_geometry import filter_poses
    from poses_to_sdf import convert_dir

    converted = convert_dir(args.pose_dir, args.out_sdf)
    print(f"\nposes -> {args.out_sdf}: {converted['poses']} from "
          f"{converted['files']} files")
    if converted["failed_files"]:
        print(f"  failed: {', '.join(converted['failures'])}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    counts = filter_poses(args.out_sdf, "docking/receptor.pdb", args.out_csv)
    print(f"geometry -> {args.out_csv}")
    for key, value in counts.items():
        print(f"  {key:16s} {value}")

    # Two columns, because they differ and only one of them is the threshold. The
    # candidates are evaluated on their best PASSING pose - a pose that misses the
    # anchors is not a binding mode this project accepts - so the reference has to be
    # PLN-1474's best passing pose too, or the comparison is not like for like. An
    # earlier version of this printout reported the unfiltered minimum and labelled it
    # the threshold; for PLN-1474 that is -6.998 against a true -6.953.
    import csv as _csv

    rows = list(_csv.DictReader(open(args.out_csv, newline="")))
    labels: list[str] = []
    for row in rows:
        if row["label"] not in labels:
            labels.append(row["label"])

    def best_of(subset):
        values = []
        for row in subset:
            try:
                values.append(float(row["affinity"]))
            except (TypeError, ValueError):
                continue
        return min(values) if values else None

    print("\naffinity per panel compound (kcal/mol):")
    print(f"  {'compound':24s} {'poses':>6s} {'passing':>8s} {'best(all)':>10s} "
          f"{'best(passing)':>14s}")
    summary = {}
    for label in labels:
        subset = [r for r in rows if r["label"] == label]
        passing = [r for r in subset if r.get("passes") == "1"]
        overall, filtered = best_of(subset), best_of(passing)
        summary[label] = filtered
        shown = f"{filtered:14.3f}" if filtered is not None else f"{'none':>14s}"
        print(f"  {label.replace(PANEL_PREFIX, ''):24s} {len(subset):6d} "
              f"{len(passing):8d} "
              f"{overall if overall is None else f'{overall:10.3f}'} {shown}")

    print("\n  A compound with no passing pose is not a failure of the compound.")
    print("  Section 8.5b's filter asks for a carboxylate on Ca501 AND a donor on")
    print("  beta1-Asn224, so it selects on BINDING MODE, not on score - which is")
    print("  why the panel's best affinity can be a compound that does not pass.")

    reference = summary.get(f"{PANEL_PREFIX}PLN-1474")
    if reference is None:
        print("\n  PLN-1474 produced no PASSING pose - section 8.7's hard filter has no"
              "\n  like-for-like reference. Do not proceed to lead selection.",
              file=sys.stderr)
        return 1
    print(f"\n  section 8.7 hard filter: keep candidates whose best PASSING pose scores"
          f"\n  below {reference:.3f} kcal/mol (PLN-1474's best passing pose)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
