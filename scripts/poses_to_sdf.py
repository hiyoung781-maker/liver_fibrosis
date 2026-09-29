"""Uni-Dock pose PDBQT -> SDF, through Meeko, so the chemistry survives.

This is not a convenience conversion; without it the section 8.5b measurement is
empty. Measured on a real ligand:

  RDKit reading the pose PDBQT directly  -> unparseable
      ("Element 'A' not found" - AutoDock's aromatic-carbon type is not an element)
  Meeko round trip                       -> 1 carboxylate match, conformer intact

scripts/pose_geometry.py finds the MIDAS anchor with the SMARTS
`[CX3](=O)[OX2H1,OX1-]`. A pose whose bonds were guessed, or not read at all, yields
zero matches, and the filter then reports every pose as failing - a result that looks
chemical and is not. Meeko can do this because ligands_to_pdbqt.py wrote the
molecule's SMILES into the PDBQT's REMARK records on the way in.

Uni-Dock batch mode writes one `<label>_out.pdbqt` per ligand into --dir. The label
is the survivor id, which is the identity every downstream artifact keys on, so it is
carried onto each pose as the SDF title.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase

__all__ = ["convert_dir", "poses_from_pdbqt"]

RDLogger.DisableLog("rdApp.*")


def mol_from_pdbqt_block(text: str, name: str = "pose"):
    """One RDKit molecule from a PDBQT MODEL block, through Meeko.

    OPEN BABEL IS NOT USED HERE, AND THAT IS THE POINT. A Meeko PDBQT carries
    its own `REMARK SMILES`, so the bond orders are recorded rather than
    guessed -- while Open Babel infers them from coordinates and gets fused
    N-heterocycles wrong. Measured on this project's own crystal ligands: it
    reduced 6MK0's aromatic 1,8-naphthyridine to a tetrahydronaphthyridine and
    turned 9CZD's tetrahydronaphthyridine into an imine, inverting the
    hydrogen-bonding character of the group that binds alphaV Asp218 in both.
    8W30's A1AFA has no such ring, which is the only reason the alphaVbeta1
    path was unharmed.

    Returns None rather than raising, so one bad pose cannot stop a batch.
    """
    from meeko import PDBQTMolecule, RDKitMolCreate

    try:
        with rdBase.BlockLogs():
            pdbqt = PDBQTMolecule(text, name=name, skip_typing=True)
            mols = RDKitMolCreate.from_pdbqt_mol(pdbqt)
    except Exception:  # noqa: BLE001 - one pose must not stop a batch
        return None
    for mol in mols or []:
        if mol is not None and mol.GetNumConformers():
            single = Chem.Mol(mol)
            single.RemoveAllConformers()
            single.AddConformer(mol.GetConformer(0), assignId=True)
            single.SetProp("_Name", name)
            return single
    return None


def poses_from_pdbqt(path: str, label: str | None = None) -> list[Chem.Mol]:
    """Every pose in one Uni-Dock output PDBQT, as RDKit molecules with coordinates.

    Meeko returns one molecule per input molecule with each pose as a conformer;
    this splits them into one molecule per pose so downstream code can treat a pose
    as a unit. Returns [] rather than raising, so one bad file cannot abort a
    7,767-ligand conversion - the caller counts the losses.
    """
    from meeko import PDBQTMolecule, RDKitMolCreate

    name = label or Path(path).stem.removesuffix("_out")
    try:
        with rdBase.BlockLogs():
            pdbqt = PDBQTMolecule.from_file(path, skip_typing=True)
            mols = RDKitMolCreate.from_pdbqt_mol(pdbqt)
    except Exception:
        return []

    # Uni-Dock records each pose's energy as `REMARK VINA RESULT: <kcal/mol>`. Meeko
    # does not put it on the RDKit molecule but exposes it as PDBQTMolecule pose.score,
    # so it is collected here and written as an SDF property; section 8.5b uses docking
    # score to break ties among poses that already passed the geometry filter.
    #
    # In its OWN try, deliberately. Collecting scores inside the block above made a
    # missing energy fatal: a PDBQT with no VINA RESULT remark - a freshly prepared
    # ligand rather than a docked output - returned zero poses instead of poses with a
    # nan score. An absent score must cost the score, never the pose.
    try:
        with rdBase.BlockLogs():
            scores = [getattr(pose, "score", None) for pose in pdbqt]
    except Exception:
        scores = []

    poses: list[Chem.Mol] = []
    for mol in mols:
        if mol is None:
            continue
        for index in range(mol.GetNumConformers()):
            single = Chem.Mol(mol)
            single.RemoveAllConformers()
            single.AddConformer(mol.GetConformer(index), assignId=True)
            single.SetProp("_Name", name)
            single.SetProp("pose", str(index + 1))
            if index < len(scores) and scores[index] is not None:
                single.SetProp("affinity", f"{float(scores[index]):.3f}")
            poses.append(single)
    return poses


def convert_dir(pose_dir: str, out_sdf: str, pattern: str = "*_out.pdbqt") -> dict:
    """Convert every Uni-Dock output in a directory into one multi-molecule SDF."""
    files = sorted(Path(pose_dir).glob(pattern))
    writer = Chem.SDWriter(out_sdf)
    counts = {"files": len(files), "poses": 0, "failed_files": 0, "failures": []}
    for path in files:
        poses = poses_from_pdbqt(str(path))
        if not poses:
            counts["failed_files"] += 1
            counts["failures"].append(path.name)
            continue
        for pose in poses:
            writer.write(pose)
            counts["poses"] += 1
    writer.close()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-dir", required=True, help="Uni-Dock --dir output")
    parser.add_argument("--out-sdf", required=True)
    parser.add_argument("--pattern", default="*_out.pdbqt")
    args = parser.parse_args(argv)

    counts = convert_dir(args.pose_dir, args.out_sdf, args.pattern)
    print(f"{args.pose_dir} -> {args.out_sdf}")
    for key in ("files", "poses", "failed_files"):
        print(f"  {key:14s} {counts[key]}")
    if counts["failures"]:
        print(f"  failed: {', '.join(counts['failures'][:20])}"
              + (" ..." if len(counts["failures"]) > 20 else ""))
    if counts["poses"] == 0 and counts["files"]:
        print("\n  NOTHING CONVERTED. Do not run the geometry filter on this - it"
              "\n  would report every pose as failing, which looks chemical.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
