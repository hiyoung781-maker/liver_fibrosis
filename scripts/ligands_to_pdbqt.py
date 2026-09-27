"""SDF -> PDBQT for Uni-Dock, via Meeko, keeping the survivor label on every file.

Uni-Dock is Vina-derived and wants PDBQT ligands. Meeko rather than Open Babel,
because Meeko embeds the molecule's chemistry in the PDBQT REMARK records, which is
what lets the docked poses be exported back to SDF with correct bond orders. That
round trip matters: scripts/pose_geometry.py finds the carboxylate with a SMARTS
match, and a PDBQT read back with guessed bonds can lose the C(=O)O pattern - which
would silently empty the very measurement section 8.5b depends on.

One PDBQT per ligand, named after the survivor label (`gen_NNNNN.pdbqt`), because
Uni-Dock takes a file of ligand paths and names its outputs after the inputs. The
label is the identity every downstream artifact keys on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

__all__ = ["prepare_one", "write_pdbqt_set"]

RDLogger.DisableLog("rdApp.*")


def prepare_one(mol: Chem.Mol) -> str | None:
    """PDBQT text for one 3D molecule, or None if Meeko refuses it.

    Meeko needs explicit hydrogens and a conformer; prepare_ligands.py supplies both.
    Returning None rather than raising keeps one awkward ligand from aborting a
    7,767-molecule run - the caller counts and reports the losses instead.
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    try:
        setups = MoleculePreparation().prepare(mol)
    except Exception:
        return None
    if not setups:
        return None

    # Meeko 0.8 returns (pdbqt_string, is_ok, error_msg); older versions returned the
    # string alone. Accept both rather than pinning a Meeko version here.
    result = PDBQTWriterLegacy.write_string(setups[0])
    if isinstance(result, tuple):
        text = result[0]
        ok = bool(result[1]) if len(result) > 1 else True
    else:
        text, ok = result, True
    return text if (ok and text) else None


def write_pdbqt_set(sdf_path: str, out_dir: str) -> dict:
    """Convert every molecule in an SDF; returns counts and the index file path.

    Also writes `ligands.txt`, one absolute PDBQT path per line - the form Uni-Dock's
    --ligand_index expects.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written, failures = [], []
    for mol in Chem.SDMolSupplier(sdf_path, removeHs=False):
        if mol is None:
            failures.append("<unparseable>")
            continue
        label = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if not label:
            failures.append("<unnamed>")
            continue
        text = prepare_one(mol)
        if text is None:
            failures.append(label)
            continue
        path = out / f"{label}.pdbqt"
        path.write_text(text)
        written.append(str(path.resolve()))

    index = out / "ligands.txt"
    index.write_text("\n".join(written) + ("\n" if written else ""))
    return {
        "written": len(written),
        "failures": len(failures),
        "failure_labels": failures,
        "index": str(index),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sdf", required=True, help="3D SDF from prepare_ligands.py")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    counts = write_pdbqt_set(args.sdf, args.out_dir)
    print(f"{args.sdf} -> {args.out_dir}")
    print(f"  written  {counts['written']}")
    print(f"  failures {counts['failures']}")
    if counts["failure_labels"]:
        print(f"  failed: {', '.join(counts['failure_labels'][:20])}"
              + (" ..." if len(counts["failure_labels"]) > 20 else ""))
    print(f"  index    {counts['index']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
