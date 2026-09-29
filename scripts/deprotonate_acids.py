"""Deprotonate carboxylic acids for docking in the adopted anion state (spec 7.4).

spec 7.4 pre-registers a redock in BOTH protonation states and states which
one wins: "결정 접촉을 재현하는 쪽을 채택. 둘 다 재현하면 음이온을 택한다
(AutoDock4에는 정전기 항이 있어 v1의 중성 선택 근거가 소멸)". v1 docked the
neutral carboxylic acid because Vina has no electrostatic term, so the charge
state could not matter to it. AutoDock4 does have one, and the MIDAS ion is
Ca2+, so the carboxylate's charge is the dominant electrostatic driver of this
binding mode. Docking the neutral acid against a +2 metal removes it.

The v2 batch docked the neutral form: docking/ligand_ref.sdf carries no
`M  CHG` record, and prepare_ligands.py's --control defaults to that file. The
anion arm of the pre-registered gate had therefore never been built. This
module builds it.

Uni-Dock passed the gate in both states (0.66 A neutral, 0.63 A anion), so
spec 7.4's tie-break applies and the anion is adopted. The generated library
must then be docked in the same state as the control that validated the
protocol: a survivor's PDBQT reads `REMARK SMILES ...C(=O)O...`, and some
carry more than one acid, so the batch mode below deprotonates every one.

COORDINATES MUST NOT MOVE. The control exists to hold the experimentally
observed pose; the RMSD criterion compares a redocked pose against exactly
these coordinates. For the library the reason is different but just as
binding: those conformers are what docking starts from, and re-embedding
would change the run rather than its protonation. Deprotonation here is a
formal-charge edit on the hydroxyl oxygen in both modes, never a
re-embedding, and the tests pin the maximum displacement at zero. A crystal
ligand SDF carries no explicit hydrogens, so there is no hydrogen to remove
-- setting the oxygen's formal charge to -1 and its explicit hydrogen count
to 0 is the whole edit.

A molecule with no carboxylic acid raises in deprotonate() -- for a single
control, writing it back unchanged would make the anion arm a silent
duplicate of the neutral arm. In the batch mode it passes through unchanged
and is counted instead, because one such molecule must not stop a library.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem

__all__ = ["ACID_SMARTS", "deprotonate", "deprotonate_sdf",
           "deprotonate_dir", "write_anion"]

# The hydroxyl oxygen of a carboxylic acid. H1 matches the implicit hydrogen
# RDKit infers for a two-connection, neutral oxygen read from a hydrogen-free
# SDF, which is how the crystal ligand arrives.
ACID_SMARTS = "[CX3](=[OX1])[OX2H1]"


def deprotonate(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of `mol` with every carboxylic acid as a carboxylate.

    Raises when the molecule has no carboxylic acid: writing it back unchanged
    would make the anion arm of the gate a silent duplicate of the neutral
    arm, reported as an independent run.
    """
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(ACID_SMARTS))
    if not matches:
        raise ValueError(
            "no carboxylic acid in this molecule, so there is nothing to "
            "deprotonate. The redocking control is an acidic integrin ligand; "
            "a molecule without one is not it."
        )
    anion = Chem.RWMol(mol)
    # The acidic hydrogen may be an implicit count or a real atom. A crystal
    # ligand SDF carries no hydrogens, so setting the count is the whole edit;
    # a prepared library molecule carries them explicitly, and leaving the H
    # bonded while charging the oxygen makes it 2-coordinate with charge -1,
    # which SanitizeMol rejects. RDKit's sanitize exceptions subclass
    # ValueError, so that rejection was indistinguishable from "no acid here"
    # and a whole library came back silently unchanged.
    acidic_hydrogens = []
    for _carbon, _carbonyl_o, hydroxyl_o in matches:
        atom = anion.GetAtomWithIdx(hydroxyl_o)
        for neighbour in atom.GetNeighbors():
            if neighbour.GetAtomicNum() == 1:
                acidic_hydrogens.append(neighbour.GetIdx())
                break
        atom.SetFormalCharge(-1)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    # Descending, so each removal cannot shift an index still to be removed.
    for index in sorted(acidic_hydrogens, reverse=True):
        anion.RemoveAtom(index)
    out = anion.GetMol()
    Chem.SanitizeMol(out)
    return out


def write_anion(reference_sdf: str, out_sdf: str) -> str:
    """Read the crystal ligand SDF, deprotonate it, write it back. Returns the path."""
    mol = Chem.MolFromMolFile(reference_sdf, removeHs=False)
    if mol is None:
        raise ValueError(f"{reference_sdf}: RDKit could not read a molecule")
    anion = deprotonate(mol)
    Path(out_sdf).parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(out_sdf)
    writer.write(anion)
    writer.close()
    return out_sdf


def deprotonate_sdf(in_sdf: str, out_sdf: str) -> dict:
    """Deprotonate every acid in every molecule of an SDF. Returns counts.

    Molecule names are preserved: every stage downstream keys on them. A
    molecule with no carboxylic acid is written unchanged and counted under
    "unchanged"; one RDKit cannot read or sanitise is counted under "failed"
    and skipped, so a single bad record cannot stop a 9,726-ligand library.
    """
    from rdkit import Chem

    counts = {"read": 0, "deprotonated": 0, "unchanged": 0, "failed": 0}
    Path(out_sdf).parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(out_sdf)
    for mol in Chem.SDMolSupplier(in_sdf, removeHs=False):
        if mol is None:
            counts["failed"] += 1
            continue
        counts["read"] += 1
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        try:
            out = deprotonate(mol)
            counts["deprotonated"] += 1
        except ValueError:
            out = mol
            counts["unchanged"] += 1
        except Exception:
            counts["failed"] += 1
            counts["read"] -= 1
            continue
        if name:
            out.SetProp("_Name", name)
        writer.write(out)
    writer.close()
    return counts


def deprotonate_dir(in_dir: str, out_dir: str, pattern: str = "*.sdf") -> dict:
    """Run deprotonate_sdf over every SDF in a directory, keeping filenames."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = {"read": 0, "deprotonated": 0, "unchanged": 0, "failed": 0,
             "files": 0}
    for path in sorted(Path(in_dir).glob(pattern)):
        counts = deprotonate_sdf(str(path), str(out / path.name))
        total["files"] += 1
        for key in ("read", "deprotonated", "unchanged", "failed"):
            total[key] += counts[key]
    return total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference", default="docking/ligand_ref.sdf",
                    help="Single molecule SDF (the crystal control).")
    p.add_argument("--out", default="docking/ligand_ref_anion.sdf",
                    help="Where to write the deprotonated control.")
    p.add_argument("--sdf-dir",
                    help="Batch mode: deprotonate every SDF in this directory "
                         "(the shard SDFs from prepare_ligands.py) instead of "
                         "a single control.")
    p.add_argument("--out-dir", help="Batch mode output directory.")
    args = p.parse_args(argv)

    if args.sdf_dir:
        if not args.out_dir:
            sys.stderr.write("--sdf-dir requires --out-dir\n")
            return 2
        counts = deprotonate_dir(args.sdf_dir, args.out_dir)
        print(f"{args.sdf_dir} -> {args.out_dir}")
        for key in ("files", "read", "deprotonated", "unchanged", "failed"):
            print(f"  {key:14s} {counts[key]}")
        return 1 if counts["failed"] else 0

    path = write_anion(args.reference, args.out)
    before = Chem.MolFromMolFile(args.reference, removeHs=False)
    after = Chem.MolFromMolFile(path, removeHs=False)
    shift = float(abs(before.GetConformer().GetPositions()
                      - after.GetConformer().GetPositions()).max())
    print(f"{args.reference} -> {path}")
    print(f"  formal charge      {Chem.GetFormalCharge(before):+d} -> "
          f"{Chem.GetFormalCharge(after):+d}")
    print(f"  max atom displacement {shift:.9f} A")
    if shift != 0.0:
        sys.stderr.write("coordinates moved; the control no longer holds the "
                         "observed pose. Refusing.\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
