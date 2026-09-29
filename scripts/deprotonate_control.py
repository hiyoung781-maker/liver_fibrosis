"""Build the anionic redocking control from the crystal ligand (spec 7.4).

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

COORDINATES MUST NOT MOVE. The control exists to hold the experimentally
observed pose; the RMSD criterion compares a redocked pose against exactly
these coordinates. Deprotonation here is a formal-charge edit on the hydroxyl
oxygen, never a re-embedding, and the tests pin the maximum displacement at
zero. The crystal ligand SDF carries no explicit hydrogens, so there is no
hydrogen to remove -- setting the oxygen's formal charge to -1 and its
explicit hydrogen count to 0 is the whole edit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem

__all__ = ["ACID_SMARTS", "deprotonate", "write_anion"]

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
    for _carbon, _carbonyl_o, hydroxyl_o in matches:
        atom = anion.GetAtomWithIdx(hydroxyl_o)
        atom.SetFormalCharge(-1)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference", default="docking/ligand_ref.sdf",
                    help="Crystal ligand SDF, neutral as deposited.")
    p.add_argument("--out", default="docking/ligand_ref_anion.sdf",
                    help="Where to write the deprotonated control.")
    args = p.parse_args(argv)

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
