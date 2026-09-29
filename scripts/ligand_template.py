"""Bond orders for a crystal ligand from its PDB chemical component, not from geometry.

Open Babel infers bond orders from coordinates, and on fused N-heterocycles it
gets them wrong. Both isoform crystal ligands were mis-perceived in the
Arg-mimic head -- the group that binds the alphaV subunit's Asp218 -- and in
OPPOSITE directions:

  JUY (6MK0)    CCD: an aromatic 1,8-naphthyridine, `c3ccc4cccnc4n3`.
                Open Babel: a tetrahydronaphthyridine, `NCCC`. Three double
                bonds reduced, and an acceptor nitrogen turned into a donor
                N-H.
  A1A6H (9CZD)  CCD: a tetrahydronaphthyridine, `NCCC3`.
                Open Babel: an imine, `N=CCC3`. The donor N-H turned into an
                sp2 acceptor.

Both errors invert the hydrogen-bonding character of the pharmacophore, and
both change the ring geometry and the rotatable-bond tree Meeko builds from
it. The section 10.3 gate therefore asked whether Uni-Dock could reproduce
the crystal pose of a molecule that was not the crystal ligand, and the
answer to that question means nothing.

8W30's A1AFA survived because it is a phenylalanine amide with no fused
N-heterocycle: there was nothing for the perception to get wrong. That is
why the alphaVbeta1 gate was sound and both isoform gates were not.

THE FIX IS A TEMPLATE, AND IT IS MANDATORY. Connectivity comes from the
coordinates, bond orders come from the deposited chemical component, and a
template that does not match the extracted ligand RAISES. Producing something
plausible from a mismatch is exactly how the wrong molecules got docked.

COORDINATES NEVER MOVE. Bond orders are assigned onto the crystal positions;
nothing is re-embedded. The RMSD criterion compares a redocked pose against
exactly these coordinates.
"""

from __future__ import annotations

import json
import urllib.request

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

__all__ = ["TemplateMismatchError", "assign_from_template", "canonical",
           "ccd_smiles"]

RDLogger.DisableLog("rdApp.*")


class TemplateMismatchError(ValueError):
    """The template does not describe the ligand in the structure.

    Raised rather than resolved, because every alternative is a confident
    wrong answer.
    """


def canonical(smiles_or_mol) -> str:
    """Canonical SMILES ignoring stereochemistry and formal charge.

    Charge is ignored because this campaign deprotonates its ligands on
    purpose (spec 7.4's anion decision) and that must not read as a template
    mismatch. Bond order is NOT ignored -- it is the thing being checked.
    """
    mol = (Chem.MolFromSmiles(smiles_or_mol)
           if isinstance(smiles_or_mol, str) else Chem.Mol(smiles_or_mol))
    if mol is None:
        raise TemplateMismatchError(f"unreadable molecule: {smiles_or_mol!r}")
    mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(mol)
    except ValueError as exc:
        raise TemplateMismatchError(f"cannot sanitise after neutralising: {exc}")
    return Chem.MolToSmiles(mol)


def ccd_smiles(comp_id: str, timeout: int = 60) -> str:
    """The deposited SMILES for a PDB chemical component.

    NOTE ON COMPONENT IDS. Five-character CCD codes do not fit the PDB
    format's three-character residue field, so a structure converted to PDB
    carries a truncated id -- 9CZD's A1A6H appears as A1A. Pass the full id;
    a truncated one fetches a different compound or nothing.
    """
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = json.load(response)
    for preferred in ("OpenEye OEToolkits", None):
        for entry in data.get("pdbx_chem_comp_descriptor", []):
            if entry.get("type") != "SMILES_CANONICAL":
                continue
            if preferred is None or entry.get("program") == preferred:
                return entry["descriptor"]
    raise TemplateMismatchError(f"{comp_id}: no canonical SMILES in the CCD")


def assign_from_template(pdb_block: str, template_smiles: str) -> Chem.Mol:
    """The ligand from `pdb_block` with the template's bond orders.

    Connectivity comes from the coordinates; bond orders come from the
    template. Raises TemplateMismatchError when the two do not describe the
    same molecule, including when the heavy-atom counts differ.
    """
    template = Chem.MolFromSmiles(template_smiles)
    if template is None:
        raise TemplateMismatchError(f"unreadable template: {template_smiles!r}")

    raw = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if raw is None:
        raise TemplateMismatchError("RDKit could not read the ligand PDB block")

    n_raw = sum(1 for a in raw.GetAtoms() if a.GetAtomicNum() > 1)
    n_template = template.GetNumHeavyAtoms()
    if n_raw != n_template:
        raise TemplateMismatchError(
            f"the structure's ligand has {n_raw} heavy atoms and the template "
            f"{n_template}. They are not the same compound, and assigning one "
            "to the other would produce a confident wrong answer.")

    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, raw)
    except Exception as exc:  # noqa: BLE001 - RDKit raises several types here
        raise TemplateMismatchError(
            f"could not map the template onto the structure's ligand: {exc}. "
            "The connectivity read from the coordinates does not match the "
            "deposited chemical component.")

    if canonical(Chem.MolToSmiles(mol)) != canonical(template_smiles):
        raise TemplateMismatchError(
            "bond-order assignment produced a different molecule from the "
            f"template:\n    got      {canonical(Chem.MolToSmiles(mol))}\n"
            f"    template {canonical(template_smiles)}")
    return mol
