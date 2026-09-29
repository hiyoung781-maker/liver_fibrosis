"""The species that binds, at pH 7.4: acids off, bases on (spec §7.4, extended).

Spec 7.4 registered the CARBOXYLATE's protonation -- neutral against anion,
anion adopted -- and nothing else. The basic nitrogen was left neutral by
default, so every docked molecule with an Arg-mimic head was a -1 ANION. The
bound species is a ZWITTERION: the spec's own text calls this family's
binding mode the "RGD zwitterionic mode".

THE CRYSTAL STRUCTURES DECIDE WHERE THE PROTON GOES. Measured on each
isoform's own deposited ligand, distances to the alphaV subunit's Asp218
carboxylate oxygens:

  9CZD  tetrahydronaphthyridine   aromatic N (no H)  2.64 A to OD2
                                  sp3 N-H            2.81 A to OD1
  6MK0  aromatic 1,8-naphthyridine  N (no H)         2.60 A to OD2
                                    N (no H)         2.85 A to OD1

In 9CZD the nitrogen WITHOUT a hydrogen is the one 2.64 A from a carboxylate
oxygen, so it is the one that carries the proton. Protonating it makes the
head a 2-aminopyridinium presenting TWO donors that chelate the carboxylate
-- the two-point contact a guanidinium makes, which is what an Arg mimic is
for.

WHAT THIS COST. Vina has no explicit electrostatic term, so a neutral head
offers it no reason to place the group against Asp218 at all. With Open
Babel's mis-perceived tetrahydro form of 6MK0's ligand the molecule happened
to carry an N-H and the crystal pose came back at 1.09 A; with the correct
aromatic form, neutral, the best of 16 poses was 4.81 A. The right
interaction had been reproduced for the wrong reason.

THE RULE IS pKa, APPLIED UNIFORMLY, AND ONE CASE DISAGREES WITH IT. Solution
pKa puts guanidines and amidines above 12, aliphatic amines at 9-10 and
2-aminopyridine-type heads near 7.5, so all are cationic at pH 7.4; plain
pyridines (~5) and anilines (~4.6) are not. 6MK0's head is a FULLY AROMATIC
1,8-naphthyridine at about 3.4, which the rule therefore leaves neutral --
while its structure shows both nitrogens 2.60 and 2.85 A from a carboxylate,
which a neutral acceptor cannot do. The rule is still applied uniformly and
the disagreement is reported, because protonating one compound on the
grounds that its answer is known is how a pre-registration stops meaning
anything.

HEAVY ATOMS NEVER MOVE. Both edits are formal-charge and hydrogen-count
changes on existing atoms.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

__all__ = ["BASIC_RULES", "protonate_bases", "to_physiological"]

# name -> (SMARTS for the atom to protonate, why)
BASIC_RULES = {
    "guanidine/amidine": (
        "[NX2;$(N=[CX3]([NX3])[#6,#7])]",
        "guanidinium pKa ~13.6, amidinium ~11-12: cationic at pH 7.4",
    ),
    "2-aminopyridine head (Arg mimic)": (
        "[nX2;$(n:c-,:[NX3;H1,H2]);!$(n:n);!$(n:c:n);!$(n:c:c:n)]",
        "2-aminopyridinium pKa ~7.5 (tetrahydronaphthyridine): cationic at "
        "pH 7.4, and protonating the RING nitrogen is what gives the head the "
        "donor pair an Arg mimic needs -- 9CZD shows that nitrogen, with no "
        "hydrogen on it, 2.64 A from Asp218 OD2. The three negative patterns "
        "require the ring to hold ONE aromatic nitrogen: a second one "
        "withdraws enough to drop the pKa out of range, and without them "
        "bexotegrast's 4-aminoquinazoline (~5.5) and GLPG0187's "
        "2-aminopyrimidine (~3.5) were being protonated too",
    ),
    "aliphatic amine": (
        "[NX3;H0,H1,H2;!$(N[!#6;!#1]);!$(N*=[O,S,N]);!$(Nc);!$(N-[SX4]);"
        "!$(N=*)]",
        "alkylammonium pKa ~9-10.7: cationic at pH 7.4. Anilines (~4.6), "
        "amides and sulfonamides are excluded by the negative patterns",
    ),
}

_ACID = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")


def protonate_bases(mol: Chem.Mol) -> Chem.Mol:
    """Charge every basic nitrogen the BASIC_RULES cover. Heavy atoms untouched.

    Rules are applied in order and an atom already charged is skipped, so a
    nitrogen matching two patterns is protonated once.
    """
    # Does the input carry hydrogens as ATOMS? A molecule from embed() does;
    # one from MolFromSmiles does not. Incrementing the hydrogen COUNT on a
    # molecule of the first kind leaves one implicit hydrogen on that
    # nitrogen and nowhere else, and Meeko refuses the result with "RDKit
    # molecule has implicit Hs. Need explicit Hs." -- which is how PLN-1474
    # and bexotegrast, the two references carrying an Arg-mimic head,
    # silently dropped out of the panel.
    explicit = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    out = Chem.RWMol(mol)
    done: list[int] = []
    for smarts, _reason in BASIC_RULES.values():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        for (index,) in out.GetMol().GetSubstructMatches(pattern):
            if index in done:
                continue
            atom = out.GetAtomWithIdx(index)
            if atom.GetFormalCharge() != 0:
                continue
            atom.SetFormalCharge(1)
            if not explicit:
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                atom.SetNoImplicit(True)
            done.append(index)

    result = out.GetMol()
    if explicit and done:
        # Add the proton as an ATOM, positioned geometrically, so the
        # molecule stays all-explicit and no heavy atom moves.
        Chem.SanitizeMol(result)
        result = Chem.AddHs(result, addCoords=True, onlyOnAtoms=done)
    Chem.SanitizeMol(result)
    return result


def to_physiological(mol: Chem.Mol) -> Chem.Mol:
    """Carboxylates deprotonated, bases protonated: the species at pH 7.4.

    A molecule with an acid and no base comes back an anion; one with both
    comes back a zwitterion. Nothing is re-embedded and no heavy atom moves.
    """
    from deprotonate_acids import deprotonate

    try:
        out = deprotonate(mol)
    except ValueError:
        out = Chem.Mol(mol)  # no carboxylic acid; bases may still apply
    return protonate_bases(out)
