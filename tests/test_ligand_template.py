import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from ligand_template import (TemplateMismatchError, assign_from_template,
                             canonical, ccd_smiles)

# The authoritative definitions, from the PDB chemical component dictionary.
CCD = {
    "A1AFA": "c1ccc(cc1)C[C@@H](C(=O)O)NC(=O)c2ccc(cc2Cl)Cl",
    "JUY": "c1ccc2c(c1)nc(s2)C(=O)N[C@@H](CCNC(=O)CCCCc3ccc4cccnc4n3)C(=O)O",
    "A1A6H": ("c1cc(c(cc1F)[C@@H](C(=O)O)N2CC[C@H](C2)OCCCCc3ccc4c(n3)NCCC4)"
              "[C@@H]5CCCCO5"),
}

LIGAND_PDB = {
    "JUY": "docking/v2/isoforms/avb3/ligand.pdb",
    "A1A6H": "docking/v2/isoforms/avb6/ligand.pdb",
}


def have(key):
    return Path(LIGAND_PDB[key]).exists()


class TestWhyThisModuleExists(unittest.TestCase):
    """Open Babel infers bond orders from coordinates, and it gets fused
    N-heterocycles wrong. Both isoform crystal ligands were mis-perceived in
    the Arg-mimic head -- the group that binds the alphaV subunit's Asp218 --
    and in OPPOSITE directions:

      JUY    CCD has an aromatic 1,8-naphthyridine (c3ccc4cccnc4n3);
             Open Babel returned a tetrahydronaphthyridine (NCCC), reducing
             three double bonds and turning an acceptor N into a donor N-H.
      A1A6H  CCD has a tetrahydronaphthyridine (NCCC3); Open Babel returned
             an imine (N=CCC3), turning the donor N-H into an sp2 acceptor.

    8W30's A1AFA came through unchanged, because it is a phenylalanine amide
    with no fused N-heterocycle -- which is why the alphaVbeta1 gate was
    sound while both isoform gates docked molecules that were not the crystal
    ligands."""

    # What Open Babel actually returned, read back from the SDFs it wrote.
    OPEN_BABEL = {
        "A1AFA": "O=C(NC(Cc1ccccc1)C(=O)O)c1ccc(Cl)cc1Cl",
        "JUY": "O=C(CCCCc1ccc2c(n1)NCCC2)NCCC(NC(=O)c1nc2ccccc2s1)C(=O)O",
        "A1A6H": ("O=C(O)C(c1cc(F)ccc1C1CCCCO1)N1CCC"
                  "(OCCCCc2ccc3c(n2)N=CCC3)C1"),
    }

    def test_open_babel_reproduced_the_alphaVbeta1_ligand(self):
        self.assertEqual(canonical(self.OPEN_BABEL["A1AFA"]),
                         canonical(CCD["A1AFA"]))

    def test_open_babel_got_both_isoform_ligands_wrong(self):
        for key in ("JUY", "A1A6H"):
            self.assertNotEqual(canonical(self.OPEN_BABEL[key]),
                                canonical(CCD[key]), key)

    def test_the_difference_is_in_the_naphthyridine_head(self):
        # JUY's aromatic ring was reduced; A1A6H's N-H became an imine.
        aromatic = Chem.MolFromSmarts("c1ccc2cccnc2n1")
        self.assertTrue(Chem.MolFromSmiles(CCD["JUY"]).HasSubstructMatch(aromatic))
        self.assertFalse(
            Chem.MolFromSmiles(self.OPEN_BABEL["JUY"]).HasSubstructMatch(aromatic))
        imine = Chem.MolFromSmarts("[NX2]=[CX3]")
        self.assertFalse(Chem.MolFromSmiles(CCD["A1A6H"]).HasSubstructMatch(imine))
        self.assertTrue(
            Chem.MolFromSmiles(self.OPEN_BABEL["A1A6H"]).HasSubstructMatch(imine))


class TestCanonical(unittest.TestCase):
    def test_it_ignores_stereochemistry_and_charge(self):
        # The campaign deprotonates its ligands on purpose; that must not
        # read as a template mismatch.
        self.assertEqual(canonical("CC(=O)O"), canonical("CC(=O)[O-]"))

    def test_it_does_not_ignore_bond_order(self):
        # The defect this module exists for.
        self.assertNotEqual(canonical("c1ccc2ccccc2c1"),
                            canonical("C1CCC2CCCCC2C1"))


@unittest.skipUnless(all(have(k) for k in LIGAND_PDB), "isoform ligands absent")
class TestAssignFromTemplate(unittest.TestCase):

    def _pdb(self, key):
        return Path(LIGAND_PDB[key]).read_text()

    def test_the_result_matches_the_CCD_definition(self):
        for key in ("JUY", "A1A6H"):
            mol = assign_from_template(self._pdb(key), CCD[key])
            self.assertEqual(canonical(Chem.MolToSmiles(mol)),
                             canonical(CCD[key]), key)

    def test_coordinates_are_the_crystal_ones(self):
        # Bond orders are assigned; nothing is re-embedded. The RMSD gate
        # compares a redocked pose against exactly these coordinates.
        import numpy as np
        for key in ("JUY", "A1A6H"):
            raw = Chem.MolFromPDBBlock(self._pdb(key), removeHs=False,
                                       sanitize=False)
            mol = assign_from_template(self._pdb(key), CCD[key])
            before = np.array(sorted(map(tuple,
                                         raw.GetConformer().GetPositions().round(3))))
            after = np.array(sorted(map(tuple,
                                        mol.GetConformer().GetPositions().round(3))))
            self.assertEqual(before.shape, after.shape, key)
            self.assertAlmostEqual(float(np.abs(before - after).max()), 0.0,
                                   places=6, msg=key)

    def test_a_wrong_template_is_refused_rather_than_forced(self):
        # Silently producing SOMETHING is how the isoform gate came to judge
        # the wrong molecules in the first place.
        with self.assertRaises(TemplateMismatchError):
            assign_from_template(self._pdb("JUY"), CCD["A1AFA"])

    def test_the_old_open_babel_result_would_not_pass_this_check(self):
        # Pins the regression: the tetrahydro form is not the CCD molecule.
        obabel_juy = "O=C(CCCCc1ccc2c(n1)NCCC2)NCCC(NC(=O)c1nc2ccccc2s1)C(=O)O"
        self.assertNotEqual(canonical(obabel_juy), canonical(CCD["JUY"]))


class TestCcdSmilesIsVerified(unittest.TestCase):
    """A fetched template is still a template: if it does not describe the
    same heavy-atom count as the ligand in the structure, assigning it would
    produce a confident wrong answer."""

    def test_heavy_atom_mismatch_raises(self):
        with self.assertRaises(TemplateMismatchError):
            assign_from_template(
                "HETATM    1  C   LIG B 900       0.000   0.000   0.000"
                "  1.00  0.00           C  \nEND\n", CCD["A1AFA"])


if __name__ == "__main__":
    unittest.main()
