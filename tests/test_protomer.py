import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from protomer import BASIC_RULES, protonate_bases, to_physiological

REFS = {
    "A1AFA": "O=C(NC(Cc1ccccc1)C(=O)O)c1ccc(Cl)cc1Cl",
    "PLN-1474": "CC1(C(=O)NC(CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1",
    "bexotegrast": "COCCN(CCCCc1ccc2c(n1)NCCC2)CCC(Nc1ncnc2ccccc12)C(=O)O",
    "CWHM-12": ("CC(C)(C)c1cc(Br)cc(C(CC(=O)O)NC(=O)CNC(=O)c2cc(O)cc"
                "(NC3=NCC(O)CN3)c2)c1"),
    "CHEMBL4649232": ("CCNC(=O)Nc1cnnc(-c2ccc(CC(NC(=O)c3c(Cl)cccc3Cl)"
                      "C(=O)O)cc2)c1"),
}


def charge(smiles):
    return Chem.GetFormalCharge(Chem.MolFromSmiles(smiles))


class TestArgMimicHead(unittest.TestCase):
    """The crystal structures decide this, not a guess. In 9CZD the
    tetrahydronaphthyridine's AROMATIC nitrogen sits 2.64 A from alphaV
    Asp218 OD2 with no hydrogen on it, while its sp3 N-H is 2.81 A from OD1.
    Protonating the aromatic nitrogen turns the head into a
    2-aminopyridinium with TWO donors chelating the carboxylate -- the same
    two-point contact a guanidinium makes, which is what an Arg mimic is
    for."""

    def test_the_aromatic_nitrogen_is_the_one_protonated(self):
        mol = protonate_bases(Chem.MolFromSmiles("c1ccc2c(n1)NCCC2"))
        charged = [a for a in mol.GetAtoms() if a.GetFormalCharge() == 1]
        self.assertEqual(len(charged), 1)
        self.assertTrue(charged[0].GetIsAromatic())
        self.assertEqual(charged[0].GetSymbol(), "N")

    def test_the_head_then_presents_two_donors(self):
        mol = protonate_bases(Chem.MolFromSmiles("c1ccc2c(n1)NCCC2"))
        donors = sum(a.GetTotalNumHs() for a in mol.GetAtoms()
                     if a.GetSymbol() == "N")
        self.assertEqual(donors, 2)


class TestWhichBasesAreProtonated(unittest.TestCase):
    """A pKa rule, applied uniformly: guanidines and amidines (pKa > 12),
    aliphatic amines (~9-10) and 2-aminopyridine-type heads (~7.5) are
    cationic at pH 7.4. Plain pyridines and anilines (~5 and ~4.6) are not,
    and neither are amides or sulfonamides."""

    def test_guanidine(self):
        self.assertEqual(charge(Chem.MolToSmiles(
            protonate_bases(Chem.MolFromSmiles("NC(=N)N")))), 1)

    def test_aliphatic_tertiary_amine(self):
        self.assertEqual(charge(Chem.MolToSmiles(
            protonate_bases(Chem.MolFromSmiles("CCN(CC)CC")))), 1)

    def test_a_plain_pyridine_is_left_neutral(self):
        self.assertEqual(charge(Chem.MolToSmiles(
            protonate_bases(Chem.MolFromSmiles("c1ccncc1")))), 0)

    def test_an_aniline_is_left_neutral(self):
        self.assertEqual(charge(Chem.MolToSmiles(
            protonate_bases(Chem.MolFromSmiles("Nc1ccccc1")))), 0)

    def test_an_amide_nitrogen_is_left_neutral(self):
        self.assertEqual(charge(Chem.MolToSmiles(
            protonate_bases(Chem.MolFromSmiles("CC(=O)NC")))), 0)

    def test_a_sulfonamide_nitrogen_is_left_neutral(self):
        self.assertEqual(charge(Chem.MolToSmiles(
            protonate_bases(Chem.MolFromSmiles("CS(=O)(=O)NC")))), 0)

    def test_every_rule_carries_its_reason(self):
        for name, (smarts, reason) in BASIC_RULES.items():
            self.assertTrue(Chem.MolFromSmarts(smarts), name)
            self.assertIn("pKa", reason, name)


class TestPhysiologicalSpecies(unittest.TestCase):
    """spec 7.4 registered the carboxylate's state and nothing else, so the
    docked species was a -1 anion. The bound species is a zwitterion -- the
    spec's own text calls this family's binding mode the RGD zwitterionic
    mode, and both isoform structures show the Arg-mimic nitrogen 2.6-2.85 A
    from a carboxylate."""

    def test_one_acid_and_one_base_gives_a_neutral_zwitterion(self):
        for name in ("PLN-1474", "CWHM-12"):
            mol = to_physiological(Chem.MolFromSmiles(REFS[name]))
            self.assertEqual(Chem.GetFormalCharge(mol), 0, name)

    def test_acids_without_a_basic_head_stay_anions(self):
        for name in ("A1AFA", "CHEMBL4649232"):
            mol = to_physiological(Chem.MolFromSmiles(REFS[name]))
            self.assertEqual(Chem.GetFormalCharge(mol), -1, name)

    def test_bexotegrast_has_both_its_bases_protonated_and_nets_plus_one(self):
        # A tetrahydronaphthyridine AND an aliphatic tertiary amine, against
        # one carboxylate. Its 4-aminoquinazoline (pKa ~5.5) must NOT be
        # protonated -- it was, until the Arg-mimic rule was narrowed to
        # rings holding a single aromatic nitrogen.
        mol = to_physiological(Chem.MolFromSmiles(REFS["bexotegrast"]))
        cations = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
        self.assertEqual(cations, 2)
        self.assertEqual(Chem.GetFormalCharge(mol), 1)

    def test_glpg0187s_aminopyrimidine_is_not_protonated(self):
        # pKa ~3.5. Only its tetrahydronaphthyridine is basic enough.
        smiles = ("COc1ccc(S(=O)(=O)NC(CNc2nc(C)nc(N3CCC(c4ccc5c(n4)NCCC5)"
                  "CC3)c2C)C(=O)O)cc1")
        mol = to_physiological(Chem.MolFromSmiles(smiles))
        self.assertEqual(sum(1 for a in mol.GetAtoms()
                             if a.GetFormalCharge() == 1), 1)
        self.assertEqual(Chem.GetFormalCharge(mol), 0)

    def test_the_carboxylate_is_still_deprotonated(self):
        mol = to_physiological(Chem.MolFromSmiles(REFS["PLN-1474"]))
        self.assertTrue(mol.HasSubstructMatch(
            Chem.MolFromSmarts("[CX3](=O)[OX1-]")))

    def test_heavy_atoms_are_untouched(self):
        for name, smiles in REFS.items():
            before = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()
            after = to_physiological(Chem.MolFromSmiles(smiles)).GetNumHeavyAtoms()
            self.assertEqual(before, after, name)


class TestTheKnownDiscrepancy(unittest.TestCase):
    """6MK0's head is a FULLY AROMATIC 1,8-naphthyridine, solution pKa about
    3.4, so the rule leaves it neutral. The structure shows both its
    nitrogens 2.60 and 2.85 A from Asp218's carboxylate oxygens, which a
    neutral acceptor cannot do. The rule is applied uniformly anyway and the
    disagreement is reported, rather than protonating one compound because
    its answer is known."""

    def test_the_aromatic_naphthyridine_is_left_neutral_by_the_rule(self):
        mol = protonate_bases(Chem.MolFromSmiles("c1ccc2cccnc2n1"))
        self.assertEqual(Chem.GetFormalCharge(mol), 0)


if __name__ == "__main__":
    unittest.main()
