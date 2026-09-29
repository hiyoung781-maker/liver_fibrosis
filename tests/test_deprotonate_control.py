import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem

from deprotonate_control import ACID_SMARTS, deprotonate, write_anion

REFERENCE = "docking/ligand_ref.sdf"


class TestDeprotonate(unittest.TestCase):
    """spec 7.4 pre-registers a redock in BOTH protonation states and prefers
    the anion, because AutoDock4 has an electrostatic term and the MIDAS ion
    is Ca2+ -- the reason v1 chose the neutral acid (Vina has no electrostatics)
    does not survive the engine change. The v2 batch docked the neutral form,
    so the anion arm of that gate had never been built."""

    def test_reference_starts_neutral(self):
        mol = Chem.MolFromMolFile(REFERENCE)
        self.assertEqual(Chem.GetFormalCharge(mol), 0)

    def test_every_carboxylic_acid_is_deprotonated(self):
        mol = Chem.MolFromMolFile(REFERENCE)
        anion = deprotonate(mol)
        self.assertEqual(Chem.GetFormalCharge(anion), -1)
        self.assertFalse(anion.GetSubstructMatches(
            Chem.MolFromSmarts(ACID_SMARTS)))

    def test_coordinates_are_untouched(self):
        # The control exists to hold the experimentally observed pose. A
        # deprotonation that moved an atom would destroy the comparison the
        # RMSD criterion makes.
        mol = Chem.MolFromMolFile(REFERENCE)
        before = mol.GetConformer().GetPositions()
        after = deprotonate(mol).GetConformer().GetPositions()
        self.assertEqual(before.shape, after.shape)
        self.assertAlmostEqual(float(abs(before - after).max()), 0.0, places=9)

    def test_heavy_atom_count_is_unchanged(self):
        mol = Chem.MolFromMolFile(REFERENCE)
        self.assertEqual(deprotonate(mol).GetNumAtoms(), mol.GetNumAtoms())

    def test_stereocentre_survives(self):
        # 8W30's ligand is one enantiomer of a racemate; losing the tag would
        # silently redock the mirror image and inflate the RMSD.
        mol = Chem.MolFromMolFile(REFERENCE)
        tags = [a.GetChiralTag() for a in mol.GetAtoms()]
        self.assertEqual([a.GetChiralTag() for a in deprotonate(mol).GetAtoms()],
                         tags)

    def test_a_molecule_without_an_acid_raises(self):
        # Silently writing an unchanged file would make the anion arm of the
        # gate a duplicate of the neutral arm, reported as an independent run.
        with self.assertRaises(ValueError):
            deprotonate(Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")))

    def test_write_anion_round_trips_through_an_sdf(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "anion.sdf"
            write_anion(REFERENCE, str(out))
            back = Chem.MolFromMolFile(str(out))
            self.assertEqual(Chem.GetFormalCharge(back), -1)


if __name__ == "__main__":
    unittest.main()
