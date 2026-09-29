import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem

from deprotonate_acids import (ACID_SMARTS, deprotonate,
                               deprotonate_sdf, write_anion)

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


class TestBatchDeprotonation(unittest.TestCase):
    """The redocking gate adopted the anion (spec 7.4), so the docked library
    has to be in the same state as the control that validated the protocol.
    Generated ligands are prepared as neutral acids -- a survivor's PDBQT
    reads `REMARK SMILES ...C(=O)O...` -- and some carry more than one acid.
    This is a formal-charge edit on already-embedded 3D structures, never a
    re-embedding: the poses are compared to a crystal structure and the
    conformers are what docking starts from."""

    @staticmethod
    def _write(tmp, smiles_list):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        path = Path(tmp) / "in.sdf"
        writer = Chem.SDWriter(str(path))
        for i, smi in enumerate(smiles_list):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
            mol.SetProp("_Name", f"gen_{i:05d}")
            writer.write(mol)
        writer.close()
        return str(path)

    def _run(self, tmp, smiles_list):
        from rdkit import Chem
        out = str(Path(tmp) / "out.sdf")
        counts = deprotonate_sdf(self._write(tmp, smiles_list), out)
        mols = [m for m in Chem.SDMolSupplier(out, removeHs=False) if m]
        return counts, mols

    def test_every_acid_in_a_diacid_is_deprotonated(self):
        # The survivor that prompted this carries two: charge must be -2.
        from rdkit import Chem
        with TemporaryDirectory() as tmp:
            _c, mols = self._run(tmp, ["OC(=O)c1ccc(C(=O)O)cc1"])
            self.assertEqual(Chem.GetFormalCharge(mols[0]), -2)

    def test_labels_survive(self):
        # Every downstream stage keys on the molecule name.
        with TemporaryDirectory() as tmp:
            _c, mols = self._run(tmp, ["CC(=O)O", "OC(=O)c1ccccc1"])
            self.assertEqual([m.GetProp("_Name") for m in mols],
                             ["gen_00000", "gen_00001"])

    def test_a_molecule_without_an_acid_passes_through_unchanged(self):
        # deprotonate() raises on those by design; a library must not stop.
        from rdkit import Chem
        with TemporaryDirectory() as tmp:
            counts, mols = self._run(tmp, ["CC(=O)O", "c1ccccc1"])
            self.assertEqual(len(mols), 2)
            self.assertEqual(Chem.GetFormalCharge(mols[1]), 0)
            self.assertEqual(counts["unchanged"], 1)
            self.assertEqual(counts["deprotonated"], 1)

    def test_heavy_atom_coordinates_are_untouched(self):
        # The acidic hydrogens are removed -- that is the deprotonation -- so
        # the heavy-atom frame is what must not move. These conformers are
        # what docking starts from.
        from rdkit import Chem
        import numpy as np
        with TemporaryDirectory() as tmp:
            src = self._write(tmp, ["OC(=O)c1ccc(C(=O)O)cc1"])
            out = str(Path(tmp) / "out.sdf")
            deprotonate_sdf(src, out)
            before = Chem.RemoveHs(next(iter(
                Chem.SDMolSupplier(src, removeHs=False))))
            after = Chem.RemoveHs(next(iter(
                Chem.SDMolSupplier(out, removeHs=False))))
            self.assertEqual(before.GetNumAtoms(), after.GetNumAtoms())
            self.assertAlmostEqual(
                float(np.abs(before.GetConformer().GetPositions()
                             - after.GetConformer().GetPositions()).max()),
                0.0, places=9)

    def test_only_the_acidic_hydrogens_are_removed(self):
        from rdkit import Chem
        with TemporaryDirectory() as tmp:
            src = self._write(tmp, ["OC(=O)c1ccc(C(=O)O)cc1"])
            out = str(Path(tmp) / "out.sdf")
            deprotonate_sdf(src, out)
            before = next(iter(Chem.SDMolSupplier(src, removeHs=False)))
            after = next(iter(Chem.SDMolSupplier(out, removeHs=False)))
            self.assertEqual(before.GetNumAtoms() - after.GetNumAtoms(), 2)

    def test_counts_add_up_to_the_input(self):
        with TemporaryDirectory() as tmp:
            counts, _m = self._run(tmp, ["CC(=O)O", "c1ccccc1", "OC(=O)CC(=O)O"])
            self.assertEqual(counts["read"], 3)
            self.assertEqual(counts["deprotonated"] + counts["unchanged"]
                             + counts["failed"], counts["read"])


if __name__ == "__main__":
    unittest.main()
