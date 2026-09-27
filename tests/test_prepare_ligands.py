import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from prepare_ligands import CONTROL_LABEL, CONTROL_SHARD, embed, write_shards

SURVIVORS = "data/survivors.smi"
CONTROL = "docking/ligand_ref.sdf"


class TestEmbed(unittest.TestCase):
    def test_produces_one_conformer_with_hydrogens(self):
        mol = embed("CCC1(CC(=O)O)c2cccn2-c2cccnc2CN1S(=O)(=O)c1ccc(-c2ccccc2)cc1F")
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetNumConformers(), 1)
        self.assertTrue(any(a.GetSymbol() == "H" for a in mol.GetAtoms()))

    def test_returns_none_on_garbage_instead_of_raising(self):
        """A 7,767-ligand run must not abort on one bad SMILES."""
        self.assertIsNone(embed("not_a_smiles"))

    def test_is_deterministic(self):
        a = embed("CC(=O)Oc1ccccc1C(=O)O")
        b = embed("CC(=O)Oc1ccccc1C(=O)O")
        self.assertAlmostEqual(
            float(abs(a.GetConformer().GetPositions()
                      - b.GetConformer().GetPositions()).max()), 0.0, places=6)


class TestControlShard(unittest.TestCase):
    """The control ligand is what makes the run self-validating: the crystal ligand
    docked with the production settings, in the same batch. It has to be findable in
    the results, which means it needs a label of its own."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        smi = Path(self.tmp.name) / "few.smi"
        smi.write_text(
            "CC(=O)Oc1ccccc1C(=O)O\tgen_00001\n"
            "O=C(O)Cc1ccccc1\tgen_00002\n"
        )
        self.smi = str(smi)
        self.out = str(Path(self.tmp.name) / "ligands")

    def test_control_is_relabelled_not_copied_verbatim(self):
        """The source SDF's own title is 'ligand_crystal.pdb', which would otherwise
        propagate into the geometry CSV as a filename-looking string."""
        write_shards(self.smi, self.out, shard_size=10, control_sdf=CONTROL)
        mol = next(iter(Chem.SDMolSupplier(str(Path(self.out) / CONTROL_SHARD),
                                          removeHs=False)))
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetProp("_Name"), CONTROL_LABEL)
        self.assertNotEqual(mol.GetProp("_Name"), "ligand_crystal.pdb")

    def test_control_label_sorts_apart_from_generated_labels(self):
        self.assertFalse(CONTROL_LABEL.startswith("gen_"))

    def test_control_keeps_its_crystal_coordinates(self):
        """It is a control precisely because it is the experimentally observed pose;
        re-embedding it would destroy the comparison.

        The invariant is the HEAVY-ATOM coordinates, not the atom count: hydrogens are
        added (with coordinates) so Meeko will accept the molecule, which takes it from
        22 atoms to 35. An earlier version of this test asserted equal atom counts and
        so conflated two different things.
        """
        original = Chem.MolFromMolFile(CONTROL, removeHs=False)
        write_shards(self.smi, self.out, shard_size=10, control_sdf=CONTROL)
        copied = next(iter(Chem.SDMolSupplier(str(Path(self.out) / CONTROL_SHARD),
                                              removeHs=False)))
        heavy = Chem.RemoveHs(copied)
        self.assertEqual(original.GetNumAtoms(), heavy.GetNumAtoms())
        delta = abs(original.GetConformer().GetPositions()
                    - heavy.GetConformer().GetPositions()).max()
        self.assertAlmostEqual(float(delta), 0.0, places=6)

    def test_control_shard_is_first_in_the_shard_list(self):
        counts = write_shards(self.smi, self.out, shard_size=10, control_sdf=CONTROL)
        shards = (Path(self.out) / "shards.txt").read_text().split()
        self.assertEqual(shards[0], CONTROL_SHARD)
        self.assertEqual(counts["shards"], len(shards))

    def test_unreadable_control_raises_rather_than_skipping_silently(self):
        with self.assertRaises(ValueError):
            write_shards(self.smi, self.out, control_sdf="docking/receptor.pdb")


class TestSharding(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        smi = Path(self.tmp.name) / "five.smi"
        smi.write_text("".join(
            f"O=C(O)C{'C' * i}c1ccccc1\tgen_{i:05d}\n" for i in range(1, 6)))
        self.smi = str(smi)
        self.out = str(Path(self.tmp.name) / "ligands")

    def test_splits_at_the_requested_size(self):
        counts = write_shards(self.smi, self.out, shard_size=2, control_sdf=None)
        self.assertEqual(counts["ligands"], 5)
        self.assertEqual(counts["shards"], 3)   # 2 + 2 + 1

    def test_labels_are_carried_onto_every_molecule(self):
        write_shards(self.smi, self.out, shard_size=2, control_sdf=None)
        names = []
        for shard in sorted(Path(self.out).glob("shard_*.sdf")):
            names += [m.GetProp("_Name")
                      for m in Chem.SDMolSupplier(str(shard), removeHs=False) if m]
        self.assertEqual(sorted(names),
                         [f"gen_{i:05d}" for i in range(1, 6)])

    def test_embedded_count_matches_what_was_written(self):
        counts = write_shards(self.smi, self.out, shard_size=2, control_sdf=None)
        written = sum(
            1 for shard in Path(self.out).glob("shard_*.sdf")
            for m in Chem.SDMolSupplier(str(shard), removeHs=False) if m)
        self.assertEqual(counts["embedded"], written)


if __name__ == "__main__":
    unittest.main()


class TestControlLigandConvertsToPdbqt(unittest.TestCase):
    """The control was the one ligand that failed PDBQT conversion while all 7,762
    generated ones succeeded. Cause: the crystal ligand arrives from the PDB with
    heavy atoms only, and Meeko requires explicit hydrogens; generated ligands get
    them from embed(), and the control bypassed it."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = str(Path(self.tmp.name) / "ligands")

    def test_control_shard_has_explicit_hydrogens(self):
        from prepare_ligands import write_control_shard

        write_control_shard(self.out, CONTROL)
        mol = next(iter(Chem.SDMolSupplier(str(Path(self.out) / CONTROL_SHARD),
                                          removeHs=False)))
        hydrogens = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "H")
        self.assertGreater(hydrogens, 0)

    def test_heavy_atom_coordinates_are_bit_identical(self):
        """addCoords=True must place hydrogens without moving anything else - the
        crystal pose is the whole value of this molecule."""
        import numpy as np
        from prepare_ligands import write_control_shard

        original = Chem.MolFromMolFile(CONTROL, removeHs=False)
        write_control_shard(self.out, CONTROL)
        written = next(iter(Chem.SDMolSupplier(str(Path(self.out) / CONTROL_SHARD),
                                              removeHs=False)))
        heavy = Chem.RemoveHs(written)
        self.assertEqual(heavy.GetNumAtoms(), original.GetNumAtoms())
        delta = np.abs(original.GetConformer().GetPositions()
                       - heavy.GetConformer().GetPositions()).max()
        self.assertAlmostEqual(float(delta), 0.0, places=6)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("meeko") is not None,
        "meeko not installed")
    def test_control_now_survives_meeko_conversion(self):
        """The regression this fix exists for, asserted end to end."""
        from ligands_to_pdbqt import prepare_one
        from prepare_ligands import write_control_shard

        write_control_shard(self.out, CONTROL)
        mol = next(iter(Chem.SDMolSupplier(str(Path(self.out) / CONTROL_SHARD),
                                          removeHs=False)))
        self.assertIsNotNone(prepare_one(mol))

    def test_the_unhydrogenated_crystal_ligand_would_have_failed(self):
        """Documents the cause so the AddHs call is not 'simplified' away later."""
        raw = Chem.MolFromMolFile(CONTROL, removeHs=False)
        self.assertEqual(sum(1 for a in raw.GetAtoms() if a.GetSymbol() == "H"), 0)
