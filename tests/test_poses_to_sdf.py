import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

try:
    import meeko  # noqa: F401
    HAVE_MEEKO = True
except Exception:
    HAVE_MEEKO = False

from ligands_to_pdbqt import prepare_one
from poses_to_sdf import convert_dir, poses_from_pdbqt

SURVIVOR = "CCC1(CC(=O)O)c2cccn2-c2cccnc2CN1S(=O)(=O)c1ccc(-c2ccccc2)cc1F"
COOH = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")


def write_pose_pdbqt(directory, label):
    """A Uni-Dock output PDBQT is a docked ligand PDBQT; Meeko's writer produces the
    same format, which is what makes this a fair stand-in."""
    mol = Chem.AddHs(Chem.MolFromSmiles(SURVIVOR))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp("_Name", label)
    path = Path(directory) / f"{label}_out.pdbqt"
    path.write_text(prepare_one(mol))
    return str(path)


@unittest.skipUnless(HAVE_MEEKO, "meeko not installed")
class TestRoundTripPreservesChemistry(unittest.TestCase):
    """The reason this module exists. Measured behaviour:

      RDKit reading the pose PDBQT directly -> unparseable, because AutoDock's
          aromatic-carbon type 'A' is not an element.
      Meeko round trip                      -> carboxylate intact.

    pose_geometry.py finds the MIDAS anchor with a SMARTS match, so a pose read
    without its chemistry yields zero matches and the filter reports every pose as
    failing - a result that looks chemical and is not.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = write_pose_pdbqt(self.tmp.name, "gen_00001")

    def test_carboxylate_survives(self):
        poses = poses_from_pdbqt(self.path)
        self.assertTrue(poses)
        self.assertEqual(len(poses[0].GetSubstructMatches(COOH)), 1)

    def test_naive_rdkit_read_of_the_same_file_fails(self):
        """Documents the counterfactual, so nobody 'simplifies' this away."""
        naive = Chem.MolFromPDBFile(self.path, removeHs=False, sanitize=False)
        if naive is not None:
            self.assertEqual(len(naive.GetSubstructMatches(COOH)), 0)

    def test_poses_carry_coordinates(self):
        pose = poses_from_pdbqt(self.path)[0]
        self.assertEqual(pose.GetNumConformers(), 1)

    def test_label_is_recovered_from_the_filename(self):
        """Uni-Dock names outputs after the input, so the survivor label arrives via
        the filename. It is the key every downstream artifact joins on."""
        pose = poses_from_pdbqt(self.path)[0]
        self.assertEqual(pose.GetProp("_Name"), "gen_00001")

    def test_explicit_label_overrides_the_filename(self):
        pose = poses_from_pdbqt(self.path, label="gen_99999")[0]
        self.assertEqual(pose.GetProp("_Name"), "gen_99999")

    def test_unreadable_file_returns_empty_not_raise(self):
        bad = Path(self.tmp.name) / "broken_out.pdbqt"
        bad.write_text("this is not a pdbqt\n")
        self.assertEqual(poses_from_pdbqt(str(bad)), [])


@unittest.skipUnless(HAVE_MEEKO, "meeko not installed")
class TestConvertDir(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for label in ("gen_00001", "gen_00002"):
            write_pose_pdbqt(self.tmp.name, label)
        self.out = str(Path(self.tmp.name) / "poses.sdf")

    def test_writes_one_sdf_holding_every_pose(self):
        counts = convert_dir(self.tmp.name, self.out)
        self.assertEqual(counts["files"], 2)
        self.assertEqual(counts["poses"], 2)
        self.assertEqual(counts["failed_files"], 0)

    def test_output_is_readable_by_the_geometry_filter(self):
        """End-to-end: the SDF this writes must be consumable by pose_geometry, with
        the carboxylate still matchable. That is the whole chain in one assertion."""
        from pose_geometry import anchor_atoms, measure_pose

        convert_dir(self.tmp.name, self.out)
        anchors = anchor_atoms("docking/receptor.pdb")
        mols = [m for m in Chem.SDMolSupplier(self.out, removeHs=False)
                if m is not None]
        self.assertEqual(len(mols), 2)
        for mol in mols:
            result = measure_pose(mol, anchors)
            self.assertGreater(result["n_carboxylate_o"], 0)
            self.assertGreater(result["n_donors"], 0)

    def test_labels_are_preserved_through_the_sdf(self):
        convert_dir(self.tmp.name, self.out)
        names = sorted(m.GetProp("_Name") for m in
                       Chem.SDMolSupplier(self.out, removeHs=False) if m)
        self.assertEqual(names, ["gen_00001", "gen_00002"])

    def test_counts_a_broken_file_without_aborting(self):
        (Path(self.tmp.name) / "broken_out.pdbqt").write_text("garbage\n")
        counts = convert_dir(self.tmp.name, self.out)
        self.assertEqual(counts["files"], 3)
        self.assertEqual(counts["poses"], 2)
        self.assertEqual(counts["failed_files"], 1)
        self.assertIn("broken_out.pdbqt", counts["failures"])


if __name__ == "__main__":
    unittest.main()
