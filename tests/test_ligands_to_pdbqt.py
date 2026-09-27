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

from ligands_to_pdbqt import prepare_one, write_pdbqt_set

SURVIVOR = "CCC1(CC(=O)O)c2cccn2-c2cccnc2CN1S(=O)(=O)c1ccc(-c2ccccc2)cc1F"


def embedded(smiles, name):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp("_Name", name)
    return mol


@unittest.skipUnless(HAVE_MEEKO, "meeko not installed")
class TestPrepareOne(unittest.TestCase):
    def test_produces_a_docking_ready_pdbqt(self):
        text = prepare_one(embedded(SURVIVOR, "gen_00001"))
        self.assertIsNotNone(text)
        for token in ("ROOT", "BRANCH", "TORSDOF"):
            self.assertIn(token, text)

    def test_embeds_the_smiles_for_the_return_trip(self):
        """Meeko writes REMARK SMILES into the PDBQT, which is what lets docked
        poses be exported back to SDF with real bond orders. Without it a pose read
        back with guessed bonds can lose the C(=O)O pattern, silently emptying the
        carboxylate measurement §8.5b depends on."""
        text = prepare_one(embedded(SURVIVOR, "gen_00001"))
        self.assertIn("REMARK SMILES", text)

    def test_returns_none_instead_of_raising_on_a_molecule_with_no_conformer(self):
        """A 7,767-ligand run must not abort on one awkward molecule."""
        flat = Chem.MolFromSmiles(SURVIVOR)
        self.assertIsNone(prepare_one(flat))


@unittest.skipUnless(HAVE_MEEKO, "meeko not installed")
class TestWritePdbqtSet(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sdf = str(Path(self.tmp.name) / "in.sdf")
        self.out = str(Path(self.tmp.name) / "pdbqt")
        writer = Chem.SDWriter(self.sdf)
        for i, smi in enumerate((SURVIVOR, "CC(=O)Oc1ccccc1C(=O)O"), start=1):
            writer.write(embedded(smi, f"gen_{i:05d}"))
        writer.close()

    def test_one_file_per_ligand_named_by_label(self):
        counts = write_pdbqt_set(self.sdf, self.out)
        self.assertEqual(counts["written"], 2)
        for name in ("gen_00001.pdbqt", "gen_00002.pdbqt"):
            self.assertTrue((Path(self.out) / name).is_file(), name)

    def test_index_file_lists_absolute_paths_one_per_line(self):
        """Uni-Dock's --ligand_index takes a file of ligand paths; relative paths
        would break as soon as the job runs from another directory."""
        counts = write_pdbqt_set(self.sdf, self.out)
        lines = Path(counts["index"]).read_text().split()
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertTrue(Path(line).is_absolute())
            self.assertTrue(Path(line).is_file())

    def test_labels_survive_into_filenames_so_results_can_be_keyed_back(self):
        counts = write_pdbqt_set(self.sdf, self.out)
        stems = sorted(Path(p).stem for p in
                       Path(counts["index"]).read_text().split())
        self.assertEqual(stems, ["gen_00001", "gen_00002"])

    def test_unnamed_molecule_is_counted_not_written(self):
        sdf = str(Path(self.tmp.name) / "unnamed.sdf")
        writer = Chem.SDWriter(sdf)
        mol = embedded(SURVIVOR, "")
        writer.write(mol)
        writer.close()
        counts = write_pdbqt_set(sdf, self.out)
        self.assertEqual(counts["written"], 0)
        self.assertEqual(counts["failures"], 1)


if __name__ == "__main__":
    unittest.main()
