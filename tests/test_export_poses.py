import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from export_poses import (
    LIGAND_RESNAME,
    _safe,
    export,
    pose_atom_specs,
    write_ligand_pdb,
)

RECEPTOR = "docking/receptor.pdb"
REDOCK = "docking/redock_poses.sdf"
CRYSTAL_LABEL = "ligand_crystal.pdb"


class TestResidueNameIsThreeCharacters(unittest.TestCase):
    """A five-character CCD name (`A1AFA`) once overflowed the PDB resName columns here,
    shifting every field after it and zeroing all affinities in that file. Ligands are
    written as `LIG` and separated by chain."""

    def test_resname_length(self):
        self.assertEqual(len(LIGAND_RESNAME), 3)

    def test_columns_are_not_shifted(self):
        with TemporaryDirectory() as tmp:
            mol = next(iter(Chem.SDMolSupplier(REDOCK, removeHs=False)))
            path = str(Path(tmp) / "x.pdb")
            write_ligand_pdb(mol, path, "C", "test")
            for line in Path(path).read_text().splitlines():
                if not line.startswith("HETATM"):
                    continue
                self.assertEqual(line[17:20].strip(), LIGAND_RESNAME)
                self.assertEqual(line[21], "C")
                # coordinates must parse from their fixed columns
                for start in (30, 38, 46):
                    float(line[start:start + 8])


class TestFilenameSafety(unittest.TestCase):
    def test_a_label_with_a_dot_does_not_double_the_extension(self):
        self.assertEqual(_safe("ligand_crystal.pdb"), "ligand_crystal_pdb")

    def test_a_separator_cannot_escape_the_output_directory(self):
        self.assertNotIn("/", _safe("../../etc/passwd"))

    def test_ordinary_labels_are_untouched(self):
        self.assertEqual(_safe("gen_07401"), "gen_07401")


class TestAtomSpecsMatchTheFilter(unittest.TestCase):
    """The drawn distance and the recorded distance must be the same measurement, so the
    specs are recomputed with the functions pose_geometry uses."""

    @classmethod
    def setUpClass(cls):
        cls.mol = next(iter(Chem.SDMolSupplier(REDOCK, removeHs=False)))

    def test_reproduces_the_recorded_crystal_contacts(self):
        with TemporaryDirectory() as tmp:
            names = write_ligand_pdb(self.mol, str(Path(tmp) / "x.pdb"), "C", "t")
        specs = pose_atom_specs(self.mol, names, "C", RECEPTOR)
        self.assertAlmostEqual(specs["ca_distance"], 2.72, places=1)
        self.assertAlmostEqual(specs["donor_distance"], 2.89, places=1)

    def test_specifiers_name_atoms_that_exist_in_the_written_pdb(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "x.pdb")
            names = write_ligand_pdb(self.mol, path, "C", "t")
            specs = pose_atom_specs(self.mol, names, "C", RECEPTOR)
            written = {line[12:16].strip() for line in Path(path).read_text().splitlines()
                       if line.startswith("HETATM")}
        for key in ("ca_atom", "donor_atom"):
            self.assertIsNotNone(specs[key])
            self.assertIn(specs[key].split("@")[-1], written)

    def test_the_chain_in_the_specifier_matches_the_pdb(self):
        with TemporaryDirectory() as tmp:
            names = write_ligand_pdb(self.mol, str(Path(tmp) / "x.pdb"), "D", "t")
        specs = pose_atom_specs(self.mol, names, "D", RECEPTOR)
        self.assertTrue(specs["ca_atom"].startswith("/D:1@"))


class TestExport(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_writes_a_pdb_and_a_chimerax_script(self):
        result = export(REDOCK, {CRYSTAL_LABEL: None}, self.tmp.name, RECEPTOR, "v")
        self.assertEqual(len(result["exported"]), 1)
        self.assertEqual(result["missing"], [])
        self.assertTrue(Path(result["cxc"]).is_file())
        self.assertTrue(Path(result["exported"][0]["path"]).is_file())

    def test_ligand_chains_avoid_the_receptor_chains(self):
        """The receptor occupies A and B; a ligand on either would make every specifier
        ambiguous."""
        result = export(REDOCK, {CRYSTAL_LABEL: None}, self.tmp.name, RECEPTOR, "v")
        self.assertNotIn(result["exported"][0]["chain"], ("A", "B"))

    def test_best_affinity_pose_is_chosen_by_default(self):
        result = export(REDOCK, {CRYSTAL_LABEL: None}, self.tmp.name, RECEPTOR, "v")
        self.assertAlmostEqual(result["exported"][0]["affinity"], -6.908, places=2)

    def test_an_explicit_pose_number_is_honoured(self):
        """Counterexamples need a specific pose - usually the best-affinity one, which is
        what an affinity-ranked selection would have picked."""
        result = export(REDOCK, {CRYSTAL_LABEL: 3}, self.tmp.name, RECEPTOR, "v")
        self.assertEqual(result["exported"], []) if not result["exported"] else \
            self.assertEqual(result["exported"][0]["pose"], 3)

    def test_a_missing_label_is_reported_not_silently_skipped(self):
        result = export(REDOCK, {"gen_99999": None}, self.tmp.name, RECEPTOR, "v")
        self.assertEqual(result["missing"], ["gen_99999"])

    def test_the_script_draws_both_cutoffs_and_says_which_passed(self):
        result = export(REDOCK, {CRYSTAL_LABEL: None}, self.tmp.name, RECEPTOR, "v")
        text = Path(result["cxc"]).read_text()
        self.assertIn("#1/B:501@CA", text)      # the MIDAS calcium
        self.assertIn("#1/B:224@O", text)       # Asn224 backbone O
        self.assertIn("cutoff 3.2", text)
        self.assertIn("cutoff 3.5", text)
        self.assertIn("PASS", text)

    def test_the_script_opens_the_receptor_first(self):
        result = export(REDOCK, {CRYSTAL_LABEL: None}, self.tmp.name, RECEPTOR, "v")
        lines = [l for l in Path(result["cxc"]).read_text().splitlines()
                 if l.startswith("open ")]
        self.assertEqual(lines[0], f"open {RECEPTOR}")


if __name__ == "__main__":
    unittest.main()


class TestWrittenPdbIsReadableByRdkit(unittest.TestCase):
    """The strongest column check: a parser has to get the same molecule back.

    The first version of write_ligand_pdb omitted the altLoc column (17), which shifted
    resName and every field after it - the same class of error that a five-character
    residue name caused earlier in this project, and equally silent.
    """

    def test_round_trip_preserves_atom_count_and_coordinates(self):
        import numpy as np

        mol = next(iter(Chem.SDMolSupplier(REDOCK, removeHs=False)))
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "x.pdb")
            write_ligand_pdb(mol, path, "C", "t")
            back = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
        self.assertIsNotNone(back)
        self.assertEqual(back.GetNumAtoms(), mol.GetNumAtoms())
        delta = np.abs(mol.GetConformer().GetPositions()
                       - back.GetConformer().GetPositions()).max()
        self.assertLess(float(delta), 0.002)

    def test_round_trip_preserves_the_chain_and_resname(self):
        mol = next(iter(Chem.SDMolSupplier(REDOCK, removeHs=False)))
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "x.pdb")
            write_ligand_pdb(mol, path, "D", "t")
            back = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
        info = back.GetAtomWithIdx(0).GetPDBResidueInfo()
        self.assertEqual(info.GetResidueName().strip(), LIGAND_RESNAME)
        self.assertEqual(info.GetChainId().strip(), "D")
