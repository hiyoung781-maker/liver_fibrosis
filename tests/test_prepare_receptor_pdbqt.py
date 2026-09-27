import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prepare_receptor_pdbqt import (
    ANCHORS,
    TOLERANCE,
    compare_anchors,
    read_pdb_atoms,
    read_pdbqt_atoms,
)

PDB = "docking/receptor.pdb"
PDBQT = "docking/receptor.pdbqt"


class TestAnchorDefinition(unittest.TestCase):
    def test_anchors_name_chain_b_explicitly(self):
        """Chain A holds four more calciums, 35-50 A from the ligand. An earlier
        analysis in this project searched chain A and got exactly those numbers, so
        the chain is part of the key, not an afterthought."""
        for chain, _, _, _ in ANCHORS:
            self.assertEqual(chain, "B")

    def test_tolerance_is_tight(self):
        """A rigid-receptor conversion must not move an atom at all; anything above
        a hundredth of an angstrom means the converter did something unexpected."""
        self.assertLessEqual(TOLERANCE, 0.01)


class TestParsers(unittest.TestCase):
    def test_reads_both_atom_and_hetatm(self):
        """obabel reclassified the six calcium HETATM records as ATOM. Code that
        looks only at HETATM loses the MIDAS metal silently."""
        atoms = read_pdb_atoms(PDB)
        records = {a["record"] for a in atoms}
        self.assertIn("ATOM", records)
        self.assertIn("HETATM", records)
        calciums = [a for a in atoms if a["resname"] == "CA"]
        self.assertEqual(len(calciums), 6)

    def test_pdbqt_parser_extracts_the_autodock_type(self):
        atoms = read_pdbqt_atoms(PDBQT)
        ca = [a for a in atoms if a["chain"] == "B" and a["resseq"] == "501"]
        self.assertEqual(len(ca), 1)
        self.assertEqual(ca[0]["adtype"], "Ca")


class TestConversionPreservedTheAnchors(unittest.TestCase):
    """The whole point. If this fails, docking must not proceed: the section 8.5b
    filter measures carboxylate-O to Ca501, so a lost or displaced calcium makes
    every pose fail a distance test and the failure reads as chemistry."""

    @classmethod
    def setUpClass(cls):
        cls.rows = compare_anchors(read_pdb_atoms(PDB), read_pdbqt_atoms(PDBQT))

    def test_both_anchors_present_before_and_after(self):
        for row in self.rows:
            with self.subTest(anchor=row["label"]):
                self.assertTrue(row["in_source"])
                self.assertTrue(row["in_converted"])

    def test_neither_anchor_moved(self):
        for row in self.rows:
            with self.subTest(anchor=row["label"]):
                self.assertIsNotNone(row["displacement"])
                self.assertLessEqual(row["displacement"], TOLERANCE)

    def test_midas_calcium_is_typed_as_calcium(self):
        ca = next(r for r in self.rows if r["label"].startswith("Ca501"))
        self.assertEqual(ca["adtype"], "Ca")

    def test_asn224_oxygen_is_typed_as_an_acceptor(self):
        """OA is AutoDock's H-bond-accepting oxygen. The filter's second criterion
        is a ligand donor reaching this atom, so an inert 'O' type would be wrong."""
        o = next(r for r in self.rows if r["label"].startswith("Asn224"))
        self.assertEqual(o["adtype"], "OA")

    def test_no_atoms_were_lost(self):
        self.assertEqual(len(read_pdb_atoms(PDB)), len(read_pdbqt_atoms(PDBQT)))

    def test_all_six_calciums_survived(self):
        after = [a for a in read_pdbqt_atoms(PDBQT) if a.get("adtype") == "Ca"]
        self.assertEqual(len(after), 6)


class TestDetectsBreakage(unittest.TestCase):
    """compare_anchors has to report failure, not just success, or it is decoration."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = read_pdb_atoms(PDB)

    def _pdbqt_without_midas(self):
        return [a for a in read_pdbqt_atoms(PDBQT)
                if not (a["chain"] == "B" and a["resseq"] == "501")]

    def test_reports_a_lost_calcium(self):
        rows = compare_anchors(self.source, self._pdbqt_without_midas())
        ca = next(r for r in rows if r["label"].startswith("Ca501"))
        self.assertTrue(ca["in_source"])
        self.assertFalse(ca["in_converted"])
        self.assertIsNone(ca["displacement"])

    def test_reports_a_displaced_calcium(self):
        atoms = read_pdbqt_atoms(PDBQT)
        for a in atoms:
            if a["chain"] == "B" and a["resseq"] == "501":
                a["xyz"] = a["xyz"] + 0.5
        rows = compare_anchors(self.source, atoms)
        ca = next(r for r in rows if r["label"].startswith("Ca501"))
        self.assertGreater(ca["displacement"], TOLERANCE)


if __name__ == "__main__":
    unittest.main()
