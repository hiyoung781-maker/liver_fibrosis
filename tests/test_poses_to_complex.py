import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from poses_to_complex import build_complex, build_many

RECEPTOR = "docking/v2/receptor_h.pdb"
GOOD_POSE = "tests/fixtures/pose_ok.pdbqt"
BROKEN_POSE = "tests/fixtures/pose_broken.pdbqt"


class TestComplexBuilding(unittest.TestCase):
    """PLIP computes donor-H...acceptor angles, so it needs explicit
    hydrogens. PDBQT merges nonpolar hydrogens into heavy atoms, so it
    cannot be handed to PLIP as-is."""

    def test_complex_contains_both_receptor_and_ligand(self):
        with TemporaryDirectory() as tmp:
            out = build_complex(GOOD_POSE, RECEPTOR, Path(tmp) / "c.pdb")
            text = Path(out).read_text()
            # LIG marks the ligand residue name; ATOM marks receptor atoms.
            # Both must be present for PLIP to see a binding site at all.
            self.assertIn("LIG", text)
            self.assertIn("ATOM", text)

    def test_ligand_has_explicit_hydrogens(self):
        with TemporaryDirectory() as tmp:
            out = build_complex(GOOD_POSE, RECEPTOR, Path(tmp) / "c.pdb")
            lig_lines = [l for l in Path(out).read_text().splitlines()
                         if "LIG" in l and l.startswith("HETATM")]
            # Without explicit H, PLIP cannot compute donor angles and its
            # hydrogen-bond verdicts silently change (documented failure:
            # 1,170/1,212 protein N atoms were mistyped as acceptors).
            self.assertTrue(any(l[76:78].strip() == "H" for l in lig_lines))


class TestFailureIsolation(unittest.TestCase):
    """One pose failing to convert must not stop a batch of ~85,000 poses
    from finishing; failures are recorded and the rest proceed."""

    def test_broken_pose_returns_none_instead_of_raising(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(
                build_complex(BROKEN_POSE, RECEPTOR, Path(tmp) / "c.pdb"))

    def test_build_many_reports_failures_and_keeps_going(self):
        with TemporaryDirectory() as tmp:
            result = build_many([GOOD_POSE, BROKEN_POSE, GOOD_POSE],
                                RECEPTOR, Path(tmp))
            self.assertEqual(len(result["built"]), 2)
            self.assertEqual(len(result["failed"]), 1)


if __name__ == "__main__":
    unittest.main()
