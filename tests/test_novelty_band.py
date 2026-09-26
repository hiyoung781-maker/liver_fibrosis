import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from novelty import cross_scaffold_band, load_smi

CORE = "data/actives_core.smi"
EXTENDED = "data/actives_extended.smi"


class TestBandReproduction(unittest.TestCase):
    """Without normalization the band must reproduce the committed JSON exactly.

    That is the check that the reimplementation matches curate_actives.py:627-641.
    """

    def test_core_unnormalized_matches_committed_json(self):
        band = cross_scaffold_band([s for s, _ in load_smi(CORE)], normalize=False)
        self.assertEqual(band["n"], 29)
        self.assertAlmostEqual(band["p25"], 0.552, places=3)
        self.assertAlmostEqual(band["p50"], 0.676, places=3)

    def test_extended_unnormalized_matches_committed_json(self):
        band = cross_scaffold_band([s for s, _ in load_smi(EXTENDED)], normalize=False)
        self.assertEqual(band["n"], 190)
        self.assertAlmostEqual(band["p25"], 0.710, places=3)
        self.assertAlmostEqual(band["p50"], 0.775, places=3)


class TestBandAfterNormalization(unittest.TestCase):
    def test_core_is_unchanged(self):
        """0 of 29 core actives change tautomer, so the band must not move."""
        band = cross_scaffold_band([s for s, _ in load_smi(CORE)], normalize=True)
        self.assertAlmostEqual(band["p25"], 0.552, places=3)

    def test_extended_p25_drops_to_0680(self):
        """14 of 190 change tautomer; the threshold gets slightly stricter."""
        band = cross_scaffold_band([s for s, _ in load_smi(EXTENDED)], normalize=True)
        self.assertAlmostEqual(band["p25"], 0.680, places=3)
        self.assertLess(band["p25"], 0.710)


class TestLoadSmi(unittest.TestCase):
    def test_skips_comments_and_blank_lines(self):
        rows = load_smi("data/benchmark_panel.smi")
        self.assertEqual(len(rows), 6)
        self.assertIn("A1AFA", [label for _, label in rows])


if __name__ == "__main__":
    unittest.main()
