import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from build_library import build_library, write_library
from library_report import (
    normalization_discrepancy,
    novelty_summary,
    objective_summary,
    percentiles,
    read_library,
)
from novelty import murcko

PLN_NONAROMATIC = "O=C(O)[C@H](CCCCCCCC1=CC=C2C(N1)=NCCC2)NC(C3(C)CCOCC3)=O"
PLN_AROMATIC = "CC1(C(=O)NC(CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1"
# 2,6-dichlorobenzamide-phenylalanine: A1AFA, a known active, carboxylate present.
A1AFA = "O=C(NC(Cc1ccccc1)C(=O)O)c1ccc(Cl)cc1Cl"
NO_CARBOXYLATE = "c1ccccc1CCN"


class TestPercentiles(unittest.TestCase):
    def test_reports_the_five_number_summary(self):
        summary = percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(summary["n"], 5)
        self.assertAlmostEqual(summary["median"], 3.0)
        self.assertAlmostEqual(summary["p10"], 1.4)
        self.assertAlmostEqual(summary["p90"], 4.6)

    def test_empty_input_does_not_crash(self):
        """An arm can legitimately come back empty from a filter; the report must
        still print rather than raise halfway through."""
        summary = percentiles([])
        self.assertEqual(summary["n"], 0)
        self.assertIsNone(summary["median"])


class TestObjectiveSummary(unittest.TestCase):
    def test_carboxylate_and_alert_rates(self):
        summary = objective_summary([A1AFA, NO_CARBOXYLATE])
        self.assertAlmostEqual(summary["carboxylate_rate"], 0.5)
        self.assertEqual(summary["n"], 2)

    def test_aniline_is_counted_as_an_alert(self):
        """NO_CARBOXYLATE is a phenethylamine, not an aniline, so it must NOT
        trip the narrowed primary-aniline alert - the guard against reverting to
        the [NH2,NH][c] pattern that zeroed 86% of actives_extended."""
        summary = objective_summary([NO_CARBOXYLATE])
        self.assertAlmostEqual(summary["alert_free_rate"], 1.0)
        aniline = objective_summary(["Nc1ccccc1C(=O)O"])
        self.assertAlmostEqual(aniline["alert_free_rate"], 0.0)


class TestNoveltySummary(unittest.TestCase):
    def test_known_scaffold_is_not_novel(self):
        known = {murcko(Chem.MolFromSmiles(A1AFA))}
        summary = novelty_summary([A1AFA, "CCOc1ccc2nc(N)sc2c1"], known)
        self.assertAlmostEqual(summary["murcko_novel_rate"], 0.5)
        self.assertEqual(summary["n_unique_scaffolds"], 2)

    def test_does_not_renormalize_its_input(self):
        """library.smi column 1 is already tautomer-canonical. Re-running
        canonical_tautomer here would repeat the whole ~0.14 s/molecule cost the
        build step already paid - 47 min for a 20,000-molecule arm.
        """
        import normalize

        before = len(normalize.TAUTOMER_FAILURES)
        calls = []
        original = normalize._TAUTOMER.Canonicalize
        normalize._TAUTOMER.Canonicalize = lambda m: (calls.append(1), original(m))[1]
        try:
            novelty_summary([A1AFA], set())
        finally:
            normalize._TAUTOMER.Canonicalize = original
        self.assertEqual(calls, [])
        self.assertEqual(len(normalize.TAUTOMER_FAILURES), before)


class TestNormalizationDiscrepancy(unittest.TestCase):
    def test_flags_a_molecule_whose_objective_score_moves(self):
        """The point of keeping both columns: the RL objective scored column 2,
        every downstream claim is about column 1, and the two can disagree."""
        rows = build_library([PLN_NONAROMATIC])["rows"]
        normalized, raw = rows[0]
        self.assertNotEqual(normalized, raw)
        summary = normalization_discrepancy(rows)
        self.assertEqual(summary["n"], 1)
        self.assertIn("carboxylate_changed", summary)
        self.assertIn("total_score_shift", summary)

    def test_identical_columns_show_no_shift(self):
        rows = [(A1AFA, A1AFA)]
        summary = normalization_discrepancy(rows)
        self.assertEqual(summary["carboxylate_changed"], 0)
        self.assertAlmostEqual(summary["total_score_shift"]["median"], 0.0)
        self.assertEqual(summary["n_shifted"], 0)
        self.assertAlmostEqual(summary["max_shift"], 0.0)

    def test_a_rare_shift_is_counted_not_averaged_away(self):
        """Percentiles alone hid this. On the real 19,485-molecule library the
        shift distribution reads 0.00 at every percentile through p90 while 31
        molecules actually flip their alert verdict - the tail is the whole
        effect. n_shifted and max_shift are what make it visible.
        """
        rows = [(A1AFA, A1AFA)] * 99 + [(NO_CARBOXYLATE, A1AFA)]
        summary = normalization_discrepancy(rows)
        self.assertAlmostEqual(summary["total_score_shift"]["p90"], 0.0)
        self.assertEqual(summary["n_shifted"], 1)
        self.assertGreater(summary["max_shift"], 0.0)


class TestReadLibrary(unittest.TestCase):
    def test_reads_both_columns_written_by_build_library(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "library.smi")
            result = build_library([PLN_AROMATIC, A1AFA])
            write_library(path, result, source="test")
            self.assertEqual(read_library(path), result["rows"])


if __name__ == "__main__":
    unittest.main()
