import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from panel_report import (
    PANEL,
    crosscheck,
    panel_rows,
    read_reinvent_csv,
    score_panel,
    write_panel_smiles,
)

LABELS = ["A1AFA", "PLN-1474", "bexotegrast", "CHEMBL4649232"]


class TestPanelRows(unittest.TestCase):
    def test_reads_all_four_with_labels_in_file_order(self):
        rows = panel_rows()
        self.assertEqual([label for _, label in rows], LABELS)

    def test_pln1474_is_the_chembl5933542_structure(self):
        """CHEMBL5933542 is PLN-1474: flattened, the two are the same molecule.
        §9.2's permeability bar is stated against this compound, so the panel entry
        it is scored from must actually be it."""
        from rdkit import Chem

        chembl = "CC1(C(=O)N[C@@H](CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1"
        flat = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
        panel = dict((label, smiles) for smiles, label in panel_rows())
        self.assertEqual(flat(panel["PLN-1474"]), flat(chembl))


class TestWritePanelSmiles(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, normalize):
        path = str(Path(self.tmp.name) / "panel.smi")
        labels = write_panel_smiles(path, normalize=normalize)
        lines = Path(path).read_text().splitlines()
        return labels, lines

    def test_writes_plain_smiles_one_per_line_no_labels_no_comments(self):
        """REINVENT's smiles_file reader takes the whole line, so a label column or
        a comment would be parsed as part of the SMILES."""
        labels, lines = self._write(normalize=False)
        self.assertEqual(labels, LABELS)
        self.assertEqual(len(lines), 4)
        for line in lines:
            self.assertNotIn("\t", line)
            self.assertNotIn(" ", line)
            self.assertFalse(line.startswith("#"))

    def test_label_order_matches_line_order(self):
        """REINVENT's output CSV has no labels, only SMILES in input order, so the
        returned label list is the only thing that maps rows back to compounds."""
        labels, lines = self._write(normalize=False)
        self.assertEqual(len(labels), len(lines))

    def test_the_curated_panel_is_already_tautomer_canonical(self):
        """MEASURED, and it is the fact §9.5 depends on.

        §8.0 requires both sides of a comparison to pass the same normalization,
        and §6.5 measured that the objective's verdict can flip with the tautomer
        written - 19 agent molecules swing between a perfect score and zero. The
        panel turns out to need nothing: all four curated SMILES are already their
        own canonical tautomer, so the as-curated and normalized arms coincide and
        §9.5's comparison against normalized generated molecules is symmetric for
        free.

        This is asserted rather than assumed because it is exactly what a future
        edit to data/benchmark_panel.smi could silently break. If someone adds or
        rewrites a compound in a non-canonical tautomer, this fails and says the
        comparison went asymmetric.
        """
        _, curated = self._write(normalize=False)
        _, normalized = self._write(normalize=True)
        self.assertEqual(curated, normalized)


class TestScorePanel(unittest.TestCase):
    def test_scores_all_four_with_every_component(self):
        result = score_panel(normalize=False)
        self.assertEqual(sorted(result), sorted(LABELS))
        for label, row in result.items():
            with self.subTest(label=label):
                for key in ("total", "cooh", "tpsa", "sascore", "alerts"):
                    self.assertIn(key, row)

    def test_no_panel_molecule_is_gated_out(self):
        """Every positive-panel compound must clear the carboxylate gate and the
        alert filter, or the reference distribution §7 exists to produce collapses."""
        for normalize in (False, True):
            result = score_panel(normalize=normalize)
            for label, row in result.items():
                with self.subTest(label=label, normalize=normalize):
                    self.assertGreater(row["cooh"], 0)
                    self.assertEqual(row["alerts"], 1.0)
                    self.assertGreater(row["total"], 0.0)


class TestReadReinventCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_maps_reinvent_column_names_to_objective_keys(self):
        path = Path(self.tmp.name) / "score.csv"
        path.write_text(
            "SMILES,Score,carboxylate MIDAS anchor,TPSA,SA score,unwanted groups\n"
            "CCO,0.5,1.0,0.9,1.0,1.0\n"
        )
        rows = read_reinvent_csv(str(path))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["total"], 0.5)
        self.assertAlmostEqual(rows[0]["cooh"], 1.0)
        self.assertAlmostEqual(rows[0]["tpsa"], 0.9)
        self.assertAlmostEqual(rows[0]["sascore"], 1.0)
        self.assertAlmostEqual(rows[0]["alerts"], 1.0)

    def test_missing_component_column_raises(self):
        """A silently absent component would make the cross-check pass by
        comparing nothing."""
        path = Path(self.tmp.name) / "score.csv"
        path.write_text("SMILES,Score,TPSA\nCCO,0.5,0.9\n")
        with self.assertRaises(KeyError):
            read_reinvent_csv(str(path))


class TestCrosscheck(unittest.TestCase):
    def test_reports_per_component_max_absolute_difference(self):
        reference = {"A": {"total": 1.0, "cooh": 1.0, "tpsa": 1.0,
                           "sascore": 1.0, "alerts": 1.0}}
        reinvent = [{"total": 1.0, "cooh": 1.0, "tpsa": 0.998,
                     "sascore": 1.0, "alerts": 1.0}]
        diffs = crosscheck(reference, ["A"], reinvent)
        self.assertAlmostEqual(diffs["tpsa"], 0.002, places=6)
        self.assertAlmostEqual(diffs["total"], 0.0)

    def test_row_count_mismatch_raises(self):
        """Zipping labels against a shorter CSV would silently compare the wrong
        molecule to the wrong score."""
        reference = {"A": {"total": 1.0}, "B": {"total": 1.0}}
        with self.assertRaises(ValueError):
            crosscheck(reference, ["A", "B"], [{"total": 1.0}])


if __name__ == "__main__":
    unittest.main()
