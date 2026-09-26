import csv
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from novelty import load_smi
from objective import (
    GATE_FLOOR,
    double_sigmoid,
    geometric_mean,
    reverse_sigmoid,
    right_step,
    score_smiles,
)


def prior_smiles():
    with open("logs/cmp.vigHoB/samples_prior.csv") as handle:
        return [row["SMILES"] for row in csv.DictReader(handle)]


class TestTransformsMatchReinvent(unittest.TestCase):
    """Validate against real REINVENT output in logs/cmp.vigHoB/new3.csv.

    That file was scored with SlogP reverse_sigmoid(2, 5, k=0.4) and TPSA
    double_sigmoid(40, 120, 120, 20, 20) - deliberately different windows from this
    module's own, so agreement validates the ported math rather than restating the
    config. REINVENT overwrites every component with 0 for a molecule its filter
    rejects, so those rows cannot validate a transform; the "alerts" column
    identifies them directly (19 rows in this file).
    """

    def setUp(self):
        with open("logs/cmp.vigHoB/new3.csv") as handle:
            self.rows = list(csv.DictReader(handle))

    def _cols(self, raw_col, score_col):
        raw, got = [], []
        for row in self.rows:
            if float(row["alerts"]) == 0.0:  # REINVENT zeroed every component
                continue
            raw.append(float(row[raw_col]))
            got.append(float(row[score_col]))
        return np.array(raw), np.array(got)

    def test_reverse_sigmoid_matches(self):
        raw, got = self._cols("SlogP (raw)", "SlogP")
        self.assertLess(np.abs(reverse_sigmoid(raw, 2.0, 5.0, 0.4) - got).max(), 1e-6)

    def test_double_sigmoid_matches(self):
        raw, got = self._cols("TPSA (raw)", "TPSA")
        computed = double_sigmoid(raw, 40.0, 120.0, 120.0, 20.0, 20.0)
        self.assertLess(np.abs(computed - got).max(), 1e-6)

    def test_geometric_mean_and_penalty_reproduce_total(self):
        rows = [r for r in self.rows if float(r["alerts"]) != 0.0]
        col = lambda name: np.array([float(r[name]) for r in rows])
        scored = [
            (col("sim A1AFA"), 1.0),
            (col("sim CHEMBL4649232"), 1.0),
            (col("sim CHEMBL5532604"), 1.0),
            (col("TPSA"), 1.0),
            (col("SlogP"), 1.0),
            (col("QED"), 1.0),
        ]
        total = geometric_mean(scored) * col("carboxylic acid") * col("alerts")
        self.assertLess(np.abs(total - col("Score")).max(), 1e-6)


class TestRightStepGate(unittest.TestCase):
    def test_right_step_is_binary_at_one(self):
        got = right_step(np.array([0.0, 1.0, 2.0]), 1.0)
        np.testing.assert_array_equal(got, np.array([0.0, 1.0, 1.0]))

    def test_gate_floor_value(self):
        """Absence of carboxylate caps the total at 1e-8 ** (1.0 / 2.5)."""
        self.assertAlmostEqual(GATE_FLOOR, 6.3096e-04, places=7)

    def test_no_carboxylate_prior_sample_reaches_the_floor_at_most(self):
        result = score_smiles(prior_smiles())
        without = result["total"][result["cooh"] == 0.0]
        self.assertEqual(len(without), 219)
        self.assertLessEqual(without.max(), GATE_FLOOR + 1e-9)


class TestCarboxylateSmartsSpecificity(unittest.TestCase):
    """Review Focus 3: the gate must not open for a look-alike group."""

    def test_matches_acid_and_anion_only(self):
        cases = {
            "CC(=O)O": 1.0,        # carboxylic acid
            "CC(=O)[O-]": 1.0,     # carboxylate anion
            "CC(=O)OC": 0.0,       # methyl ester
            "CC(=O)N": 0.0,        # amide
            "CC=O": 0.0,           # aldehyde
            "c1nnn[nH]1": 0.0,     # tetrazole bioisostere
        }
        result = score_smiles(list(cases))
        # Zip through result["smiles"], not `cases`: score_smiles drops unparseable
        # input, so pairing by position would silently shift every case after a
        # parse failure onto the wrong score.
        got = dict(zip(result["smiles"], result["cooh"]))
        for smiles, expected in cases.items():
            with self.subTest(smiles=smiles):
                self.assertEqual(got[smiles], expected)


class TestSpecNumbers(unittest.TestCase):
    """Pin every distribution the spec claims, so a drift in RDKit is loud."""

    def test_acid_subset_distribution(self):
        result = score_smiles(prior_smiles())
        keep = (result["cooh"] > 0) & (result["alerts"] > 0)
        acid = np.sort(result["total"][keep])
        self.assertEqual(len(acid), 254)
        self.assertAlmostEqual(float(np.median(acid)), 0.178, delta=0.01)
        self.assertAlmostEqual(float((acid > 0.95).mean()), 0.33, delta=0.02)
        self.assertAlmostEqual(float((acid > 0.5).mean()), 0.43, delta=0.02)

    def test_benchmark_panel_scores(self):
        rows = load_smi("data/benchmark_panel.smi")
        result = score_smiles([s for s, _ in rows])
        # Zip through result["smiles"], not `rows`: score_smiles drops unparseable
        # input, so pairing labels with scores by position would silently shift
        # every label after a parse failure onto the wrong molecule's score.
        label_of = dict((smiles, label) for smiles, label in rows)
        got = {label_of[s]: total
               for s, total in zip(result["smiles"], result["total"])}
        self.assertEqual(len(got), len(rows))
        expected = {
            "A1AFA": 1.000,
            "PLN-1474": 0.998,
            "bexotegrast": 0.878,
            "CHEMBL4649232": 0.060,
            "GLPG0187": 0.001,
            "CWHM-12": 0.001,
        }
        for label, value in expected.items():
            with self.subTest(label=label):
                self.assertAlmostEqual(float(got[label]), value, delta=0.005)

    def test_sascore_window_does_not_punish_any_reference_molecule(self):
        """The guard rail must be indistinguishable from inert on the references.

        reverse_sigmoid is a logistic curve: it approaches 1.0 asymptotically and
        never reaches it. The worst reference molecule (SA 5.095, in
        actives_extended) scores 0.99998, which under the geometric mean's 0.2
        exponent depresses a total by 2.2e-06 - inert for every practical purpose.
        The rejected (3, 6) window gave that same molecule 0.093, so this bound is
        wide enough to be true and tight enough to catch a reversion.
        """
        for path in ("data/actives_core.smi", "data/actives_extended.smi",
                     "data/benchmark_panel.smi", "data/similarity_refs.smi"):
            with self.subTest(path=path):
                result = score_smiles([s for s, _ in load_smi(path)])
                self.assertGreater(float(result["sascore"].min()), 0.9999)


if __name__ == "__main__":
    unittest.main()
