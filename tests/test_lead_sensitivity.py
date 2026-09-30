import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from funnel import PLN1474_CUTOFF
from lead_filter import CACO2_MARGIN, stages
from lead_sensitivity import CUTOFFS, sweep

FUNNEL_FIELDS = ["label", "geometry_pass", "best_passing_affinity",
                 "plip_gate_pass", "plip_metal_any_passing",
                 "plip_hbond_any_passing", "n_poses"]
ADMET_FIELDS = ["label", "Caco2_Wang", "SR-MMP", "NR-AhR", "DILI"]
REFERENCE = "PANEL_PLN-1474"
REF = {"label": REFERENCE, "Caco2_Wang": -5.586,
       "SR-MMP": 0.0208, "NR-AhR": 0.0093, "DILI": 0.455}


def write(path, fields, rows):
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build(tmp, mols):
    """mols: (label, affinity, caco2)."""
    f, a = Path(tmp) / "f.csv", Path(tmp) / "a.csv"
    write(f, FUNNEL_FIELDS,
          [{"label": l, "geometry_pass": "True",
            "best_passing_affinity": f"{aff}", "plip_gate_pass": "True",
            "plip_metal_any_passing": "True",
            "plip_hbond_any_passing": "True", "n_poses": "5"}
           for l, aff, _ in mols])
    write(a, ADMET_FIELDS,
          [REF] + [{"label": l, "Caco2_Wang": c, "SR-MMP": 0.005,
                    "NR-AhR": 0.002, "DILI": 0.10} for l, _, c in mols])
    selectivity = {"values": {**{l: 0.5 for l, _, _ in mols}, REFERENCE: 0.0},
                   "reference": 0.0, "calibration_passed": True}
    return str(f), str(a), selectivity


class TestTheCutoffsAreSection77s(unittest.TestCase):
    """Not chosen here. They are the five seeds' measured values plus the
    median, fixed before the sweep runs, so a reader sees the whole curve
    rather than the point that gave someone the answer they wanted."""

    def test_the_preregistered_value_is_in_the_sweep(self):
        self.assertIn(PLN1474_CUTOFF, [c for c, _ in CUTOFFS])

    def test_the_range_is_the_measured_one(self):
        values = [c for c, _ in CUTOFFS]
        self.assertAlmostEqual(min(values), -7.319)
        self.assertAlmostEqual(max(values), -6.802)

    def test_they_are_ordered_strict_to_loose(self):
        values = [c for c, _ in CUTOFFS]
        self.assertEqual(values, sorted(values))


class TestSweep(unittest.TestCase):

    def test_the_lead_set_never_shrinks_as_the_cutoff_loosens(self):
        with TemporaryDirectory() as tmp:
            f, a, sel = build(tmp, [("strong", -7.9, -5.0),
                                    ("weak", -6.9, -5.0)])
            counts = [len(r["leads"]) for r in sweep(f, a, sel, "better",
                                                     REFERENCE)]
        self.assertEqual(counts, sorted(counts))

    def test_a_molecule_below_every_cutoff_is_a_lead_throughout(self):
        with TemporaryDirectory() as tmp:
            f, a, sel = build(tmp, [("strong", -8.5, -5.0)])
            results = sweep(f, a, sel, "better", REFERENCE)
        self.assertTrue(all(r["leads"] == ["strong"] for r in results))

    def test_a_molecule_only_at_the_worst_seed_is_visible_as_such(self):
        # -6.85 clears only the loosest cutoff. The report has to be able to
        # say "this one is not a pre-registered lead".
        with TemporaryDirectory() as tmp:
            f, a, sel = build(tmp, [("marginal", -6.85, -5.0)])
            results = sweep(f, a, sel, "better", REFERENCE)
        self.assertEqual([len(r["leads"]) for r in results], [0, 0, 0, 1])


class TestCaco2Modes(unittest.TestCase):
    """Caco-2 left the filter on 2026-09-30 and is back as an option, because
    the campaign asked for it in selection. 'better' drops the 0.5-log margin
    that no molecule cleared; 'margin' is section 9.1 as written."""

    def _leads(self, mode, caco2):
        with TemporaryDirectory() as tmp:
            f, a, sel = build(tmp, [("x", -8.0, caco2)])
            return sweep(f, a, sel, mode, REFERENCE)[1]["leads"]

    def test_off_ignores_caco2_entirely(self):
        self.assertEqual(self._leads("off", -9.9), ["x"])

    def test_better_asks_only_that_it_beat_the_reference(self):
        self.assertEqual(self._leads("better", -5.586 + 0.01), ["x"])
        self.assertEqual(self._leads("better", -5.586 - 0.01), [])

    def test_margin_is_the_preregistered_half_log(self):
        self.assertEqual(CACO2_MARGIN, 0.5)
        self.assertEqual(self._leads("margin", -5.586 + 0.3), [])
        self.assertEqual(self._leads("margin", -5.586 + 0.6), ["x"])

    def test_caco2_becomes_a_sixth_stage_only_when_enabled(self):
        self.assertNotIn("caco2", stages("off"))
        self.assertIn("caco2", stages("better"))
        self.assertIn("caco2", stages("margin"))


if __name__ == "__main__":
    unittest.main()
