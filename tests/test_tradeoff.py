import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from funnel import PLN1474_CUTOFF
from lead_cards import DELTA_SD, SEED_RANGE, SEED_SD
from tradeoff import (BORDERLINE_BAND, REFERENCE, TOXICITY_ENDPOINTS,
                      categorise, pareto_front, rows)

FUNNEL_FIELDS = ["label", "geometry_pass", "best_passing_affinity",
                 "plip_gate_pass", "plip_hbond_any_passing"]
ADMET_FIELDS = ["label", "Caco2_Wang"] + list(TOXICITY_ENDPOINTS)

# PLN-1474's own values, which every margin on a card is measured against.
REF_TOX = {"SR-MMP": 0.0208, "NR-AhR": 0.0093, "DILI": 0.455}
SEL_REF = -0.222


def write(path, fields, rows_):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_)


def funnel(label, affinity, gate="True", hbond="True"):
    return {"label": label, "geometry_pass": "True",
            "best_passing_affinity": f"{affinity}", "plip_gate_pass": gate,
            "plip_hbond_any_passing": hbond}


def admet(label, mmp=0.005, ahr=0.002, dili=0.10, caco2=-5.0):
    return {"label": label, "Caco2_Wang": caco2,
            "SR-MMP": mmp, "NR-AhR": ahr, "DILI": dili}


def build(tmp, entries, selectivity_values):
    """entries: (label, affinity, admet kwargs, gate, hbond)."""
    f, a = Path(tmp) / "f.csv", Path(tmp) / "a.csv"
    write(f, FUNNEL_FIELDS,
          [funnel(l, aff, gate, hb) for l, aff, _, gate, hb in entries])
    write(a, ADMET_FIELDS,
          [{"label": REFERENCE, "Caco2_Wang": -5.586, **REF_TOX}]
          + [admet(l, **kw) for l, _, kw, _, _ in entries])
    selectivity = {"values": {**selectivity_values, REFERENCE: SEL_REF},
                   "reference": SEL_REF}
    return rows(str(f), str(a), selectivity)


class TestThePopulationIsTheStructuralStages(unittest.TestCase):
    """The trade-off is between selectivity and toxicity. Whether the pose is
    a pose is not a trade-off, so geometry, affinity and the section 8.2 gate
    filter the population rather than appearing as axes in it."""

    def test_a_molecule_above_the_affinity_cutoff_is_excluded(self):
        with TemporaryDirectory() as tmp:
            got = build(tmp, [("weak", PLN1474_CUTOFF + 0.1, {}, "True", "True"),
                              ("ok", -8.0, {}, "True", "True")],
                        {"weak": 1.0, "ok": 1.0})
            self.assertEqual([e["label"] for e in got], ["ok"])

    def test_a_molecule_failing_the_interaction_gate_is_excluded(self):
        with TemporaryDirectory() as tmp:
            got = build(tmp, [("nogate", -8.0, {}, "False", "True")],
                        {"nogate": 1.0})
            self.assertEqual(got, [])

    def test_the_cutoff_is_the_imported_one(self):
        self.assertAlmostEqual(PLN1474_CUTOFF, -7.290, places=3)


class TestCategories(unittest.TestCase):
    """Each label states what may be claimed. Only LEAD passes the
    pre-registered filter; the rest are cases to discuss, and a reader who
    takes one for a lead has been misled by the document."""

    def _one(self, tmp, affinity, selectivity, **tox):
        got = build(tmp, [("x", affinity, tox, "True", "True")],
                    {"x": selectivity})
        return got[0]

    def test_lead_passes_everything(self):
        with TemporaryDirectory() as tmp:
            e = self._one(tmp, -8.0, SEL_REF + 1.0)
            self.assertTrue(e["all_stages"])
            self.assertEqual(e["category"], "LEAD")

    def test_selectivity_case_has_a_real_margin_and_one_bad_endpoint(self):
        with TemporaryDirectory() as tmp:
            e = self._one(tmp, -8.0, SEL_REF + 1.0, dili=0.60)
            self.assertEqual(e["tox_worse"], ["DILI"])
            self.assertEqual(e["category"], "SELECTIVITY")
            self.assertFalse(e["all_stages"])

    def test_a_selectivity_margin_inside_the_noise_is_not_a_selectivity_case(self):
        # The whole point: 0.005 over the reference is not over the reference.
        with TemporaryDirectory() as tmp:
            e = self._one(tmp, -8.0, SEL_REF + 0.005, dili=0.60)
            self.assertLess(e["selectivity_margin"], DELTA_SD)
            self.assertNotEqual(e["category"], "SELECTIVITY")

    def test_safety_case_is_clean_on_toxicity_and_fails_selectivity(self):
        with TemporaryDirectory() as tmp:
            e = self._one(tmp, -8.0, SEL_REF - 0.005)
            self.assertEqual(e["tox_worse"], [])
            self.assertFalse(e["selectivity_pass"])
            self.assertEqual(e["category"], "SAFETY")

    def test_a_lead_whose_selectivity_margin_is_noise_is_flagged(self):
        # gen_01883 is exactly this: it passes the stage by 0.005 against a
        # noise band of 0.221. It is a lead, and its selectivity says nothing.
        with TemporaryDirectory() as tmp:
            e = self._one(tmp, -8.0, SEL_REF + 0.005)
            self.assertEqual(e["category"], "LEAD")
            self.assertTrue(e["selectivity_pass"])
            self.assertTrue(e["selectivity_within_noise"])

    def test_a_lead_with_a_real_selectivity_margin_is_not_flagged(self):
        with TemporaryDirectory() as tmp:
            e = self._one(tmp, -8.0, SEL_REF + 1.0)
            self.assertEqual(e["category"], "LEAD")
            self.assertFalse(e["selectivity_within_noise"])

    def test_two_bad_endpoints_is_not_a_trade_off_case(self):
        with TemporaryDirectory() as tmp:
            e = self._one(tmp, -8.0, SEL_REF + 1.0, dili=0.60, mmp=0.9)
            self.assertEqual(len(e["tox_worse"]), 2)
            self.assertEqual(e["category"], "other")

    def test_borderline_affinity_outranks_lead(self):
        # Clearing the cutoff by 0.01 against a seed range of 0.517 has not
        # been shown to clear it, whatever the other axes say.
        with TemporaryDirectory() as tmp:
            e = self._one(tmp, PLN1474_CUTOFF - 0.01, SEL_REF + 1.0)
            self.assertTrue(e["borderline_affinity"])
            self.assertTrue(e["all_stages"])
            self.assertEqual(e["category"], "BORDERLINE")

    def test_a_clear_margin_is_not_borderline(self):
        with TemporaryDirectory() as tmp:
            e = self._one(tmp, PLN1474_CUTOFF - 0.30, SEL_REF + 1.0)
            self.assertFalse(e["borderline_affinity"])

    def test_the_band_is_the_sd_not_the_range(self):
        """Section 7.7 reports SD 0.156 and range 0.517. The range is the max
        minus the min of five draws -- an extreme statistic that grows with
        the number of seeds, not an uncertainty interval. Used as a band it
        called 107 of 179 TL-C molecules borderline, margins of 0.5
        included."""
        self.assertAlmostEqual(BORDERLINE_BAND, SEED_SD)
        self.assertLess(BORDERLINE_BAND, SEED_RANGE)

    def test_a_margin_of_a_third_of_a_kcal_is_a_real_margin(self):
        with TemporaryDirectory() as tmp:
            e = self._one(tmp, PLN1474_CUTOFF - 0.35, SEL_REF + 1.0, dili=0.60)
            self.assertFalse(e["borderline_affinity"])
            self.assertEqual(e["category"], "SELECTIVITY")


class TestParetoFront(unittest.TestCase):
    """Non-dominated on selectivity up and the section 9.2 composite down."""

    def _entry(self, label, selectivity, toxicity):
        return {"label": label, "selectivity": selectivity,
                "toxicity": toxicity}

    def test_a_dominated_molecule_is_dropped(self):
        entries = [self._entry("best", 1.0, 0.05),
                   self._entry("worse", 0.5, 0.10)]
        self.assertEqual([e["label"] for e in pareto_front(entries)], ["best"])

    def test_a_trade_off_keeps_both(self):
        entries = [self._entry("selective", 1.0, 0.20),
                   self._entry("clean", 0.2, 0.03)]
        self.assertEqual({e["label"] for e in pareto_front(entries)},
                         {"selective", "clean"})

    def test_ties_are_not_self_dominated(self):
        entries = [self._entry("a", 1.0, 0.05), self._entry("b", 1.0, 0.05)]
        self.assertEqual(len(pareto_front(entries)), 2)

    def test_a_molecule_without_a_selectivity_value_is_not_ranked(self):
        entries = [self._entry("a", None, 0.05), self._entry("b", 1.0, 0.10)]
        self.assertEqual([e["label"] for e in pareto_front(entries)], ["b"])

    def test_the_front_is_sorted_most_selective_first(self):
        entries = [self._entry("mid", 0.5, 0.10), self._entry("top", 1.0, 0.20),
                   self._entry("clean", 0.1, 0.01)]
        self.assertEqual([e["label"] for e in pareto_front(entries)],
                         ["top", "mid", "clean"])


if __name__ == "__main__":
    unittest.main()
