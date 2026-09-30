import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lead_filter import (CACO2_ENDPOINT, REFERENCE, STAGES, TOXICITY_ENDPOINTS,
                         lead_rows, summarize)

FUNNEL_FIELDS = ["label", "n_poses", "n_passing_poses", "geometry_pass",
                 "best_affinity", "best_passing_affinity", "affinity_pass",
                 "plip_metal_any_passing", "plip_hbond_any_passing",
                 "min_ca_dist", "min_donor_dist"]
ADMET_FIELDS = ["label", CACO2_ENDPOINT] + list(TOXICITY_ENDPOINTS)


def write(path, fields, rows):
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def funnel(label, affinity=-7.5):
    return {"label": label, "n_poses": 4, "n_passing_poses": 2,
            "geometry_pass": True, "best_affinity": affinity,
            "best_passing_affinity": affinity, "affinity_pass": True,
            "plip_metal_any_passing": True, "plip_hbond_any_passing": True,
            "min_ca_dist": 2.6, "min_donor_dist": 2.9}


def admet(label, caco2=-5.0, mmp=0.010, ahr=0.005, dili=0.20):
    return {"label": label, CACO2_ENDPOINT: caco2,
            "SR-MMP": mmp, "NR-AhR": ahr, "DILI": dili}


REF_ADMET = admet(REFERENCE, caco2=-5.586, mmp=0.0208, ahr=0.0093, dili=0.455)


class TestStageOrder(unittest.TestCase):
    """Every stage is a conjunction, so the order does not change the final
    set -- it changes the counts the report shows. The order runs strongest
    evidence first: structure, then a same-engine relative affinity, then
    selectivity, then model predictions, with Caco-2 last because it is the
    one this project itself calls partly circular (predicted Caco-2 correlates
    with TPSA at r = -0.571 and TPSA is an axis the RL objective optimises)."""

    def test_selectivity_sits_between_affinity_and_toxicity(self):
        self.assertEqual(STAGES, ("geometry", "affinity", "selectivity",
                                  "toxicity", "caco2"))


class TestPerEndpointToxicity(unittest.TestCase):
    """An equal mean hides a single bad endpoint behind two good ones.
    Measured: gen_03050 passes the mean at 0.1442 while its SR-MMP is 0.3323,
    sixteen times PLN-1474's 0.0208. Requiring EVERY endpoint to beat the
    reference removes the dynamic-range domination section 9.2 admitted it
    could not fix."""

    def _rows(self, tmp, admet_rows, selectivity=None):
        f, a = Path(tmp) / "f.csv", Path(tmp) / "a.csv"
        write(f, FUNNEL_FIELDS, [funnel(r["label"]) for r in admet_rows
                                if r["label"] != REFERENCE])
        write(a, ADMET_FIELDS, admet_rows)
        return {r["label"]: r for r in lead_rows(str(f), str(a), -6.807,
                                                 selectivity=selectivity)}

    def test_one_bad_endpoint_fails_even_with_a_good_mean(self):
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp, [REF_ADMET,
                                    admet("hidden", mmp=0.3323, ahr=0.0033,
                                          dili=0.097)])
            row = rows["hidden"]
            self.assertLess(row["toxicity"], 0.1618)      # the mean passes
            self.assertFalse(row["toxicity_pass"])        # the rule does not
            self.assertEqual(row["toxicity_worse_than_reference"], ["SR-MMP"])

    def test_all_three_below_the_reference_passes(self):
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp, [REF_ADMET, admet("clean")])
            self.assertTrue(rows["clean"]["toxicity_pass"])
            self.assertEqual(rows["clean"]["toxicity_worse_than_reference"], [])

    def test_a_tie_on_any_endpoint_fails(self):
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp, [REF_ADMET, admet("tie", mmp=0.0208)])
            self.assertFalse(rows["tie"]["toxicity_pass"])

    def test_the_mean_is_still_reported(self):
        # It is the pre-registered number and stays in the record.
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp, [REF_ADMET, admet("clean")])
            self.assertAlmostEqual(rows["clean"]["toxicity"],
                                   (0.010 + 0.005 + 0.20) / 3, places=6)
            self.assertTrue(rows["clean"]["toxicity_mean_pass"])


class TestSelectivityStage(unittest.TestCase):
    """Selectivity is a HARD FILTER at stage 3, applied unconditionally.

    Spec 10.4's clause making criterion 6 unevaluable when the calibration
    gate fails is amended away (2026-09-30, change ledger). The calibration
    verdict is still computed and reported beside the filter, because it is
    what says whether the protocol has demonstrated any discriminating power
    -- but it no longer gates the stage. A molecule with no selectivity value
    fails: the stage cannot pass what it did not measure."""

    def _rows(self, tmp, selectivity):
        f, a = Path(tmp) / "f.csv", Path(tmp) / "a.csv"
        write(f, FUNNEL_FIELDS, [funnel("a"), funnel("b")])
        write(a, ADMET_FIELDS, [REF_ADMET, admet("a"), admet("b")])
        return {r["label"]: r for r in lead_rows(str(f), str(a), -6.807,
                                                 selectivity=selectivity)}

    def test_it_filters_on_being_above_the_reference(self):
        sel = {"reference": 0.10, "values": {"a": 0.30, "b": 0.05}}
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp, sel)
            self.assertTrue(rows["a"]["selectivity_pass"])
            self.assertFalse(rows["b"]["selectivity_pass"])

    def test_a_tie_with_the_reference_does_not_pass(self):
        sel = {"reference": 0.10, "values": {"a": 0.10, "b": 0.11}}
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp, sel)
            self.assertFalse(rows["a"]["selectivity_pass"])
            self.assertTrue(rows["b"]["selectivity_pass"])

    def test_it_applies_whatever_the_calibration_gate_said(self):
        # The amendment: the verdict is recorded, not obeyed.
        sel = {"calibration_passed": False, "reference": 0.10,
               "values": {"a": 0.30, "b": 0.05}}
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp, sel)
            self.assertTrue(rows["a"]["selectivity_pass"])
            self.assertFalse(rows["b"]["selectivity_pass"])
            self.assertFalse(rows["a"]["selectivity_calibrated"])

    def test_a_molecule_with_no_value_fails(self):
        sel = {"reference": 0.10, "values": {"a": 0.30}}
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp, sel)
            self.assertFalse(rows["b"]["selectivity_pass"])
            self.assertEqual(rows["b"]["selectivity_status"], "missing")

    def test_no_selectivity_data_at_all_fails_every_molecule(self):
        # The stage is a filter now; with nothing measured, nothing passes.
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp, None)
            self.assertFalse(rows["a"]["selectivity_pass"])
            self.assertEqual(rows["a"]["selectivity_status"], "missing")


class TestSummaryOrder(unittest.TestCase):
    def test_the_funnel_narrows_in_stage_order(self):
        rows = [{"label": "a", "geometry_pass": True, "affinity_pass": True,
                 "selectivity_pass": True, "toxicity_pass": True,
                 "caco2_pass": True, "all_stages": True},
                {"label": "b", "geometry_pass": True, "affinity_pass": True,
                 "selectivity_pass": True, "toxicity_pass": True,
                 "caco2_pass": False, "all_stages": False},
                {"label": "c", "geometry_pass": True, "affinity_pass": True,
                 "selectivity_pass": False, "toxicity_pass": True,
                 "caco2_pass": True, "all_stages": False}]
        s = summarize(rows)
        counts = [s[f"stage_{i}_{name}"] for i, name in enumerate(STAGES, 1)]
        self.assertEqual(counts, [3, 3, 2, 2, 1])
        self.assertEqual(s["leads"], ["a"])


if __name__ == "__main__":
    unittest.main()
