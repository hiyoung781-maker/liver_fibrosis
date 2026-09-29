import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lead_filter import (CACO2_ENDPOINT, CACO2_MARGIN, REFERENCE,
                         TOXICITY_ENDPOINTS, lead_rows, summarize,
                         toxicity_composite)

FUNNEL_FIELDS = ["label", "n_poses", "n_passing_poses", "geometry_pass",
                 "best_affinity", "best_passing_affinity", "affinity_pass",
                 "plip_metal_any_passing", "plip_hbond_any_passing",
                 "min_ca_dist", "min_donor_dist"]
ADMET_FIELDS = ["label", CACO2_ENDPOINT] + list(TOXICITY_ENDPOINTS)


def write(path, fields, rows):
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def funnel(label, affinity=-7.5, geometry=True):
    return {"label": label, "n_poses": 4, "n_passing_poses": 2,
            "geometry_pass": geometry, "best_affinity": affinity,
            "best_passing_affinity": affinity if geometry else "",
            "affinity_pass": geometry and affinity < -7.290,
            "plip_metal_any_passing": True, "plip_hbond_any_passing": True,
            "min_ca_dist": 2.6, "min_donor_dist": 2.9}


def admet(label, caco2=-5.0, mmp=0.02, ahr=0.01, dili=0.40):
    return {"label": label, CACO2_ENDPOINT: caco2,
            "SR-MMP": mmp, "NR-AhR": ahr, "DILI": dili}


class TestToxicityComposite(unittest.TestCase):
    """spec 9.2 replaces v1's DILI 2.0 / hERG 1.0 / CYP 0.5 weighting, which
    had no basis and failed its own sanity check -- CWHM-12, a parenteral
    pan-alphaV tool compound, came out LESS toxic than PLN-1474. The v2
    definition is the equal mean of the three endpoints ADMET-AI predicts
    best: SR-MMP (AUROC 0.925), NR-AhR (0.904), DILI (0.881)."""

    def test_it_is_the_plain_mean_of_three_endpoints(self):
        row = {"SR-MMP": "0.0208", "NR-AhR": "0.0093", "DILI": "0.455"}
        self.assertAlmostEqual(toxicity_composite(row),
                               (0.0208 + 0.0093 + 0.455) / 3, places=6)

    def test_it_reproduces_the_specs_predicted_reference(self):
        # spec 9.2: PLN-1474's documented values give about 0.162.
        row = {"SR-MMP": "0.0208", "NR-AhR": "0.0093", "DILI": "0.455"}
        self.assertAlmostEqual(toxicity_composite(row), 0.162, places=3)

    def test_exactly_three_endpoints_are_used(self):
        self.assertEqual(set(TOXICITY_ENDPOINTS), {"SR-MMP", "NR-AhR", "DILI"})

    def test_a_missing_endpoint_gives_no_score_rather_than_a_partial_one(self):
        # Averaging two of three silently changes the definition, and the
        # result would still look like a number the filter could use.
        self.assertIsNone(toxicity_composite({"SR-MMP": "0.02", "DILI": "0.4"}))


class TestFourStageFilter(unittest.TestCase):
    """spec 9.1 has FOUR filters in this order: geometry, Caco-2, affinity,
    toxicity -- all relative to PLN-1474. The order does not change the final
    set, since every stage is a conjunction, but it does change the counts the
    report shows, and the pre-registered order is the one to report."""

    def _run(self, tmp, funnel_rows, admet_rows, cutoff=-7.290):
        f = Path(tmp) / "funnel.csv"
        a = Path(tmp) / "admet.csv"
        write(f, FUNNEL_FIELDS, funnel_rows)
        write(a, ADMET_FIELDS, admet_rows)
        return {r["label"]: r for r in lead_rows(str(f), str(a), cutoff)}

    def _reference(self, caco2=-5.586, dili=0.455):
        return admet(REFERENCE, caco2=caco2, dili=dili)

    def test_caco2_needs_a_half_log_margin_not_merely_better(self):
        with TemporaryDirectory() as tmp:
            rows = self._run(
                tmp,
                [funnel("near"), funnel("clear")],
                [self._reference(),
                 admet("near", caco2=-5.586 + 0.3),
                 admet("clear", caco2=-5.586 + 0.6)])
            self.assertFalse(rows["near"]["caco2_pass"])
            self.assertTrue(rows["clear"]["caco2_pass"])

    def test_the_margin_is_the_pre_registered_half_log(self):
        self.assertEqual(CACO2_MARGIN, 0.5)

    def test_toxicity_must_be_strictly_below_the_reference(self):
        with TemporaryDirectory() as tmp:
            rows = self._run(
                tmp,
                [funnel("tie"), funnel("lower")],
                [self._reference(dili=0.455),
                 admet("tie", dili=0.455),
                 admet("lower", dili=0.400)])
            self.assertFalse(rows["tie"]["toxicity_pass"])
            self.assertTrue(rows["lower"]["toxicity_pass"])

    def test_a_ligand_absent_from_the_admet_run_fails_visibly(self):
        # It must not silently pass, and it must not vanish from the
        # denominator either.
        with TemporaryDirectory() as tmp:
            rows = self._run(tmp, [funnel("ghost")], [self._reference()])
            self.assertIn("ghost", rows)
            self.assertFalse(rows["ghost"]["caco2_pass"])
            self.assertFalse(rows["ghost"]["toxicity_pass"])
            self.assertEqual(rows["ghost"]["admet_status"], "missing")

    def test_a_missing_reference_is_an_error_not_a_default(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._run(tmp, [funnel("a")], [admet("a")])

    def test_all_four_flags_are_present_on_every_row(self):
        with TemporaryDirectory() as tmp:
            rows = self._run(tmp, [funnel("a")],
                             [self._reference(), admet("a")])
            for key in ("geometry_pass", "caco2_pass", "affinity_pass",
                        "toxicity_pass", "all_four"):
                self.assertIn(key, rows["a"])


class TestSummary(unittest.TestCase):
    def _rows(self):
        return [
            {"label": "a", "geometry_pass": True, "caco2_pass": True,
             "affinity_pass": True, "toxicity_pass": True, "all_four": True},
            {"label": "b", "geometry_pass": True, "caco2_pass": True,
             "affinity_pass": True, "toxicity_pass": False, "all_four": False},
            {"label": "c", "geometry_pass": True, "caco2_pass": False,
             "affinity_pass": True, "toxicity_pass": True, "all_four": False},
        ]

    def test_the_funnel_reports_the_pre_registered_order(self):
        s = summarize(self._rows())
        self.assertEqual(s["stage_1_geometry"], 3)
        self.assertEqual(s["stage_2_caco2"], 2)
        self.assertEqual(s["stage_3_affinity"], 2)
        self.assertEqual(s["stage_4_toxicity"], 1)
        self.assertEqual(s["leads"], ["a"])

    def test_each_stage_is_cumulative_so_the_counts_never_rise(self):
        s = summarize(self._rows())
        counts = [s["stage_1_geometry"], s["stage_2_caco2"],
                  s["stage_3_affinity"], s["stage_4_toxicity"]]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_an_empty_input_does_not_divide_by_zero(self):
        self.assertEqual(summarize([])["stage_1_geometry"], 0)


if __name__ == "__main__":
    unittest.main()
