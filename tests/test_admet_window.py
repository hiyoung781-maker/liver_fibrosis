import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from admet_window import (CACO2_TIERS, DEFAULT_GATE_JSON, admet_verdict,
                          apply_window, read_admet, read_survivors,
                          toxicity_thresholds)
from lead_filter import CACO2_ENDPOINT, TOXICITY_ENDPOINTS

# .gitignore excludes results/*, so the §9.2 thresholds and v3's published lead
# table are runtime artifacts rather than committed fixtures: present in a
# working campaign directory, absent in a fresh clone. Every test that needs one
# skips instead of failing, the same convention test_rl_config.py uses for the
# REINVENT4 runtime. Regenerate with scripts/approved_drug_gate.py.
GATE_JSON = Path(DEFAULT_GATE_JSON)
V3_LEADS = Path("results/v3_final/TL-C/leads_unevaluable.csv")
needs_gate = unittest.skipUnless(GATE_JSON.exists(), f"{GATE_JSON} not present")
needs_v3 = unittest.skipUnless(V3_LEADS.exists(), f"{V3_LEADS} not present")


def pred(caco2=None, srmmp=0.0, nrahr=0.0, dili=0.0):
    return {CACO2_ENDPOINT: "" if caco2 is None else str(caco2),
            "SR-MMP": str(srmmp), "NR-AhR": str(nrahr), "DILI": str(dili)}


class TestPreRegisteredTiers(unittest.TestCase):
    """§5.5 fixed both tiers BEFORE v4's counts existed, because v3's Tier 1
    intersection was empty and loosening a threshold after seeing a zero is the
    post-hoc move §4.6 exists to prevent."""

    def test_tiers_are_the_bcs_bands_not_a_reference_compound(self):
        """PLN-1474 predicts 2.59e-06 cm/s - the BOTTOM of the BCS moderate
        band - so `caco2 > PLN-1474` only means "not catastrophically bad". Same
        defect §2.1 found in the old TPSA ceiling, which was anchored on two
        alphaV compounds and penalised one of them."""
        self.assertEqual(CACO2_TIERS, (("tier1_bcs_high", -5.0),
                                       ("tier2_bcs_moderate", -5.7)))

    def test_tier1_is_primary_and_comes_first(self):
        self.assertEqual(CACO2_TIERS[0][0], "tier1_bcs_high")
        self.assertGreater(CACO2_TIERS[0][1], CACO2_TIERS[1][1])


@needs_gate
class TestToxicityThresholds(unittest.TestCase):
    """§9.2's thresholds are READ from approved_drug_gate.py's output, not
    restated. The campaign has been bitten twice by a constant copied into a
    second place and frozen there - the TPSA window (§2.2) and PROPERTY_WINDOW
    (§5.4)."""

    def test_reads_the_adopted_sixtieth_percentile(self):
        got = toxicity_thresholds()
        self.assertEqual(set(got), set(TOXICITY_ENDPOINTS))
        data = json.loads(Path(DEFAULT_GATE_JSON).read_text())
        want = next(e for e in data["sweep"] if e["percentile"] == 60.0)
        for endpoint in TOXICITY_ENDPOINTS:
            self.assertAlmostEqual(got[endpoint], want["thresholds"][endpoint])

    def test_refuses_a_percentile_the_gate_did_not_adopt(self):
        with self.assertRaises(ValueError):
            toxicity_thresholds(DEFAULT_GATE_JSON, percentile=30.0)


@needs_gate
class TestAdmetVerdict(unittest.TestCase):
    def setUp(self):
        self.th = toxicity_thresholds()

    def test_a_missing_prediction_fails_rather_than_passes(self):
        """This gate decides what gets DOCKED. Letting an absent value through
        would send exactly the molecules the predictor could not handle to the
        expensive stage."""
        for absent in (None, {}):
            with self.subTest(prediction=absent):
                v = admet_verdict(absent, -5.0, self.th)
                self.assertFalse(v["pass"])
                self.assertFalse(v["caco2_pass"])
                self.assertFalse(v["toxicity_pass"])

    def test_a_prediction_without_a_caco2_value_fails_the_permeability_cut(self):
        v = admet_verdict(pred(caco2=None), -5.0, self.th)
        self.assertFalse(v["caco2_pass"])

    def test_the_caco2_bound_is_strict(self):
        """-5.0 is the BCS boundary itself, so a molecule exactly on it is not
        high-permeability."""
        self.assertFalse(admet_verdict(pred(caco2=-5.0), -5.0, self.th)["caco2_pass"])
        self.assertTrue(admet_verdict(pred(caco2=-4.999), -5.0, self.th)["caco2_pass"])

    def test_toxicity_needs_every_endpoint_below_not_their_mean(self):
        """§9.2 is a conjunction. A molecule can have an excellent mean and still
        fail - gen_01883 had the cleanest mean of all 231 (0.0724) and failed on
        SR-MMP 0.015 against a 0.011112 threshold, by 0.004."""
        clean = pred(caco2=-4.5, srmmp=0.001, nrahr=0.001, dili=0.1)
        self.assertTrue(admet_verdict(clean, -5.0, self.th)["toxicity_pass"])
        # One endpoint over, the other two far under: mean is still tiny.
        one_over = pred(caco2=-4.5, srmmp=0.015, nrahr=0.0, dili=0.0)
        self.assertFalse(admet_verdict(one_over, -5.0, self.th)["toxicity_pass"])

    @needs_v3
    def test_the_toxicity_comparison_matches_lead_filter_on_real_rows(self):
        """THE CROSS-CHECK THAT MATTERS. lead_filter.py already published a
        toxicity verdict for v3's 295 molecules; this stage must not disagree
        about what "clears toxicity" means, or the funnel would apply two
        different rules at two different points."""
        rows = list(csv.DictReader(V3_LEADS.open()))
        for row in rows:
            with self.subTest(label=row["label"]):
                mine = admet_verdict(row, float("-inf"), self.th)["toxicity_pass"]
                self.assertEqual(mine, row["toxicity_pass"].strip() == "True")


@needs_gate
class TestApplyWindow(unittest.TestCase):
    def setUp(self):
        self.th = toxicity_thresholds()

    def test_both_tiers_are_always_counted_even_when_tier1_is_empty(self):
        """The fallback stays honest only if its count is reported BESIDE the
        primary's rather than quietly replacing it."""
        entries = [("A", "a"), ("B", "b")]
        admet = {"A": pred(caco2=-5.3, srmmp=0.001, nrahr=0.001, dili=0.1),
                 "B": pred(caco2=-6.5, srmmp=0.001, nrahr=0.001, dili=0.1)}
        result = apply_window(entries, admet, self.th)
        self.assertEqual(set(result["tiers"]), {n for n, _ in CACO2_TIERS})
        self.assertEqual(result["tiers"]["tier1_bcs_high"]["survivors"], 0)
        self.assertEqual(result["tiers"]["tier2_bcs_moderate"]["survivors"], 1)

    def test_missing_predictions_are_counted_not_silently_dropped(self):
        entries = [("A", "a"), ("GONE", "b")]
        admet = {"A": pred(caco2=-4.5, srmmp=0.001, nrahr=0.001, dili=0.1)}
        result = apply_window(entries, admet, self.th)
        self.assertEqual(result["input"], 2)
        self.assertEqual(result["admet_missing"], 1)
        self.assertEqual(result["tiers"]["tier1_bcs_high"]["survivors"], 1)

    def test_toxicity_count_is_independent_of_the_caco2_tier(self):
        entries = [("A", "a")]
        admet = {"A": pred(caco2=-6.9, srmmp=0.001, nrahr=0.001, dili=0.1)}
        result = apply_window(entries, admet, self.th)
        self.assertEqual(result["toxicity_pass"], 1)
        for name, _floor in CACO2_TIERS:
            self.assertEqual(result["tiers"][name]["survivors"], 0)

    @needs_v3
    def test_reproduces_the_measured_v3_intersection(self):
        """REGRESSION LOCK ON THE FINDING. v3's affinity-passing set contains 37
        BCS-high molecules with affinities of -9.107 to -7.85 - stronger than
        every lead the campaign chose - and NONE of them clears §9.2. The empty
        Tier 1 intersection is why §5.5 moves these cuts before docking: a
        reordering cannot find an intersection that is empty, and neither can
        Pareto selection."""
        rows = [r for r in csv.DictReader(V3_LEADS.open())
                if r["affinity_pass"].strip() == "True"]
        # lead_filter writes the column as `caco2`; admet_predict as Caco2_Wang.
        admet = {r["label"]: {**r, CACO2_ENDPOINT: r["caco2"]} for r in rows}
        result = apply_window([(r["label"], r["label"]) for r in rows],
                              admet, self.th)
        self.assertEqual(result["tiers"]["tier1_bcs_high"]["caco2_pass"], 37)
        self.assertEqual(result["tiers"]["tier1_bcs_high"]["survivors"], 0)
        self.assertEqual(result["toxicity_pass"], 2)


class TestReaders(unittest.TestCase):
    def test_read_admet_rejects_a_file_without_the_endpoints(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            path.write_text("smiles,something\nCC,1\n")
            with self.assertRaises(KeyError):
                read_admet(str(path))

    def test_read_survivors_skips_comments_and_keeps_labels(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.smi"
            path.write_text("# header\n# more\nCC(=O)O\tgen_00001\nCCO\tgen_00002\n")
            self.assertEqual(read_survivors(str(path)),
                             [("CC(=O)O", "gen_00001"), ("CCO", "gen_00002")])


if __name__ == "__main__":
    unittest.main()
