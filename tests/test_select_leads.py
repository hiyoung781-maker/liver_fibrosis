import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from select_leads import (
    CONTEXT_ENDPOINTS,
    PERMEABILITY_ENDPOINT,
    PERMEABILITY_FLOOR,
    EXCLUDED_ENDPOINTS,
    PANEL_PREFIX,
    REFERENCE,
    TOXICITY_WEIGHTS,
    best_pose_per_ligand,
    select_leads,
    toxicity_score,
)

GEO_HEADER = ("label,pose,affinity,ca_dist,ca_dist_far,donor_dist,"
              "n_carboxylate_o,n_donors,passes\n")
HEPATIC = ("DILI", "SR-MMP", "SR-p53", "NR-AhR")


def write_geometry(path, rows):
    with open(path, "w") as handle:
        handle.write(GEO_HEADER)
        for label, pose, affinity, passes in rows:
            handle.write(f"{label},{pose},{affinity},2.7,4.4,2.9,2,2,{passes}\n")
    return str(path)


def write_admet(path, rows):
    fields = (["smiles", "label"] + list(TOXICITY_WEIGHTS) + CONTEXT_ENDPOINTS)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, values in rows:
            row = {"smiles": "CCO", "label": label}
            row.update({k: values.get(k, 0.3) for k in TOXICITY_WEIGHTS})
            row.update({k: values.get(k, -5.0) for k in CONTEXT_ENDPOINTS})
            writer.writerow(row)
    return str(path)


class TestWeights(unittest.TestCase):
    def test_hepatic_endpoints_carry_double_weight(self):
        """The indication is liver fibrosis - the drug is for a liver that is already
        damaged, so hepatotoxicity is not one liability among many."""
        for endpoint in HEPATIC:
            self.assertEqual(TOXICITY_WEIGHTS[endpoint], 2.0, endpoint)
        self.assertEqual(TOXICITY_WEIGHTS["hERG"], 1.0)
        self.assertEqual(TOXICITY_WEIGHTS["AMES"], 1.0)

    def test_cyp_inhibition_is_down_weighted(self):
        """Interaction risk, not direct injury - it should inform the rank, not run it."""
        for endpoint in ("CYP3A4_Veith", "CYP2C9_Veith", "CYP2D6_Veith"):
            self.assertEqual(TOXICITY_WEIGHTS[endpoint], 0.5, endpoint)

    def test_hepatic_group_outweighs_everything_else_combined(self):
        hepatic = sum(TOXICITY_WEIGHTS[e] for e in HEPATIC)
        other = sum(w for e, w in TOXICITY_WEIGHTS.items() if e not in HEPATIC)
        self.assertGreater(hepatic, other)

    def test_sr_mmp_is_included_and_is_the_mechanism_link(self):
        """SR-MMP is mitochondrial membrane potential disruption - a principal DILI
        mechanism, and at AUROC 0.925 the best-performing endpoint in the panel."""
        self.assertIn("SR-MMP", TOXICITY_WEIGHTS)

    def test_weak_endpoints_are_excluded_with_their_numbers(self):
        """Excluded on ADMET-AI's own reported performance, and the reason travels with
        the decision. ClinTox is the instructive one: AUROC 0.928 but AUPRC 0.607, so
        the AUROC is class-imbalance inflation."""
        for endpoint in ("ClinTox", "Carcinogens_Lagunin", "LD50_Zhu",
                         "Bioavailability_Ma", "Skin_Reaction"):
            self.assertIn(endpoint, EXCLUDED_ENDPOINTS)
            self.assertNotIn(endpoint, TOXICITY_WEIGHTS)
            self.assertRegex(EXCLUDED_ENDPOINTS[endpoint], r"0\.\d")

    def test_excluded_endpoints_are_still_reported_as_context(self):
        for endpoint in ("ClinTox", "Carcinogens_Lagunin", "LD50_Zhu"):
            self.assertIn(endpoint, CONTEXT_ENDPOINTS)


class TestToxicityScore(unittest.TestCase):
    def test_uniform_input_gives_that_value(self):
        self.assertAlmostEqual(
            toxicity_score({k: "0.4" for k in TOXICITY_WEIGHTS}), 0.4)

    def test_lower_is_better_and_hepatic_moves_it_more(self):
        base = {k: "0.2" for k in TOXICITY_WEIGHTS}
        hepatic = dict(base, DILI="0.9")
        ddi = dict(base, CYP3A4_Veith="0.9")
        self.assertGreater(toxicity_score(hepatic), toxicity_score(ddi))

    def test_missing_endpoints_are_skipped_not_treated_as_zero(self):
        """Treating an absent prediction as 0.0 would reward a molecule for having no
        prediction at all."""
        partial = {"DILI": "0.8"}
        self.assertAlmostEqual(toxicity_score(partial), 0.8)

    def test_all_missing_gives_nan(self):
        score = toxicity_score({})
        self.assertNotEqual(score, score)


class TestBestPosePerLigand(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "geometry.csv"

    def test_takes_the_best_affinity_among_passing_poses(self):
        write_geometry(self.path, [("gen_00001", 1, -6.0, 1),
                                   ("gen_00001", 2, -7.5, 1)])
        self.assertAlmostEqual(
            float(best_pose_per_ligand(str(self.path))["gen_00001"]["affinity"]), -7.5)

    def test_ignores_poses_that_failed_the_geometry_filter(self):
        """A better-scoring pose that misses the anchors is not a binding mode this
        project accepts, so it must not become the ligand's affinity."""
        write_geometry(self.path, [("gen_00001", 1, -9.9, 0),
                                   ("gen_00001", 2, -6.0, 1)])
        self.assertAlmostEqual(
            float(best_pose_per_ligand(str(self.path))["gen_00001"]["affinity"]), -6.0)

    def test_a_ligand_with_no_passing_pose_is_absent(self):
        write_geometry(self.path, [("gen_00001", 1, -9.9, 0)])
        self.assertEqual(best_pose_per_ligand(str(self.path)), {})


class TestSelectLeads(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        d = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

        # three distinct scaffolds, one repeat of the first
        self.smiles = {
            "gen_00001": "O=C(O)Cc1ccccc1",
            "gen_00002": "O=C(O)Cc1ccncc1",
            "gen_00003": "O=C(O)CCc1ccccc1",      # same Murcko as gen_00001
            "gen_00004": "O=C(O)Cc1cccs1",
        }
        (d / "survivors.smi").write_text(
            "".join(f"{s}\t{l}\n" for l, s in self.smiles.items()))
        self.survivors = str(d / "survivors.smi")

        self.geometry = write_geometry(d / "geometry.csv", [
            ("gen_00001", 1, -8.0, 1),   # beats the reference
            ("gen_00002", 1, -7.5, 1),   # beats
            ("gen_00003", 1, -7.9, 1),   # beats, duplicate scaffold
            ("gen_00004", 1, -6.0, 1),   # does NOT beat
        ])
        self.panel = write_geometry(d / "panel.csv", [
            (f"{PANEL_PREFIX}{REFERENCE}", 1, -7.0, 1),
            (f"{PANEL_PREFIX}A1AFA", 1, -6.7, 1),
        ])
        # Caco-2 values clear the 0.5 log floor against the reference's -5.6, so these
        # tests exercise the affinity filter and the ranking rather than permeability.
        self.admet = write_admet(d / "admet.csv", [
            ("gen_00001", {"DILI": 0.9, "Caco2_Wang": -4.9}),   # worst hepatic
            ("gen_00002", {"DILI": 0.1, "Caco2_Wang": -4.9}),   # best
            ("gen_00003", {"DILI": 0.2, "Caco2_Wang": -4.9}),
            ("gen_00004", {"DILI": 0.05, "Caco2_Wang": -4.9}),
            (f"{PANEL_PREFIX}{REFERENCE}", {"DILI": 0.455, "Caco2_Wang": -5.6}),
        ])

    def _run(self, n=20):
        return select_leads(self.geometry, self.panel, self.admet, self.survivors,
                            None, n)

    def test_hard_filter_drops_anything_not_better_than_the_reference(self):
        result = self._run()
        self.assertAlmostEqual(result["reference_affinity"], -7.0)
        self.assertNotIn("gen_00004", [l["label"] for l in result["leads"]])

    def test_the_filter_is_strict_not_inclusive(self):
        """"Better than" means strictly better; a tie is not better."""
        tie = write_geometry(Path(self.tmp.name) / "tie.csv",
                             [("gen_00002", 1, -7.0, 1)])
        result = select_leads(tie, self.panel, self.admet, self.survivors, None, 20)
        self.assertEqual(result["after_hard_filter"], 0)

    def test_ranked_by_toxicity_ascending(self):
        labels = [l["label"] for l in self._run()["leads"]]
        self.assertEqual(labels[0], "gen_00002")

    def test_one_lead_per_murcko_scaffold(self):
        result = self._run()
        scaffolds = [l["scaffold"] for l in result["leads"]]
        self.assertEqual(len(scaffolds), len(set(scaffolds)))

    def test_the_duplicate_scaffold_keeps_the_less_toxic_member(self):
        """gen_00003 (DILI 0.2) and gen_00001 (0.9) share a scaffold; the ranking must
        decide which represents it."""
        labels = [l["label"] for l in self._run()["leads"]]
        self.assertIn("gen_00003", labels)
        self.assertNotIn("gen_00001", labels)

    def test_n_leads_caps_the_output(self):
        self.assertEqual(len(self._run(n=1)["leads"]), 1)

    def test_the_panel_is_scored_on_the_same_composite(self):
        """"Least toxic" needs something to be least toxic than."""
        panel = self._run()["panel"]
        self.assertTrue(panel)
        self.assertIn(REFERENCE, [p["label"] for p in panel])

    def test_permeability_is_a_hard_filter_with_a_margin(self):
        """§9.2 (as revised by §7.5) pre-registers "beats PLN-1474 on predicted
        permeability", and §9 was frozen before sampling - a lead that misses it is not
        a lead. The margin is the resolvable floor: model MAE 0.3115 against a median
        inter-laboratory spread of 0.57 log on the same compound."""
        self.assertEqual(PERMEABILITY_FLOOR, 0.5)
        self.assertEqual(PERMEABILITY_ENDPOINT, "Caco2_Wang")

    def test_a_candidate_inside_the_noise_floor_is_dropped(self):
        """Better, but not by more than the floor, is not a difference."""
        admet = write_admet(Path(self.tmp.name) / "narrow.csv", [
            ("gen_00002", {"Caco2_Wang": -5.3}),        # 0.3 better than -5.6
            (f"{PANEL_PREFIX}{REFERENCE}", {"Caco2_Wang": -5.6}),
        ])
        result = select_leads(self.geometry, self.panel, admet, self.survivors,
                              None, 20)
        self.assertEqual(result["after_permeability"], 0)
        self.assertEqual(result["leads"], [])

    def test_a_candidate_past_the_floor_survives(self):
        admet = write_admet(Path(self.tmp.name) / "wide.csv", [
            ("gen_00002", {"Caco2_Wang": -4.9}),        # 0.7 better
            (f"{PANEL_PREFIX}{REFERENCE}", {"Caco2_Wang": -5.6}),
        ])
        result = select_leads(self.geometry, self.panel, admet, self.survivors,
                              None, 20)
        self.assertEqual(result["after_permeability"], 1)
        self.assertEqual([l["label"] for l in result["leads"]], ["gen_00002"])

    def test_the_funnel_reports_each_stage_separately(self):
        """Affinity and permeability are distinct filters and their counts are what
        make the trade-off between them visible."""
        result = self._run()
        self.assertIn("after_affinity", result)
        self.assertIn("after_permeability", result)
        self.assertLessEqual(result["after_permeability"], result["after_affinity"])

    def test_leads_beating_the_reference_toxicity_are_flagged(self):
        """Most leads do NOT, once permeability is enforced - the two axes pull against
        each other. The flag is what keeps that visible instead of implied."""
        result = self._run()
        for lead in result["leads"]:
            self.assertIn("beats_reference_toxicity", lead)
            self.assertEqual(lead["beats_reference_toxicity"],
                             int(lead["toxicity_score"] < result["reference_toxicity"]))

    def test_missing_admet_reference_row_raises(self):
        """Without PLN-1474's ADMET row there is no permeability or toxicity reference,
        and silently ranking against nothing would look like a result."""
        admet = write_admet(Path(self.tmp.name) / "nopanel_admet.csv",
                            [("gen_00002", {})])
        with self.assertRaises(ValueError):
            select_leads(self.geometry, self.panel, admet, self.survivors, None, 20)

    def test_missing_reference_raises_rather_than_guessing(self):
        empty = write_geometry(Path(self.tmp.name) / "nopanel.csv",
                               [(f"{PANEL_PREFIX}A1AFA", 1, -6.7, 1)])
        with self.assertRaises(ValueError):
            select_leads(self.geometry, empty, self.admet, self.survivors, None, 20)


if __name__ == "__main__":
    unittest.main()
