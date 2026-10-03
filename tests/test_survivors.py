import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import objective
from build_survivors import (WINDOW, V4_WINDOW, passes_window,
                             library_path, survivors_path, window_for_arm,
                             write_survivors)
from novelty import load_smi, murcko

# A generated molecule that clears every §8.2 cut.
GOOD = "CCC1(CC(=O)O)c2cccn2-c2cccnc2CN1S(=O)(=O)c1ccc(-c2ccccc2)cc1F"
NO_ACID = "c1ccccc1CCNc1ccccc1"
TOO_GREASY = "CCCCCCCCCCCCCCCCCCc1ccc(C(=O)O)cc1"


class TestWindow(unittest.TestCase):
    def test_window_values_are_the_pre_registered_ones(self):
        """§8.2 pre-registers these. A silent edit here would move the goalposts
        after §6.3 already reported the survivor count against them."""
        self.assertEqual(WINDOW["logp_max"], 5.0)
        self.assertEqual(WINDOW["mw_range"], (250.0, 550.0))
        self.assertEqual(WINDOW["tpsa_range"], (40.0, 115.0))

    def test_tpsa_window_matches_the_rl_objective(self):
        """§8.2 says TPSA uses the same window as the RL objective. The RL side is
        a double_sigmoid over (low, high); triage takes that as a hard range."""
        self.assertEqual(WINDOW["tpsa_range"],
                         (objective.TPSA_WINDOW[0], objective.TPSA_WINDOW[1]))


class TestPassesWindow(unittest.TestCase):
    def test_accepts_a_compliant_molecule(self):
        self.assertTrue(passes_window(Chem.MolFromSmiles(GOOD)))

    def test_rejects_excess_lipophilicity(self):
        """The §6.3 finding: logP is the cut that removes most of the library."""
        self.assertFalse(passes_window(Chem.MolFromSmiles(TOO_GREASY)))

    def test_rejects_unparseable(self):
        self.assertFalse(passes_window(None))


class TestWriteSurvivors(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.library = str(Path(self.tmp.name) / "library.smi")
        self.out = str(Path(self.tmp.name) / "survivors.smi")

    def test_applies_every_cut_and_labels_sequentially(self):
        Path(self.library).write_text(
            "# header\n"
            f"{GOOD} {GOOD}\n"
            f"{NO_ACID} {NO_ACID}\n"
            f"{TOO_GREASY} {TOO_GREASY}\n"
        )
        known = set()
        counts = write_survivors(self.out, self.library, known)
        rows = load_smi(self.out)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "gen_00001")
        self.assertEqual(counts["survivors"], 1)
        self.assertEqual(counts["input"], 3)

    def test_murcko_gate_uses_the_known_set(self):
        """§8.3's gate. Putting GOOD's own scaffold in `known` must remove it."""
        Path(self.library).write_text(f"{GOOD} {GOOD}\n")
        known = {murcko(Chem.MolFromSmiles(GOOD))}
        counts = write_survivors(self.out, self.library, known)
        self.assertEqual(counts["survivors"], 0)

    def test_header_records_every_cut_count(self):
        Path(self.library).write_text(f"{GOOD} {GOOD}\n{NO_ACID} {NO_ACID}\n")
        write_survivors(self.out, self.library, set())
        header = Path(self.out).read_text()
        for key in ("carboxylate", "alert-free", "logP", "MW", "TPSA", "Murcko"):
            self.assertIn(key, header)

    def test_reproduces_the_reported_7767(self):
        """§6.3 and the ADMET run both report 7,767 survivors of data/library.smi.
        This pins that number to the committed inputs."""
        with open("data/known_scaffolds.smi") as handle:
            known = {line.strip() for line in handle
                     if line.strip() and not line.startswith("#")}
        counts = write_survivors(self.out, "data/library.smi", known)
        self.assertEqual(counts["input"], 19485)
        self.assertEqual(counts["survivors"], 7767)


class TestV4WindowIsSeparateFromTheBlueprintOne(unittest.TestCase):
    """§8.2's comment says the gate must not diverge from the RL objective. For v4
    that intent points away from objective.TPSA_WINDOW, not at it: v4's RL window
    was re-derived to (60, 140) (results/v4_tpsa_window.md) while the blueprint
    constant stays at (40, 115).

    WHY THIS IS A SECOND CONSTANT. TestReportedCounts recomputes the 7,767
    survivors of data/library.smi that §6.3 and the ADMET run both cite. Moving
    WINDOW itself would change that number under a published result, so the
    blueprint window remains the default and v4 passes its own."""

    def test_the_blueprint_window_is_untouched(self):
        self.assertEqual(WINDOW["tpsa_range"], (40.0, 115.0))

    def test_the_v4_window_comes_from_the_rl_single_source(self):
        from make_rl_config import RL_TRANSFORM_OVERRIDES

        override = RL_TRANSFORM_OVERRIDES["TPSA"]
        self.assertEqual(V4_WINDOW["tpsa_range"], (override["low"], override["high"]))
        self.assertEqual(V4_WINDOW["tpsa_range"], (60.0, 140.0))

    def test_only_the_tpsa_bound_differs_between_the_two_windows(self):
        """logP and MW were not re-derived, so a v4 survivor set must differ from
        a v2 one on TPSA alone - otherwise two cuts moved and the funnel's
        attribution to §2 would be wrong."""
        self.assertEqual(V4_WINDOW["logp_max"], WINDOW["logp_max"])
        self.assertEqual(V4_WINDOW["mw_range"], WINDOW["mw_range"])

    def test_v4_arms_get_the_v4_window_and_v2_arms_do_not(self):
        self.assertEqual(window_for_arm("TL-B"), V4_WINDOW)
        for arm in ("TL-C", "TL-A-prime"):
            with self.subTest(arm=arm):
                self.assertEqual(window_for_arm(arm), WINDOW)

    def test_an_undeclared_arm_falls_back_to_the_pre_registered_window(self):
        """Forgetting to declare an arm must not silently widen the gate. An arm
        has to be in make_rl_config.ARMS to be treated as v4."""
        self.assertEqual(window_for_arm("TL-Z"), WINDOW)
        self.assertEqual(window_for_arm(None), WINDOW)

    def test_a_molecule_between_the_two_ceilings_separates_them(self):
        """TPSA ~121 is what TL-B's median actually looks like (spec §3.4 measured
        121.8): rejected by the old ceiling, admitted by the new one. This is the
        molecule the old gate would have thrown away."""
        mid = Chem.MolFromSmiles(
            "O=C(O)CNC(=O)c1ccc(NC(=O)c2ccc(C(=O)NC)cc2)cc1")
        from rdkit.Chem import Descriptors
        self.assertTrue(115.0 < Descriptors.TPSA(mid) < 140.0,
                        Descriptors.TPSA(mid))
        self.assertFalse(passes_window(mid, WINDOW))
        self.assertTrue(passes_window(mid, V4_WINDOW))

    def test_paths_are_scoped_by_campaign_not_hardcoded_to_v2(self):
        self.assertEqual(library_path("TL-B"), "data/v4_TL-B/library.smi")
        self.assertEqual(survivors_path("TL-B"), "data/v4_TL-B/survivors.smi")
        self.assertEqual(library_path("TL-C"), "data/v2_TL-C/library.smi")
        self.assertEqual(survivors_path("TL-C"), "data/v2_TL-C/survivors.smi")


if __name__ == "__main__":
    unittest.main()
