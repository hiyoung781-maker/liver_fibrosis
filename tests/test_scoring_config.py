import sys
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import objective
from make_scoring_config import (
    FRAG,
    SKETCH,
    aggregator_type,
    build_config,
    strip_stage_prefix,
)
from novelty import load_smi


class TestStripStagePrefix(unittest.TestCase):
    def test_rewrites_table_headers(self):
        text = "[[stage.scoring.component]]\n[stage.scoring.component.TPSA]\n"
        self.assertEqual(
            strip_stage_prefix(text),
            "[[scoring.component]]\n[scoring.component.TPSA]\n",
        )

    def test_leaves_comments_and_values_untouched(self):
        """A comment mentioning stage.scoring, or a SMARTS containing a bracket,
        must survive verbatim - the frag's comments carry the rationale that makes
        the generated config auditable."""
        text = '# see [stage.scoring] in the sketch\nparams.smarts = "[CX3](=O)[OX2H1,OX1-]"\n'
        self.assertEqual(strip_stage_prefix(text), text)

    def test_only_rewrites_the_stage_prefix(self):
        text = "[[stage.scoring.component]]\n[diversity_filter]\n"
        self.assertEqual(
            strip_stage_prefix(text), "[[scoring.component]]\n[diversity_filter]\n"
        )


class TestAggregator(unittest.TestCase):
    def test_reads_the_aggregator_from_the_sketch(self):
        """Not hardcoded. The frag holds the components but the sketch declares
        [stage.scoring] type, and §5.4's three-file SlogP disagreement is what
        hand-copying either of them risks repeating."""
        self.assertEqual(aggregator_type(), "geometric_mean")


class TestBuildConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = build_config("panel.smi", "out.csv")
        cls.doc = tomllib.loads(cls.text)
        with open(FRAG, "rb") as handle:
            cls.frag = tomllib.load(handle)

    def test_declares_a_scoring_run(self):
        self.assertEqual(self.doc["run_type"], "scoring")
        self.assertEqual(self.doc["parameters"]["smiles_file"], "panel.smi")
        self.assertEqual(self.doc["parameters"]["output_csv"], "out.csv")

    def test_aggregator_matches_the_sketch(self):
        self.assertEqual(self.doc["scoring"]["type"], aggregator_type())

    def test_component_tree_is_identical_to_the_frag(self):
        """The whole point of generating rather than copying: §7 requires the
        IDENTICAL scoring function from §5.1, and this makes that checkable."""
        self.assertEqual(
            self.doc["scoring"]["component"],
            self.frag["stage"]["scoring"]["component"],
        )

    def test_carries_the_four_components_in_order(self):
        names = [next(iter(c)) for c in self.doc["scoring"]["component"]]
        self.assertEqual(names, ["GroupCount", "TPSA", "SAScore", "CustomAlerts"])

    def test_carries_the_narrowed_aniline_pattern(self):
        """§7's precondition, pinned in the generated artifact itself."""
        alerts = self.doc["scoring"]["component"][-1]["CustomAlerts"]["endpoint"][0]
        self.assertIn("[NX3;H2][c]", alerts["params"]["smarts"])
        self.assertNotIn("[NH2,NH][c]", alerts["params"]["smarts"])

    def test_committed_config_matches_the_generator(self):
        """configs/score_panel.toml is committed, so it can drift from the frag.
        This fails if the frag changes and the config is not regenerated."""
        committed = Path("configs/score_panel.toml")
        expected = build_config(
            "results/panel_curated.smi", "results/score_panel.csv"
        )
        self.assertEqual(committed.read_text(), expected)


class TestPanelPrecondition(unittest.TestCase):
    """§7 forbids running without checking this, so it is a test, not a note.

    The old `[NH2,NH][c]` zeroed 5 of the 6 positive-panel molecules on EVERY
    component, which collapses the reference distribution §7 exists to produce.
    Exempting the panel from the filter is not the fix and would be worse: §9.5
    requires the panel and the generated set to pass the SAME funnel.
    """

    def test_no_panel_molecule_trips_any_alert_or_pains(self):
        for smiles, label in load_smi("data/benchmark_panel.smi"):
            with self.subTest(label=label):
                mol = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(mol)
                for pattern, smarts in zip(objective._ALERTS, objective.ALERT_SMARTS):
                    self.assertFalse(mol.HasSubstructMatch(pattern),
                                     f"{label} matches {smarts}")
                self.assertFalse(objective._PAINS.HasMatch(mol),
                                 f"{label} matches PAINS")

    def test_every_panel_molecule_carries_the_carboxylate(self):
        """The MIDAS anchor gate is a hard cut. A positive-panel molecule without
        the carboxylate would score at GATE_FLOOR and make the panel useless as a
        reference."""
        scored = objective.score_smiles(
            [s for s, _ in load_smi("data/benchmark_panel.smi")])
        self.assertEqual(len(scored["smiles"]), 6)
        self.assertTrue(all(scored["cooh"] > 0))


if __name__ == "__main__":
    unittest.main()
