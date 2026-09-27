import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import objective
from build_survivors import WINDOW, passes_window, write_survivors
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


if __name__ == "__main__":
    unittest.main()
