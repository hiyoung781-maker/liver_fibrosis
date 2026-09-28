import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from admet_input import (
    EXCLUDE_LABELS,
    PANEL_PREFIX,
    passing_labels,
    summarise,
    write_admet_input,
)
from novelty import load_smi

GEOMETRY_HEADER = ("label,pose,affinity,ca_dist,ca_dist_far,donor_dist,"
                   "n_carboxylate_o,n_donors,passes\n")


def geometry_csv(path, rows):
    with open(path, "w") as handle:
        handle.write(GEOMETRY_HEADER)
        for label, pose, passes in rows:
            handle.write(f"{label},{pose},-6.5,2.7,4.4,2.9,2,2,{passes}\n")
    return str(path)


class TestPassingLabels(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "geometry.csv"

    def test_a_ligand_qualifies_if_any_pose_passed(self):
        """§8.5b asks whether a binding mode EXISTS, not whether every sampled mode
        is one. Requiring all poses to pass would measure something else."""
        geometry_csv(self.path, [("gen_00001", 1, 0), ("gen_00001", 2, 1),
                                 ("gen_00002", 1, 0)])
        self.assertEqual(passing_labels(str(self.path)), ["gen_00001"])

    def test_each_label_appears_once(self):
        geometry_csv(self.path, [("gen_00001", 1, 1), ("gen_00001", 2, 1),
                                 ("gen_00001", 3, 1)])
        self.assertEqual(passing_labels(str(self.path)), ["gen_00001"])

    def test_the_control_is_excluded(self):
        """CONTROL_crystal is the redocking reference, not a candidate. Scoring it as
        one would put the crystal ligand into the lead set."""
        geometry_csv(self.path, [("CONTROL_crystal", 1, 1), ("gen_00001", 1, 1)])
        self.assertEqual(passing_labels(str(self.path)), ["gen_00001"])
        self.assertIn("CONTROL_crystal", EXCLUDE_LABELS)

    def test_no_passing_poses_gives_an_empty_list(self):
        geometry_csv(self.path, [("gen_00001", 1, 0)])
        self.assertEqual(passing_labels(str(self.path)), [])

    def test_missing_column_raises(self):
        """A silent empty list looks like a run where nothing passed."""
        self.path.write_text("label,pose\ngen_00001,1\n")
        with self.assertRaises(KeyError):
            passing_labels(str(self.path))


class TestWriteAdmetInput(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.geometry = Path(self.tmp.name) / "geometry.csv"
        self.survivors = Path(self.tmp.name) / "survivors.smi"
        self.survivors.write_text(
            "# header\nCC(=O)Oc1ccccc1C(=O)O\tgen_00001\n"
            "O=C(O)Cc1ccccc1\tgen_00002\n")
        self.out = str(Path(self.tmp.name) / "admet_in.csv")

    def test_writes_survivors_and_panel_together(self):
        """§7.5 made the permeability comparison prediction-to-prediction, which only
        holds if both sides pass through one model in one run."""
        geometry_csv(self.geometry, [("gen_00001", 1, 1), ("gen_00002", 1, 0)])
        counts = write_admet_input(self.out, str(self.geometry),
                                   str(self.survivors))
        self.assertEqual(counts["written"], 1)
        # data/benchmark_panel.smi holds the 4 target-profile panel compounds
        # (PANEL_COMPOUNDS + CHEMBL4649232 + A1AFA), not the pan-alphaV tools.
        self.assertEqual(counts["panel"], 4)
        rows = list(csv.DictReader(open(self.out)))
        labels = [r["label"] for r in rows]
        self.assertIn("gen_00001", labels)
        self.assertNotIn("gen_00002", labels)
        self.assertEqual(sum(1 for l in labels if l.startswith(PANEL_PREFIX)), 4)

    def test_panel_labels_are_prefixed_so_they_can_be_separated(self):
        geometry_csv(self.geometry, [("gen_00001", 1, 1)])
        write_admet_input(self.out, str(self.geometry), str(self.survivors))
        panel = [r["label"] for r in csv.DictReader(open(self.out))
                 if r["label"].startswith(PANEL_PREFIX)]
        self.assertIn(f"{PANEL_PREFIX}PLN-1474", panel)

    def test_smiles_come_from_the_survivor_file_not_the_geometry_csv(self):
        geometry_csv(self.geometry, [("gen_00001", 1, 1)])
        write_admet_input(self.out, str(self.geometry), str(self.survivors))
        row = next(r for r in csv.DictReader(open(self.out))
                   if r["label"] == "gen_00001")
        self.assertEqual(row["smiles"], "CC(=O)Oc1ccccc1C(=O)O")

    def test_a_label_absent_from_survivors_is_reported(self):
        """The two files must come from the same library. If a passing label is not in
        data/survivors.smi they do not, and proceeding would silently drop molecules."""
        geometry_csv(self.geometry, [("gen_99999", 1, 1)])
        counts = write_admet_input(self.out, str(self.geometry),
                                   str(self.survivors))
        self.assertEqual(counts["missing_from_survivors"], ["gen_99999"])
        self.assertEqual(counts["written"], 0)

    def test_the_real_survivor_file_covers_a_plausible_label(self):
        """Guards the label convention itself: gen_NNNNN, zero-padded to five."""
        labels = {label for _, label in load_smi("data/survivors.smi")}
        self.assertIn("gen_00001", labels)
        self.assertEqual(len(labels), 7767)


class TestSummarise(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.preds = Path(self.tmp.name) / "preds.csv"
        header = ["smiles", "label", "tpsa", "Caco2_Wang", "hERG", "AMES", "DILI",
                  "ClinTox", "Carcinogens_Lagunin", "LD50_Zhu", "CYP3A4_Veith",
                  "CYP2C9_Veith", "CYP2D6_Veith", "PAMPA_NCATS", "HIA_Hou",
                  "Bioavailability_Ma", "Pgp_Broccatelli", "Solubility_AqSolDB"]
        rows = []
        for i in range(1, 4):
            rows.append(["CCO", f"gen_{i:05d}"] + [str(80 + i)] + [str(-5.0 - 0.1 * i)]
                        + ["0.3"] * 15)
        rows.append(["CCO", f"{PANEL_PREFIX}PLN-1474", "100.6", "-5.586"]
                    + ["0.4"] * 15)
        with open(self.preds, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def test_reports_survivor_percentiles_and_the_panel(self):
        text = summarise(str(self.preds))
        self.assertIn("Caco-2", text)
        self.assertIn("PLN-1474", text)

    def test_states_that_it_is_a_screen_not_a_gate(self):
        """§7.5 removed ADMET from §9's criteria. A table of numbers invites being read
        as a gate, so the output says otherwise in its own header."""
        text = summarise(str(self.preds))
        self.assertIn("NOT A GATE", text)

    def test_reports_the_noise_floor_and_the_tpsa_correlation(self):
        """Both are what make the permeability figure interpretable: model error is the
        size of assay disagreement, and the signal is partly an axis we chose."""
        text = summarise(str(self.preds))
        self.assertIn("0.5 log", text)
        self.assertIn("TPSA", text)

    def test_writes_to_a_file_when_asked(self):
        out = Path(self.tmp.name) / "summary.txt"
        summarise(str(self.preds), str(out))
        self.assertIn("Caco-2", out.read_text())


if __name__ == "__main__":
    unittest.main()
