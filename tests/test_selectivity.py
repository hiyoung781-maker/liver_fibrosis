import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from selectivity import (CALIBRATION, LigandMismatchError, MEASURED_IC50,
                         calibration_verdict, deltas, ligand_identity,
                         merge_affinities, percentile_ranks,
                         rank_selectivity, rank_sum, shared_ligands)

# alphaVbeta1 and alphaVbeta6 scores, kcal/mol, same ligand each row.
AVB1 = {"PLN-1474": -6.8, "CHEMBL4649232": -7.2, "CWHM-12": -7.0,
        "GLPG0187": -7.4, "A1AFA": -6.9, "bexotegrast": -6.5}


class TestDeltas(unittest.TestCase):
    """spec 10.1 never compares absolute scores between receptors -- AD4 and
    Vina scores scale with ligand size, so only the SAME ligand's difference
    cancels that. Delta = affinity(subtype) - affinity(alphaVbeta1); a larger
    (more positive) Delta means the off-target binds worse, i.e. selective."""

    def test_delta_is_subtype_minus_target(self):
        d = deltas(AVB1, {"PLN-1474": -5.8})
        self.assertAlmostEqual(d["PLN-1474"], 1.0)

    def test_a_ligand_missing_from_either_side_is_dropped_not_zeroed(self):
        # A zero would read as "equally potent on both", which is a claim.
        d = deltas(AVB1, {"PLN-1474": -5.8, "unknown": -7.0})
        self.assertEqual(set(d), {"PLN-1474"})

    def test_more_positive_is_more_selective(self):
        d = deltas(AVB1, {"PLN-1474": -5.0, "CWHM-12": -7.5})
        self.assertGreater(d["PLN-1474"], d["CWHM-12"])


class TestCalibrationGate(unittest.TestCase):
    """spec 10.4, pre-registered: CWHM-12 and GLPG0187 must come out
    NON-selective and PLN-1474 and CHEMBL4649232 SELECTIVE. The answer is
    measured IC50, not a decoy set, which makes it a stronger control -- and
    if the protocol cannot reproduce it, no selectivity statement about a
    generated molecule is made and criterion 6 is recorded unevaluable."""

    def test_the_four_controls_are_the_registered_ones(self):
        self.assertEqual(set(CALIBRATION["non_selective"]),
                         {"CWHM-12", "GLPG0187"})
        self.assertEqual(set(CALIBRATION["selective"]),
                         {"PLN-1474", "CHEMBL4649232"})

    def test_passes_when_the_selective_pair_ranks_above_the_other(self):
        d = {"PLN-1474": 1.2, "CHEMBL4649232": 1.0,
             "CWHM-12": -0.3, "GLPG0187": 0.1}
        self.assertTrue(calibration_verdict(d)["passed"])

    def test_fails_when_a_non_selective_control_looks_selective(self):
        d = {"PLN-1474": 1.2, "CHEMBL4649232": 1.0,
             "CWHM-12": 2.0, "GLPG0187": 0.1}
        v = calibration_verdict(d)
        self.assertFalse(v["passed"])
        self.assertIn("CWHM-12", str(v["failed"]))

    def test_a_prefixed_control_label_is_still_found(self):
        # The reference set is docked under whichever name its input used.
        # Reading PANEL_PLN-1474 as "PLN-1474 is missing" costs a full
        # re-dock to discover, and the report looks identical either way.
        d = {"PANEL_PLN-1474": 1.2, "PANEL_CHEMBL4649232": 1.0,
             "PANEL_CWHM-12": -0.3, "PANEL_GLPG0187": 0.1}
        v = calibration_verdict(d)
        self.assertTrue(v["passed"], v["failed"])
        self.assertEqual(v["missing"], [])

    def test_the_two_naming_conventions_can_be_mixed(self):
        d = {"PLN-1474": 1.2, "PANEL_CHEMBL4649232": 1.0,
             "CWHM-12": -0.3, "PANEL_GLPG0187": 0.1}
        self.assertTrue(calibration_verdict(d)["passed"])

    def test_a_missing_control_fails_rather_than_being_skipped(self):
        d = {"PLN-1474": 1.2, "CWHM-12": -0.3, "GLPG0187": 0.1}
        v = calibration_verdict(d)
        self.assertFalse(v["passed"])
        self.assertIn("CHEMBL4649232", str(v["failed"]))

    def test_the_measured_ic50s_travel_with_the_gate(self):
        # The gate's answer key. CWHM-12 is nearly equipotent on avb1 and
        # avb6 (1.8 against 1.5 nM), which is why one isoform gives this gate
        # little to discriminate with -- a limitation, not a pass.
        self.assertAlmostEqual(MEASURED_IC50["CWHM-12"]["avb1"], 1.8)
        self.assertAlmostEqual(MEASURED_IC50["CWHM-12"]["avb6"], 1.5)
        self.assertAlmostEqual(MEASURED_IC50["GLPG0187"]["avb1"], 1.3)
        self.assertAlmostEqual(MEASURED_IC50["GLPG0187"]["avb6"], 1.4)


class TestRankSum(unittest.TestCase):
    """spec 10.1: rank each validated isoform's Delta DESCENDING (1 = most
    selective), and a molecule's rank-sum is the reverse of the sum of those
    ranks, so a LARGER value is more selective. Per-isoform score scales are
    absorbed by the ranking step."""

    def test_larger_is_more_selective(self):
        scores = rank_sum({"avb6": {"a": 2.0, "b": 0.5, "c": -1.0}})
        self.assertGreater(scores["a"], scores["b"])
        self.assertGreater(scores["b"], scores["c"])

    def test_one_isoform_is_enough_to_compute_it(self):
        scores = rank_sum({"avb6": {"a": 1.0, "b": 0.0}})
        self.assertEqual(set(scores), {"a", "b"})

    def test_ranks_combine_across_isoforms(self):
        # 'a' is best on one and worst on the other; 'b' the reverse. Equal.
        scores = rank_sum({"x": {"a": 2.0, "b": 1.0},
                           "y": {"a": 1.0, "b": 2.0}})
        self.assertAlmostEqual(scores["a"], scores["b"])

    def test_a_molecule_absent_from_an_isoform_is_not_ranked_there(self):
        scores = rank_sum({"x": {"a": 2.0, "b": 1.0}, "y": {"a": 1.0}})
        self.assertIn("b", scores)

    def test_no_isoforms_gives_no_scores(self):
        self.assertEqual(rank_sum({}), {})


class TestWithinReceptorRanks(unittest.TestCase):
    """Delta = score(subtype) - score(target) cancels the LIGAND-side
    systematic error, which is what spec 10.1 argues, and leaves the
    RECEPTOR-side offset untouched -- a larger, more buried or more
    hydrophobic pocket scores every ligand better. Delta then conflates "this
    ligand prefers alphaVbeta1" with "alphaVbeta6 scores everything
    generously", and selectivity is exactly the difference between those two.

    Ranking within each receptor removes the offset, because each receptor's
    ranking is normalised inside that receptor."""

    def test_a_uniform_receptor_offset_does_not_move_the_rank_measure(self):
        target = {"a": -9.0, "b": -8.0, "c": -7.0}
        subtype = {k: v + 1.5 for k, v in target.items()}   # all 1.5 weaker
        sel = rank_selectivity(target, subtype)
        # Every Delta is +1.5, which would read as uniformly selective.
        self.assertTrue(all(abs(v - 1.5) < 1e-9
                            for v in deltas(target, subtype).values()))
        # The rank measure says none of them is selective relative to another.
        self.assertTrue(all(abs(v) < 1e-9 for v in sel.values()))

    def test_it_detects_a_genuine_preference(self):
        target = {"a": -9.0, "b": -8.0, "c": -7.0}
        subtype = {"a": -7.0, "b": -8.0, "c": -9.0}   # order reversed
        sel = rank_selectivity(target, subtype)
        self.assertGreater(sel["a"], sel["c"])

    def test_percentiles_run_from_zero_to_one_with_strongest_highest(self):
        pct = percentile_ranks({"a": -9.0, "b": -8.0, "c": -7.0})
        self.assertAlmostEqual(pct["a"], 1.0)
        self.assertAlmostEqual(pct["c"], 0.0)

    def test_ties_share_a_percentile(self):
        pct = percentile_ranks({"a": -8.0, "b": -8.0})
        self.assertAlmostEqual(pct["a"], pct["b"])

    def test_a_single_molecule_has_no_rank_information(self):
        # A percentile needs a population; one molecule is not one.
        self.assertEqual(percentile_ranks({"a": -8.0}), {})

    def test_only_molecules_present_on_both_sides_are_scored(self):
        sel = rank_selectivity({"a": -9.0, "b": -8.0, "c": -7.0},
                               {"a": -7.0, "b": -8.0})
        self.assertEqual(set(sel), {"a", "b"})

    def test_one_shared_molecule_yields_nothing(self):
        # It cannot be ranked against anything, so there is no rank
        # information to report -- an empty result, not a zero.
        self.assertEqual(rank_selectivity({"a": -9.0, "b": -8.0},
                                          {"a": -7.0}), {})


POSE = """\
MODEL 1
REMARK SMILES {smiles}
REMARK VINA RESULT:      {score}      0.000      0.000
ATOM      1  C   UNL     1       1.000   2.000   3.000  1.00  0.00    +0.000 C
ENDMDL
"""


def pose_dir(tmp, entries):
    d = Path(tmp)
    d.mkdir(parents=True, exist_ok=True)
    for label, smiles, score in entries:
        (d / f"{label}_out.pdbqt").write_text(
            POSE.format(smiles=smiles, score=score))
    return str(d)


class TestLigandIdentity(unittest.TestCase):
    """The Delta and the rank comparison both assume the SAME ligand file was
    docked against both receptors. Uni-Dock copies the input PDBQT's REMARK
    SMILES into its output, so that line identifies the molecule that was
    actually docked -- and it is how the protomer mix-up would have been
    caught earlier, where the same label meant two different molecules."""

    def test_it_reads_the_remark_smiles(self):
        text = POSE.format(smiles="CC(=O)[O-]", score="-7.0")
        self.assertEqual(ligand_identity(text), "CC(=O)[O-]")

    def test_a_pose_without_the_remark_has_no_identity(self):
        self.assertIsNone(ligand_identity("MODEL 1\nENDMDL\n"))


class TestSharedLigands(unittest.TestCase):

    def test_matching_identities_are_shared(self):
        with TemporaryDirectory() as tmp:
            a = pose_dir(Path(tmp) / "a", [("x", "CCO", "-7.0")])
            b = pose_dir(Path(tmp) / "b", [("x", "CCO", "-6.0")])
            self.assertEqual(shared_ligands(a, b), {"x"})

    def test_the_same_label_with_a_different_molecule_raises(self):
        # Exactly the failure the protomer re-run produced: one label, two
        # molecules. Silently comparing them would compare nothing.
        with TemporaryDirectory() as tmp:
            a = pose_dir(Path(tmp) / "a", [("x", "CCO", "-7.0")])
            b = pose_dir(Path(tmp) / "b", [("x", "CCN", "-6.0")])
            with self.assertRaises(LigandMismatchError) as cm:
                shared_ligands(a, b)
            self.assertIn("x", str(cm.exception))

    def test_a_label_present_on_one_side_only_is_simply_not_shared(self):
        with TemporaryDirectory() as tmp:
            a = pose_dir(Path(tmp) / "a", [("x", "CCO", "-7.0"),
                                           ("y", "CCC", "-7.5")])
            b = pose_dir(Path(tmp) / "b", [("x", "CCO", "-6.0")])
            self.assertEqual(shared_ligands(a, b), {"x"})


class TestMergeAffinities(unittest.TestCase):
    """alphaVbeta1's scores are spread over several docking directories -- the
    two arms and the reference set -- because they were run separately. The
    target side is assembled from all of them."""

    def test_directories_are_merged(self):
        with TemporaryDirectory() as tmp:
            a = pose_dir(Path(tmp) / "a", [("x", "CCO", "-7.0")])
            b = pose_dir(Path(tmp) / "b", [("y", "CCC", "-6.0")])
            merged = merge_affinities([a, b])
            self.assertEqual(set(merged), {"x", "y"})

    def test_a_label_in_two_directories_with_two_molecules_raises(self):
        with TemporaryDirectory() as tmp:
            a = pose_dir(Path(tmp) / "a", [("x", "CCO", "-7.0")])
            b = pose_dir(Path(tmp) / "b", [("x", "CCN", "-6.0")])
            with self.assertRaises(LigandMismatchError):
                merge_affinities([a, b])

    def test_the_same_molecule_twice_keeps_the_better_score(self):
        with TemporaryDirectory() as tmp:
            a = pose_dir(Path(tmp) / "a", [("x", "CCO", "-7.0")])
            b = pose_dir(Path(tmp) / "b", [("x", "CCO", "-8.0")])
            self.assertAlmostEqual(merge_affinities([a, b])["x"], -8.0)


if __name__ == "__main__":
    unittest.main()
