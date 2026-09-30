import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cpd25_gate import CASES, GATE_RULES, decide, gate_predicate


def pose(n, affinity, metal=1, asn224=0, window=0, sidechain=0,
         residues="", status="ok"):
    return {"label": "CHEMBL5532604", "pose": str(n), "affinity": str(affinity),
            "status": status, "metal_ca501": str(metal),
            "hbond_asn224": str(asn224), "hbond_anchor_window": str(window),
            "hbond_asn224_sidechain": str(sidechain),
            "hbond_bb_residues": residues, "tyr178_contact": "0",
            "hydrophobic_pocket": "1"}


class TestWhichPoseIsJudged(unittest.TestCase):
    """§8.2: the best-affinity pose AMONG THOSE reporting a Ca501 metal
    complex. A top-scoring pose that misses the MIDAS is not a model of how
    this chemotype binds, and reading the anchor off it would define the gate
    from a pose the gate itself rejects."""

    def test_a_better_scoring_pose_without_the_metal_is_not_judged(self):
        rows = [pose(1, -9.5, metal=0, asn224=1),   # best, but no MIDAS
                pose(2, -8.0, metal=1, asn224=0, window=1)]
        verdict = decide(rows)
        self.assertEqual(verdict["pose"]["pose"], "2")
        self.assertEqual(verdict["case"], 2)

    def test_the_best_metal_pose_wins_among_several(self):
        rows = [pose(1, -7.0, asn224=0), pose(2, -8.4, asn224=1),
                pose(3, -7.9, asn224=0)]
        self.assertEqual(decide(rows)["pose"]["pose"], "2")

    def test_failed_rows_are_not_judged(self):
        rows = [pose(1, -9.9, asn224=1, status="plip_failed"),
                pose(2, -7.0, asn224=0)]
        verdict = decide(rows)
        self.assertEqual(verdict["pose"]["pose"], "2")
        self.assertEqual(verdict["n_poses"], 1)


class TestTheFourCases(unittest.TestCase):
    def test_case_1_leaves_the_gate_unchanged(self):
        verdict = decide([pose(1, -8.0, asn224=1, window=1, residues="224")])
        self.assertEqual(verdict["case"], 1)
        self.assertEqual(verdict["gate"]["columns"],
                         ["metal_ca501", "hbond_asn224"])

    def test_case_2_relaxes_to_the_anchor_window(self):
        verdict = decide([pose(1, -8.0, asn224=0, window=1, residues="225")])
        self.assertEqual(verdict["case"], 2)
        self.assertEqual(verdict["gate"]["columns"],
                         ["metal_ca501", "hbond_anchor_window"])

    def test_case_3_reduces_to_the_metal_alone(self):
        verdict = decide([pose(1, -8.0, asn224=0, window=0)])
        self.assertEqual(verdict["case"], 3)
        self.assertEqual(verdict["gate"]["columns"], ["metal_ca501"])

    def test_case_4_when_no_pose_reaches_the_metal(self):
        verdict = decide([pose(1, -8.0, metal=0), pose(2, -7.0, metal=0)])
        self.assertEqual(verdict["case"], 4)
        self.assertIsNone(verdict["pose"])
        self.assertEqual(verdict["gate"]["columns"],
                         ["metal_ca501", "hbond_asn224"])

    def test_asn224_takes_precedence_over_the_window(self):
        # Case 2 is reached only when Asn224 is ABSENT. A pose with both must
        # not relax the gate.
        verdict = decide([pose(1, -8.0, asn224=1, window=1)])
        self.assertEqual(verdict["case"], 1)

    def test_a_sidechain_contact_alone_does_not_reach_case_1(self):
        # §8.0 ⓑ names the backbone O; the side-chain column is an
        # observation and must not decide the gate.
        verdict = decide([pose(1, -8.0, asn224=0, window=0, sidechain=1)])
        self.assertEqual(verdict["case"], 3)

    def test_every_case_has_a_rule_and_a_description(self):
        self.assertEqual(set(CASES), set(GATE_RULES))
        self.assertEqual(set(CASES), {1, 2, 3, 4})


class TestApplyingTheGate(unittest.TestCase):
    """The decision is made on compound 25 and APPLIED to rows written long
    before it, so the predicate reads the same columns plip_batch writes."""

    def test_case_1_requires_both_columns(self):
        passes = gate_predicate(GATE_RULES[1]["columns"])
        self.assertTrue(passes({"metal_ca501": "1", "hbond_asn224": "1"}))
        self.assertFalse(passes({"metal_ca501": "1", "hbond_asn224": "0"}))
        self.assertFalse(passes({"metal_ca501": "0", "hbond_asn224": "1"}))

    def test_case_3_ignores_the_anchor_entirely(self):
        passes = gate_predicate(GATE_RULES[3]["columns"])
        self.assertTrue(passes({"metal_ca501": "1", "hbond_asn224": "0"}))

    def test_a_missing_column_fails_rather_than_passing(self):
        # An older plip.csv has no hbond_anchor_window. Treating absence as
        # satisfied would pass every pose through a gate that never ran.
        passes = gate_predicate(GATE_RULES[2]["columns"])
        self.assertFalse(passes({"metal_ca501": "1"}))

    def test_a_blank_cell_fails(self):
        passes = gate_predicate(GATE_RULES[1]["columns"])
        self.assertFalse(passes({"metal_ca501": "1", "hbond_asn224": ""}))


if __name__ == "__main__":
    unittest.main()
