import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from funnel import PLN1474_CUTOFF, ligand_rows, stratify, summarize

GEOMETRY_FIELDS = ["label", "pose", "affinity", "ca_dist", "ca_dist_far",
                   "donor_dist", "n_carboxylate_o", "n_donors", "passes"]
PLIP_FIELDS = ["label", "pose", "affinity", "status", "metal_ca501",
               "hbond_asn224", "tyr178_contact", "hydrophobic_pocket"]


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def geom(label, pose, affinity, passes, ca=2.6, donor=2.9):
    return {"label": label, "pose": pose, "affinity": f"{affinity:.3f}",
            "ca_dist": f"{ca:.3f}", "ca_dist_far": "4.500",
            "donor_dist": f"{donor:.3f}", "n_carboxylate_o": 2, "n_donors": 3,
            "passes": int(passes)}


def plip(label, pose, affinity=-7.0, metal=1, hbond=1):
    return {"label": label, "pose": pose, "affinity": f"{affinity:.3f}",
            "status": "ok", "metal_ca501": metal, "hbond_asn224": hbond,
            "tyr178_contact": 0, "hydrophobic_pocket": 1}


class TestLigandRows(unittest.TestCase):
    """Section 8.5b passes a LIGAND if ANY of its poses makes both contacts,
    and section 8.7 then filters on the best PASSING pose's affinity -- not
    on the best pose overall. The panel run makes the difference concrete:
    CHEMBL4649232 holds the panel's best affinity (-8.012) with no passing
    pose at all, so scoring it on the best pose overall would promote a
    compound whose binding mode the filter rejected."""

    def _rows(self, tmp, geometry, plip_rows, cutoff=PLN1474_CUTOFF):
        g = Path(tmp) / "geometry.csv"
        p = Path(tmp) / "plip.csv"
        write_csv(g, GEOMETRY_FIELDS, geometry)
        write_csv(p, PLIP_FIELDS, plip_rows)
        return {r["label"]: r for r in ligand_rows(str(g), str(p), cutoff)}

    def test_a_ligand_passes_geometry_if_any_pose_does(self):
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp,
                              [geom("a", 1, -6.0, False), geom("a", 2, -7.5, True)],
                              [plip("a", 1), plip("a", 2)])
            self.assertTrue(rows["a"]["geometry_pass"])

    def test_affinity_comes_from_the_best_PASSING_pose(self):
        # Pose 1 scores better but fails geometry; using it would let the
        # filter pass a ligand on a pose the filter rejected.
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp,
                              [geom("a", 1, -9.0, False), geom("a", 2, -7.0, True)],
                              [plip("a", 1), plip("a", 2)])
            self.assertAlmostEqual(rows["a"]["best_passing_affinity"], -7.0)
            self.assertAlmostEqual(rows["a"]["best_affinity"], -9.0)

    def test_a_ligand_with_no_passing_pose_has_no_passing_affinity(self):
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp, [geom("a", 1, -9.0, False)], [plip("a", 1)])
            self.assertFalse(rows["a"]["geometry_pass"])
            self.assertIsNone(rows["a"]["best_passing_affinity"])
            self.assertFalse(rows["a"]["affinity_pass"])

    def test_the_affinity_cutoff_is_strict(self):
        # The panel gives PLN-1474's best passing pose as the threshold; a
        # candidate that merely ties it has not surpassed it.
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp,
                              [geom("tie", 1, PLN1474_CUTOFF, True),
                               geom("better", 1, PLN1474_CUTOFF - 0.01, True)],
                              [plip("tie", 1), plip("better", 1)])
            self.assertFalse(rows["tie"]["affinity_pass"])
            self.assertTrue(rows["better"]["affinity_pass"])

    def test_plip_is_read_from_the_passing_poses_only(self):
        # A failing pose's interactions say nothing about the binding mode
        # the campaign selected.
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp,
                              [geom("a", 1, -8.0, False), geom("a", 2, -7.5, True)],
                              [plip("a", 1, metal=1, hbond=1),
                               plip("a", 2, metal=0, hbond=0)])
            self.assertFalse(rows["a"]["plip_metal_any_passing"])
            self.assertFalse(rows["a"]["plip_hbond_any_passing"])

    def test_a_pose_missing_from_plip_does_not_drop_the_ligand(self):
        # PLIP can fail on one pose; that must cost the PLIP columns for that
        # pose, never the ligand, or the funnel loses its denominator.
        with TemporaryDirectory() as tmp:
            rows = self._rows(tmp,
                              [geom("a", 1, -7.5, True), geom("a", 2, -7.4, True)],
                              [plip("a", 1)])
            self.assertIn("a", rows)
            self.assertEqual(rows["a"]["n_poses"], 2)


class TestSummary(unittest.TestCase):
    def _rows(self):
        return [
            {"label": "a", "n_poses": 1, "geometry_pass": True,
             "best_passing_affinity": -7.5, "affinity_pass": True,
             "plip_metal_any_passing": True, "plip_hbond_any_passing": True},
            {"label": "b", "n_poses": 9, "geometry_pass": True,
             "best_passing_affinity": -7.0, "affinity_pass": False,
             "plip_metal_any_passing": True, "plip_hbond_any_passing": False},
            {"label": "c", "n_poses": 3, "geometry_pass": False,
             "best_passing_affinity": None, "affinity_pass": False,
             "plip_metal_any_passing": False, "plip_hbond_any_passing": False},
        ]

    def test_the_funnel_counts_narrow_monotonically(self):
        s = summarize(self._rows())
        self.assertEqual(s["ligands"], 3)
        self.assertEqual(s["geometry_pass"], 2)
        self.assertEqual(s["affinity_pass"], 1)
        self.assertLessEqual(s["affinity_pass"], s["geometry_pass"])

    def test_stratification_groups_by_pose_count(self):
        """TL-A-prime got a median of 3 poses per ligand and TL-C 6, and a
        one-pose ligand gets one chance at the geometry gate while a
        fifteen-pose ligand gets fifteen. Comparing the arms' raw pass rates
        compares pose counts, so the rate is also reported per bucket."""
        buckets = stratify(self._rows())
        self.assertEqual(buckets["1"]["ligands"], 1)
        self.assertEqual(buckets["1"]["geometry_pass"], 1)
        self.assertEqual(buckets["2-4"]["ligands"], 1)
        self.assertEqual(buckets["2-4"]["geometry_pass"], 0)
        self.assertEqual(buckets["8-15"]["ligands"], 1)

    def test_an_empty_input_does_not_divide_by_zero(self):
        self.assertEqual(summarize([])["ligands"], 0)
        self.assertEqual(stratify([]), {})


if __name__ == "__main__":
    unittest.main()
