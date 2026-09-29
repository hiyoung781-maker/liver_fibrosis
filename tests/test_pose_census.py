import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pose_census import census, summarize

TWO_POSES = """\
MODEL 1
REMARK VINA RESULT:      -6.908      0.000      0.000
ATOM      1  C   UNL     1       1.474 114.913  40.627 -0.15 -0.07    +0.253 C
ENDMDL
MODEL 2
REMARK VINA RESULT:      -6.828      1.213      1.901
ATOM      1  C   UNL     1       2.474 114.913  40.627 -0.15 -0.07    +0.253 C
ENDMDL
"""

ONE_POSE = """\
MODEL 1
REMARK VINA RESULT:      -7.100      0.000      0.000
ATOM      1  C   UNL     1       1.474 114.913  40.627 -0.15 -0.07    +0.253 C
ENDMDL
"""


class TestCensus(unittest.TestCase):
    """Uni-Dock returned 2 and 4 poses for the control where AutoDock-GPU
    returned 20. Section 8.5b's whole strategy is to generate many poses and
    select on geometry, because the scoring function has no metal term, so
    how many poses each ligand actually got is a number the lead report has
    to carry -- not an implementation detail.

    A file existing is not a pose. Uni-Dock exits 0 having written a file
    with no MODEL block, and counting files would report that as success."""

    def _dir(self, tmp, contents):
        for name, text in contents.items():
            (Path(tmp) / f"{name}_out.pdbqt").write_text(text)
        return tmp

    def test_counts_poses_per_ligand_not_files(self):
        with TemporaryDirectory() as tmp:
            rows = census(self._dir(tmp, {"a": TWO_POSES, "b": ONE_POSE}))
            self.assertEqual({r["label"]: r["poses"] for r in rows},
                             {"a": 2, "b": 1})

    def test_an_empty_file_is_reported_as_zero_not_skipped(self):
        with TemporaryDirectory() as tmp:
            rows = census(self._dir(tmp, {"a": TWO_POSES, "empty": ""}))
            self.assertEqual({r["label"]: r["poses"] for r in rows}["empty"], 0)

    def test_best_affinity_is_the_lowest_in_the_file(self):
        with TemporaryDirectory() as tmp:
            rows = census(self._dir(tmp, {"a": TWO_POSES}))
            self.assertAlmostEqual(rows[0]["best_affinity"], -6.908)

    def test_summary_names_the_ligands_with_no_poses(self):
        with TemporaryDirectory() as tmp:
            rows = census(self._dir(tmp, {"a": TWO_POSES, "empty": ""}))
            summary = summarize(rows)
            self.assertEqual(summary["ligands"], 2)
            self.assertEqual(summary["no_poses"], ["empty"])

    def test_summary_reports_the_distribution(self):
        with TemporaryDirectory() as tmp:
            rows = census(self._dir(tmp, {"a": TWO_POSES, "b": ONE_POSE,
                                          "c": TWO_POSES}))
            summary = summarize(rows)
            self.assertEqual(summary["min_poses"], 1)
            self.assertEqual(summary["max_poses"], 2)
            self.assertEqual(summary["median_poses"], 2)
            self.assertEqual(summary["histogram"], {1: 1, 2: 2})

    def test_summary_of_nothing_does_not_divide_by_zero(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(summarize(census(tmp))["ligands"], 0)


if __name__ == "__main__":
    unittest.main()
