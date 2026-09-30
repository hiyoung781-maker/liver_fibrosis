import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from plip_batch import (FIELDS, StaleResumeError, already_done,
                        iter_poses, row_from_record, thread_limits)

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


class TestIterPoses(unittest.TestCase):
    """PLIP runs per POSE, not per ligand: ~53,000 of them across two arms.
    Each row has to carry the ligand label and the pose rank, or a passing
    pose cannot be traced back to the molecule that owns it."""

    def test_yields_one_entry_per_pose_with_its_label_and_rank(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "gen_00042_out.pdbqt").write_text(TWO_POSES)
            entries = list(iter_poses(tmp))
            self.assertEqual([(e["label"], e["pose"]) for e in entries],
                             [("gen_00042", 1), ("gen_00042", 2)])

    def test_affinity_travels_with_the_pose(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "a_out.pdbqt").write_text(TWO_POSES)
            self.assertAlmostEqual(list(iter_poses(tmp))[0]["affinity"], -6.908)

    def test_an_empty_output_file_yields_nothing_rather_than_raising(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "empty_out.pdbqt").write_text("")
            self.assertEqual(list(iter_poses(tmp)), [])


class TestRowFromRecord(unittest.TestCase):
    """The four section 8.0 criteria come from validate_plip.crystal_verdict,
    so the batch and the crystal gate cannot drift apart. Raw counts travel
    too: the gate decision from compound 25 has not been made yet, and a
    criterion that was not recorded cannot be applied afterwards."""

    EMPTY = {k: [] for k in ("metal_complexes", "hbonds", "hydrophobic_contacts",
                             "pi_stacking", "salt_bridges", "water_bridges",
                             "halogen_bonds")}

    def _record(self, **kwargs):
        record = {k: list(v) for k, v in self.EMPTY.items()}
        record.update(kwargs)
        return record

    def test_a_pose_with_no_interactions_is_a_row_of_zeros_not_an_error(self):
        row = row_from_record({"label": "a", "pose": 1, "affinity": -6.9},
                              self.EMPTY)
        self.assertEqual(row["metal_ca501"], 0)
        self.assertEqual(row["n_hbonds"], 0)

    def test_the_crystal_criteria_match_validate_plip(self):
        record = self._record(
            metal_complexes=[{"reschain_lig": "B", "resnr_lig": "501",
                              "metal_type": "Ca", "location": "ligand"}],
            hbonds=[{"reschain": "B", "resnr": 224, "protisdon": "False",
                     "sidechain": "False"}],
            hydrophobic_contacts=[{"reschain": "A", "resnr": 178},
                                  {"reschain": "B", "resnr": 225}])
        row = row_from_record({"label": "a", "pose": 1, "affinity": -6.9}, record)
        self.assertEqual(row["metal_ca501"], 1)
        self.assertEqual(row["hbond_asn224"], 1)
        self.assertEqual(row["tyr178_contact"], 1)
        self.assertEqual(row["hydrophobic_pocket"], 1)

    def test_the_anchor_window_columns_travel_with_every_row(self):
        """§8.2 case 2 relaxes the gate to any beta1 backbone O in 223-226.
        The decision is made from compound 25, but it is APPLIED to rows
        written long before it -- so the column has to be there already."""
        record = self._record(hbonds=[
            {"reschain": "B", "resnr": 225, "protisdon": "False",
             "sidechain": "False"},
            {"reschain": "B", "resnr": 224, "protisdon": "False",
             "sidechain": "True"},
        ])
        row = row_from_record({"label": "a", "pose": 1, "affinity": -6.9}, record)
        self.assertEqual(row["hbond_asn224"], 0)          # side-chain only
        self.assertEqual(row["hbond_asn224_sidechain"], 1)
        self.assertEqual(row["hbond_anchor_window"], 1)   # Leu225 backbone
        self.assertEqual(row["hbond_bb_residues"], "225")

    def test_a_backbone_hbond_outside_the_window_is_recorded_but_does_not_count(self):
        record = self._record(hbonds=[
            {"reschain": "B", "resnr": 180, "protisdon": "False",
             "sidechain": "False"}])
        row = row_from_record({"label": "a", "pose": 1, "affinity": -6.9}, record)
        self.assertEqual(row["hbond_anchor_window"], 0)
        self.assertEqual(row["hbond_bb_residues"], "180")

    def test_raw_counts_are_recorded_for_criteria_not_yet_decided(self):
        record = self._record(
            hbonds=[{"reschain": "B", "resnr": 224, "protisdon": "False"},
                    {"reschain": "B", "resnr": 133, "protisdon": "True"}],
            pi_stacking=[{"reschain": "A", "resnr": 178}])
        row = row_from_record({"label": "a", "pose": 2, "affinity": -7.0}, record)
        self.assertEqual(row["n_hbonds"], 2)
        self.assertEqual(row["n_pi_stacking"], 1)

    def test_every_declared_field_is_present(self):
        row = row_from_record({"label": "a", "pose": 1, "affinity": None},
                              self.EMPTY)
        self.assertEqual(set(row), set(FIELDS))


class TestResume(unittest.TestCase):
    """53,000 poses is long enough that the run gets interrupted -- it already
    was once, every worker's plip killed with SIGINT. Starting over throws
    away hours of finished work, so a partial CSV is read back and its poses
    skipped."""

    HEADER = ",".join(FIELDS)

    def test_no_csv_means_nothing_is_done(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(already_done(Path(tmp) / "absent.csv"), set())

    def test_finished_poses_are_read_back_with_their_affinity(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "plip.csv"
            csv_path.write_text(
                self.HEADER + "\n"
                + "gen_00001,1,-6.9,ok" + ",0" * (len(FIELDS) - 4) + "\n"
                + "gen_00001,2,-6.8,ok" + ",0" * (len(FIELDS) - 4) + "\n")
            self.assertEqual(already_done(csv_path),
                             {("gen_00001", 1, "-6.9"), ("gen_00001", 2, "-6.8")})

    def test_a_header_only_csv_is_not_mistaken_for_finished_work(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "plip.csv"
            csv_path.write_text(self.HEADER + "\n")
            self.assertEqual(already_done(csv_path), set())

    def test_a_truncated_last_row_is_not_counted_as_done(self):
        # The interruption lands mid-write: the last line can be a fragment,
        # and counting it done would leave one pose permanently unmeasured.
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "plip.csv"
            csv_path.write_text(
                self.HEADER + "\n"
                + "gen_00001,1,-6.9,ok" + ",0" * (len(FIELDS) - 4) + "\n"
                + "gen_00002,1,-6.5")
            self.assertEqual(already_done(csv_path), {("gen_00001", 1, "-6.9")})


class TestThreadLimits(unittest.TestCase):
    """64 worker processes each importing numpy will each start a thread pool
    sized to the whole node, so the node runs 64x64 threads fighting for 64
    cores. The user's own node template sets OMP_NUM_THREADS=1 for the same
    reason; the workers here set it for themselves so a plain command line
    does not have to remember."""

    def test_every_common_thread_variable_is_pinned_to_one(self):
        limits = thread_limits()
        self.assertEqual(set(limits.values()), {"1"})
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertIn(name, limits)


class TestResumeRefusesAnotherRunsCsv(unittest.TestCase):
    """A re-docking with corrected protomers produced new poses under the same
    labels and pose numbers. --resume matched on (label, pose) alone, declared
    "34485 of 34183 poses already recorded" -- more than exist -- and skipped
    almost everything, leaving a CSV of the PREVIOUS run's interactions that
    would have flowed straight into the funnel.

    The affinity is part of the key, so a pose whose docking changed is
    redone, and a count above the current one is refused outright."""

    HEADER = ",".join(FIELDS)

    def _csv(self, tmp, rows):
        path = Path(tmp) / "plip.csv"
        path.write_text(self.HEADER + "\n" + "".join(
            f"{l},{p},{a},ok" + ",0" * (len(FIELDS) - 4) + "\n"
            for l, p, a in rows))
        return path

    def test_a_pose_whose_affinity_changed_is_not_counted_done(self):
        with TemporaryDirectory() as tmp:
            path = self._csv(tmp, [("gen_00001", 1, "-6.900")])
            done = already_done(path)
            self.assertIn(("gen_00001", 1, "-6.900"), done)
            self.assertNotIn(("gen_00001", 1, "-7.100"), done)

    def test_more_recorded_than_present_is_refused(self):
        with TemporaryDirectory() as tmp:
            path = self._csv(tmp, [("a", 1, "-6.9"), ("a", 2, "-6.8"),
                                   ("b", 1, "-7.0")])
            with self.assertRaises(StaleResumeError):
                already_done(path, n_current=2)

    def test_an_equal_or_smaller_count_resumes_normally(self):
        with TemporaryDirectory() as tmp:
            path = self._csv(tmp, [("a", 1, "-6.9"), ("a", 2, "-6.8")])
            self.assertEqual(len(already_done(path, n_current=5)), 2)


if __name__ == "__main__":
    unittest.main()
