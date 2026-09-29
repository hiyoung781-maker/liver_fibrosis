import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from plip_batch import FIELDS, iter_poses, row_from_record

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
            hbonds=[{"reschain": "B", "resnr": 224, "protisdon": "False"}],
            hydrophobic_contacts=[{"reschain": "A", "resnr": 178},
                                  {"reschain": "B", "resnr": 225}])
        row = row_from_record({"label": "a", "pose": 1, "affinity": -6.9}, record)
        self.assertEqual(row["metal_ca501"], 1)
        self.assertEqual(row["hbond_asn224"], 1)
        self.assertEqual(row["tyr178_contact"], 1)
        self.assertEqual(row["hydrophobic_pocket"], 1)

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


if __name__ == "__main__":
    unittest.main()
