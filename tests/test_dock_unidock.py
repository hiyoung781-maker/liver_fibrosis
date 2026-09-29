import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from dock_unidock import (DEFAULTS, box_from_ligand, build_command,
                          parse_unidock_pdbqt)

REF = "docking/ligand_ref.sdf"


class TestDefaultsMatchTheValidatedRun(unittest.TestCase):
    """smina was run with --autobox_add 6 --exhaustiveness 16 --num_modes 20
    --seed 42, and that is the run §8.5(a) validated at 0.63 A. Uni-Dock has to
    reproduce it or the validation stops applying."""

    def test_pinned_values(self):
        self.assertEqual(DEFAULTS["autobox_add"], 6.0)
        self.assertEqual(DEFAULTS["exhaustiveness"], 16)
        self.assertEqual(DEFAULTS["num_modes"], 20)
        self.assertEqual(DEFAULTS["seed"], 42)

    def test_scoring_is_vina_not_ad4(self):
        """Uni-Dock also offers ad4 and vinardo. Only `vina` is the function the
        recorded redocking result was obtained with."""
        self.assertEqual(DEFAULTS["scoring"], "vina")


class TestBoxFromLigand(unittest.TestCase):
    def test_reproduces_sminas_autobox_rule(self):
        """smina spans [min - pad, max + pad] per axis, so size = extent + 2*pad.
        Computed independently here from the ligand coordinates."""
        pad = 6.0
        xyz = Chem.MolFromMolFile(REF, removeHs=False).GetConformer().GetPositions()
        lo, hi = xyz.min(axis=0) - pad, xyz.max(axis=0) + pad
        box = box_from_ligand(REF, pad)
        for i, axis in enumerate("xyz"):
            self.assertAlmostEqual(box[f"center_{axis}"], (lo[i] + hi[i]) / 2, places=6)
            self.assertAlmostEqual(box[f"size_{axis}"], hi[i] - lo[i], places=6)

    def test_size_is_extent_plus_twice_the_padding(self):
        xyz = Chem.MolFromMolFile(REF, removeHs=False).GetConformer().GetPositions()
        extent = xyz.max(axis=0) - xyz.min(axis=0)
        box = box_from_ligand(REF, 6.0)
        for i, axis in enumerate("xyz"):
            self.assertAlmostEqual(box[f"size_{axis}"], extent[i] + 12.0, places=6)

    def test_matches_the_recorded_box(self):
        """Pins the six numbers actually used, so a change to the reference ligand
        or the padding cannot move the search space silently."""
        box = box_from_ligand(REF, 6.0)
        self.assertAlmostEqual(box["center_x"], 2.300, places=2)
        self.assertAlmostEqual(box["center_y"], 114.144, places=2)
        self.assertAlmostEqual(box["center_z"], 40.194, places=2)
        self.assertAlmostEqual(box["size_x"], 16.957, places=2)
        self.assertAlmostEqual(box["size_y"], 18.471, places=2)
        self.assertAlmostEqual(box["size_z"], 21.167, places=2)

    def test_the_box_contains_both_geometry_filter_anchors(self):
        """If Ca501 or Asn224's oxygen sat outside the search space, no pose could
        ever satisfy §8.5b - the filter would be measuring an unreachable target."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from pose_geometry import anchor_atoms

        box = box_from_ligand(REF, 6.0)
        lo = np.array([box["center_x"] - box["size_x"] / 2,
                       box["center_y"] - box["size_y"] / 2,
                       box["center_z"] - box["size_z"] / 2])
        hi = lo + np.array([box["size_x"], box["size_y"], box["size_z"]])
        for label, xyz in anchor_atoms("docking/receptor.pdb").items():
            with self.subTest(anchor=label):
                self.assertTrue(np.all(xyz >= lo) and np.all(xyz <= hi),
                                f"{label} at {xyz} is outside {lo}..{hi}")

    def test_unreadable_ligand_raises(self):
        with self.assertRaises(ValueError):
            box_from_ligand("docking/receptor.pdb")


class TestBuildCommand(unittest.TestCase):
    def setUp(self):
        self.box = box_from_ligand(REF, 6.0)

    def test_passes_all_six_box_numbers(self):
        cmd = build_command("r.pdbqt", "l.txt", "out", self.box)
        for key in ("center_x", "center_y", "center_z",
                    "size_x", "size_y", "size_z"):
            self.assertIn(f"--{key}", cmd)

    def test_uses_ligand_index_not_a_7767_argument_list(self):
        """Uni-Dock also accepts ligands as positional arguments to --gpu_batch;
        7,767 paths would risk the argv length limit, so the index file is used."""
        cmd = build_command("r.pdbqt", "l.txt", "out", self.box)
        self.assertIn("--ligand_index", cmd)
        self.assertEqual(cmd[cmd.index("--ligand_index") + 1], "l.txt")

    def test_pins_scoring_seed_and_search_effort(self):
        cmd = build_command("r.pdbqt", "l.txt", "out", self.box)
        for flag, value in (("--scoring", "vina"), ("--seed", "42"),
                            ("--exhaustiveness", "16"), ("--num_modes", "20")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)

    def test_search_mode_is_absent_unless_asked_for(self):
        """--search_mode overrides exhaustiveness with a preset. §8.5b needs many
        poses because the scoring function has no metal term, so a cheaper preset
        is not a free trade - it must be requested explicitly."""
        self.assertNotIn("--search_mode", build_command("r", "l", "o", self.box))
        self.assertIn("--search_mode",
                      build_command("r", "l", "o", self.box, search_mode="fast"))


UNIDOCK_OUT = """\
MODEL 1
REMARK VINA RESULT:      -6.908      0.000      0.000
REMARK SMILES O=C(N[C@@H](Cc1ccccc1)C(=O)[O-])c1ccc(Cl)cc1Cl
ROOT
ATOM      1  C   UNL     1       1.474 114.913  40.627 -0.15 -0.07    +0.253 C
ENDROOT
TORSDOF 5
ENDMDL
MODEL 2
REMARK VINA RESULT:      -6.828      1.213      1.901
ROOT
ATOM      1  C   UNL     1       2.474 114.913  40.627 -0.15 -0.07    +0.253 C
ENDROOT
TORSDOF 5
ENDMDL
"""


class TestParseUnidockPdbqt(unittest.TestCase):
    """The redocking gate was written against AutoDock-GPU's .dlg. Reverting to
    Uni-Dock (spec 7.4's pre-registered consequence) means the same gate has to
    read Uni-Dock's output, which records each pose's energy as
    `REMARK VINA RESULT: <kcal/mol> <lb> <ub>` inside a MODEL/ENDMDL block.
    The return shape matches dock_autodock_gpu.parse_dlg so validate_redock
    does not care which engine produced the poses."""

    def test_every_model_becomes_a_pose(self):
        self.assertEqual(len(parse_unidock_pdbqt(UNIDOCK_OUT)), 2)

    def test_affinity_is_the_first_number_of_the_vina_result_remark(self):
        poses = parse_unidock_pdbqt(UNIDOCK_OUT)
        self.assertAlmostEqual(poses[0]["affinity"], -6.908)
        self.assertAlmostEqual(poses[1]["affinity"], -6.828)

    def test_ranks_count_from_one_in_file_order(self):
        self.assertEqual([p["rank"] for p in parse_unidock_pdbqt(UNIDOCK_OUT)], [1, 2])

    def test_block_carries_the_atom_records_and_is_self_contained(self):
        block = parse_unidock_pdbqt(UNIDOCK_OUT)[0]["pdbqt_block"]
        self.assertTrue(block.startswith("MODEL"))
        self.assertTrue(block.rstrip().endswith("ENDMDL"))
        self.assertIn("ATOM      1  C   UNL", block)

    def test_a_pose_without_an_energy_keeps_the_pose(self):
        # Same rule as parse_dlg: a v1 regression put score extraction in the
        # structure-parsing try and took the pose count to zero. An absent
        # score must cost the score, never the pose.
        text = UNIDOCK_OUT.replace(
            "REMARK VINA RESULT:      -6.828      1.213      1.901\n", "")
        poses = parse_unidock_pdbqt(text)
        self.assertEqual(len(poses), 2)
        self.assertIsNone(poses[1]["affinity"])

    def test_empty_input_returns_no_poses_rather_than_raising(self):
        self.assertEqual(parse_unidock_pdbqt(""), [])



if __name__ == "__main__":
    unittest.main()
