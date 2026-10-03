import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from dock_unidock import (AFFINITY_FLOOR, DEFAULTS, box_from_ligand, build_command,
                          filter_done, parse_unidock_pdbqt, split_index)

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


SMINA_OUT = """\
MODEL 1
REMARK minimizedAffinity -7.0807085
REMARK minimizedRMSD -1
REMARK SMILES O=C(N[C@@H](Cc1ccccc1)C(=O)[O-])c1ccc(Cl)cc1Cl
ROOT
ATOM      1  C   UNL     1       1.474 114.913  40.627  1.00  0.00    +0.253 C
ENDROOT
TORSDOF 5
ENDMDL
MODEL 2
REMARK minimizedAffinity -6.9
REMARK minimizedRMSD -1
ROOT
ATOM      1  C   UNL     1       2.474 114.913  40.627  1.00  0.00    +0.253 C
ENDROOT
TORSDOF 5
ENDMDL
"""


class TestParseSminaPdbqt(unittest.TestCase):
    """The alphaVbeta6 side had to move from Uni-Dock to smina when the cluster
    GPU allocation ended. smina records the SAME Vina 1.1.2 score as
    `REMARK minimizedAffinity <kcal/mol>` -- one number, a different name -- so
    the parser accepts both spellings and every downstream reader stays engine-
    agnostic. A parser that silently returned affinity None here would not
    raise: it would hand selectivity.py a receptor with no scores, and the
    percentile axis would be computed from an empty side."""

    def test_smina_affinity_is_read(self):
        poses = parse_unidock_pdbqt(SMINA_OUT)
        self.assertAlmostEqual(poses[0]["affinity"], -7.0807085)
        self.assertAlmostEqual(poses[1]["affinity"], -6.9)

    def test_smina_poses_are_counted_and_ranked_like_unidock(self):
        poses = parse_unidock_pdbqt(SMINA_OUT)
        self.assertEqual([p["rank"] for p in poses], [1, 2])

    def test_the_minimized_rmsd_remark_is_not_mistaken_for_an_affinity(self):
        # `REMARK minimizedRMSD -1` sits right after the affinity and also
        # carries a negative number. Matching it would overwrite every score
        # with -1 and leave the file looking parsed.
        poses = parse_unidock_pdbqt(SMINA_OUT)
        self.assertAlmostEqual(poses[0]["affinity"], -7.0807085)

    def test_unidock_output_still_parses_unchanged(self):
        poses = parse_unidock_pdbqt(UNIDOCK_OUT)
        self.assertAlmostEqual(poses[0]["affinity"], -6.908)


BROKEN_OUT = """\
MODEL 1
REMARK VINA RESULT:    -178.439      0.000      0.000
REMARK SMILES O=C([O-])c1ccccc1
ROOT
ATOM      1  C   UNL     1       1.474 114.913  40.627  1.00  0.00    +0.253 C
ENDROOT
TORSDOF 5
ENDMDL
MODEL 2
REMARK VINA RESULT:      -7.200      1.213      1.901
ROOT
ATOM      1  C   UNL     1       2.474 114.913  40.627  1.00  0.00    +0.253 C
ENDROOT
TORSDOF 5
ENDMDL
"""


class TestAffinityFloor(unittest.TestCase):
    """126 ligands across the two arms carry Meeko's macrocycle glue atoms
    (`CG0`/`G0`), pseudo-atoms inserted to open a ring that Vina has no
    parameters for, and 251 of their poses score between -21.9 and -219.017
    while the strongest physical pose in the campaign is -10.340. Read as
    scores, those poses pass any affinity filter trivially. smina confirms the
    mechanism by refusing to parse the same ligands outright."""

    def test_a_score_below_the_floor_is_missing_not_excellent(self):
        poses = parse_unidock_pdbqt(BROKEN_OUT)
        self.assertIsNone(poses[0]["affinity"])

    def test_it_costs_the_score_and_not_the_pose(self):
        # Same rule an absent energy remark follows: dropping the pose would
        # shrink the denominator of every rate computed downstream.
        poses = parse_unidock_pdbqt(BROKEN_OUT)
        self.assertEqual(len(poses), 2)
        self.assertIn("ATOM      1  C   UNL", poses[0]["pdbqt_block"])

    def test_a_physical_score_on_another_pose_survives(self):
        poses = parse_unidock_pdbqt(BROKEN_OUT)
        self.assertAlmostEqual(poses[1]["affinity"], -7.2)

    def test_both_measured_boundaries_fall_on_the_right_side(self):
        # The two distributions do not touch: -10.340 is the strongest physical
        # pose in the campaign and -21.913 the weakest artifact. Any floor
        # between them behaves identically, which is why -15.0 is not a tuned
        # number -- these two assertions are what it has to get right.
        kept = BROKEN_OUT.replace("-178.439", "-10.340")
        self.assertAlmostEqual(parse_unidock_pdbqt(kept)[0]["affinity"], -10.340)
        dropped = BROKEN_OUT.replace("-178.439", "-21.913")
        self.assertIsNone(parse_unidock_pdbqt(dropped)[0]["affinity"])

    def test_the_floor_is_stated_once_and_imported_by_the_csv_reader(self):
        # funnel.py reads affinities out of geometry.csv, the one path that does
        # not go through this parser, so a second literal there could drift.
        import funnel
        self.assertIs(funnel.AFFINITY_FLOOR, AFFINITY_FLOOR)

    def test_a_positive_affinity_is_kept(self):
        # Vina reports positive energies for clashing poses; those are physical
        # readings and they fail every `< cutoff` filter on their own.
        text = BROKEN_OUT.replace("-178.439", "8.354")
        self.assertAlmostEqual(parse_unidock_pdbqt(text)[0]["affinity"], 8.354)


RE_DOCKED_OUT = """\
MODEL 1
REMARK minimizedAffinity -7.29541302
REMARK minimizedRMSD -1
REMARK VINA RESULT:    -7.046      0.000      0.000
REMARK INTER + INTRA:          -8.667
REMARK SMILES O=C([O-])c1ccccc1
ROOT
ATOM      1  C   UNL     1       1.474 114.913  40.627  1.00  0.00    +0.253 C
ENDROOT
TORSDOF 5
ENDMDL
"""


class TestTwoEnergyRemarksInOneBlock(unittest.TestCase):
    """A ligand re-docked from another run's output pose carries that run's
    energy remarks, because an engine copies its input's remarks into its own
    output. The block then holds two energies: this engine's first, then the
    inherited one. Assigning on every match let the inherited one win, so the
    alphaVbeta6 side reported alphaVbeta1's affinities and every selectivity
    value came out exactly +0.000 -- which is how it was caught."""

    def test_the_first_energy_remark_wins(self):
        poses = parse_unidock_pdbqt(RE_DOCKED_OUT)
        self.assertAlmostEqual(poses[0]["affinity"], -7.29541302)

    def test_the_inherited_energy_is_not_what_is_reported(self):
        poses = parse_unidock_pdbqt(RE_DOCKED_OUT)
        self.assertNotAlmostEqual(poses[0]["affinity"], -7.046)

    def test_a_block_with_only_an_inherited_remark_still_reads(self):
        # Uni-Dock's own output has VINA RESULT first and nothing else; the rule
        # must not make the ordinary case depend on a second remark existing.
        text = RE_DOCKED_OUT.replace(
            "REMARK minimizedAffinity -7.29541302\nREMARK minimizedRMSD -1\n", "")
        self.assertAlmostEqual(parse_unidock_pdbqt(text)[0]["affinity"], -7.046)


class TestFilterDone(unittest.TestCase):
    """The resume path. Uni-Dock has no --resume, and the first pass is 11 GPU
    hours / 62 CPU hours, so an interruption at hour ten would otherwise restart
    from zero. Outputs are named after their ligand, which makes resuming an
    index filter instead of engine state."""

    @staticmethod
    def _index(tmp, labels):
        ligands = Path(tmp) / "pdbqt"
        ligands.mkdir(exist_ok=True)
        paths = []
        for label in labels:
            path = ligands / f"{label}.pdbqt"
            path.write_text("REMARK stub\n")
            paths.append(str(path))
        index = Path(tmp) / "ligands.txt"
        index.write_text("\n".join(paths) + "\n")
        return str(index)

    @staticmethod
    def _outputs(tmp, labels_with_mtime):
        """Write `<label>_out.pdbqt` files with explicit mtimes.

        mtime is set rather than relied upon: the files are created within the
        same clock tick, so without this the "most recent" output would be
        whichever one the filesystem happened to stamp last and the re-queue
        test would pass or fail at random.
        """
        out = Path(tmp) / "poses"
        out.mkdir(exist_ok=True)
        for label, mtime in labels_with_mtime:
            path = out / f"{label}_out.pdbqt"
            path.write_text("MODEL 1\nENDMDL\n")
            os.utime(path, (mtime, mtime))
        return str(out)

    def test_finished_ligands_are_dropped_and_order_is_preserved(self):
        with TemporaryDirectory() as tmp:
            index = self._index(tmp, ["a", "b", "c", "d"])
            # 'd' is newest, so it is the one re-queued; a and b are genuinely done.
            out = self._outputs(tmp, [("a", 1000), ("b", 2000), ("d", 3000)])
            state = filter_done(index, out)
        self.assertEqual([Path(p).stem for p in state["remaining"]], ["c", "d"])
        self.assertEqual(state["done"], ["a", "b"])
        self.assertEqual(state["total"], 4)

    def test_the_most_recently_modified_output_is_requeued(self):
        """A process killed mid-write truncates exactly one file, and a truncated
        PDBQT is indistinguishable from a ligand whose search found few poses --
        which §4.1's retention curve and §4.6's criterion (c) both read. Re-docking
        it costs ~8 s and Uni-Dock overwrites by name."""
        with TemporaryDirectory() as tmp:
            index = self._index(tmp, ["a", "b", "c"])
            out = self._outputs(tmp, [("a", 1000), ("b", 5000), ("c", 2000)])
            state = filter_done(index, out)
        self.assertEqual(state["requeued"], "b")
        self.assertIn("b", [Path(p).stem for p in state["remaining"]])
        self.assertNotIn("b", state["done"])

    def test_a_fresh_or_missing_out_dir_returns_the_whole_index(self):
        """--resume on a first run has to be a no-op, so the flag can always be
        passed rather than remembered only after a crash."""
        with TemporaryDirectory() as tmp:
            index = self._index(tmp, ["a", "b"])
            missing = str(Path(tmp) / "never_created")
            state = filter_done(index, missing)
            self.assertEqual([Path(p).stem for p in state["remaining"]], ["a", "b"])
            self.assertEqual(state["done"], [])
            self.assertIsNone(state["requeued"])

            empty = self._outputs(tmp, [])
            state = filter_done(index, empty)
            self.assertEqual([Path(p).stem for p in state["remaining"]], ["a", "b"])
            self.assertIsNone(state["requeued"])

    def test_an_exact_label_match_not_a_prefix_match(self):
        """`gen_01` and `gen_010` are different ligands. A prefix test would
        retire both on one output and silently shrink the library."""
        with TemporaryDirectory() as tmp:
            index = self._index(tmp, ["gen_01", "gen_010", "gen_0100"])
            # Two outputs so the newest re-queue does not mask the matching rule.
            out = self._outputs(tmp, [("gen_01", 1000), ("gen_0100", 3000)])
            state = filter_done(index, out)
        self.assertEqual(state["done"], ["gen_01"])
        self.assertEqual([Path(p).stem for p in state["remaining"]],
                         ["gen_010", "gen_0100"])

    def test_a_fully_docked_library_leaves_only_the_requeued_ligand(self):
        """Not an error: a finished run that gets resumed should re-dock the one
        possibly-truncated output and stop, not raise."""
        with TemporaryDirectory() as tmp:
            index = self._index(tmp, ["a", "b"])
            out = self._outputs(tmp, [("a", 1000), ("b", 2000)])
            state = filter_done(index, out)
        self.assertEqual([Path(p).stem for p in state["remaining"]], ["b"])
        self.assertEqual(state["done"], ["a"])

    def test_the_filtered_index_shards_round_robin_over_what_is_left(self):
        """§7's open item 4: the 2-GPU path re-shards the REMAINING ligands, so a
        restart keeps both devices fed instead of inheriting the original split."""
        with TemporaryDirectory() as tmp:
            index = self._index(tmp, ["a", "b", "c", "d", "e", "f"])
            out = self._outputs(tmp, [("a", 1000), ("b", 2000)])
            state = filter_done(index, out)
            resume = Path(tmp) / "resume_index.txt"
            resume.write_text("\n".join(state["remaining"]) + "\n")
            shards = split_index(str(resume), 2, str(Path(tmp) / "shards"))

            # INSIDE the with-block: split_index's output is files, and reading
            # them after TemporaryDirectory cleans up raises FileNotFoundError.
            self.assertEqual(len(shards), 2)
            per_shard = [[Path(line).stem for line in Path(s).read_text().split()]
                         for s in shards]

        # a is done; b is re-queued (newest output). filter_done preserves the
        # ORIGINAL index order rather than appending the re-queued ligand, so
        # b c d e f remain -> round-robin b,d,f | c,e.
        self.assertEqual(per_shard, [["b", "d", "f"], ["c", "e"]])


if __name__ == "__main__":
    unittest.main()
