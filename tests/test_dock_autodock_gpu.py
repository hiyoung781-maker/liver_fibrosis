import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dock_autodock_gpu import (NRUN, build_command, build_filelist_command,
                               find_fld, parse_dlg, preflight, run_batch,
                               write_filelist)

DLG = """\
DOCKED: MODEL        1
DOCKED: USER    Estimated Free Energy of Binding    =   -6.75 kcal/mol
DOCKED: ATOM      1  C   LIG d   1      12.345  23.456  34.567  1.00  0.00     0.012 C
DOCKED: ENDMDL
DOCKED: MODEL        2
DOCKED: USER    Estimated Free Energy of Binding    =   -6.31 kcal/mol
DOCKED: ATOM      1  C   LIG d   1      12.000  23.000  34.000  1.00  0.00     0.012 C
DOCKED: ENDMDL
"""


class TestParseDlg(unittest.TestCase):
    def test_extracts_every_pose_with_its_energy(self):
        poses = parse_dlg(DLG)
        self.assertEqual([p["rank"] for p in poses], [1, 2])
        self.assertAlmostEqual(poses[0]["affinity"], -6.75)
        self.assertAlmostEqual(poses[1]["affinity"], -6.31)

    def test_strips_the_docked_prefix_from_coordinates(self):
        """DOCKED: 접두사가 남으면 어떤 PDBQT 파서도 읽지 못한다."""
        poses = parse_dlg(DLG)
        self.assertTrue(poses[0]["pdbqt_block"].startswith("MODEL"))
        self.assertNotIn("DOCKED:", poses[0]["pdbqt_block"])

    def test_missing_energy_costs_the_score_not_the_pose(self):
        """v1의 회귀: 에너지 누락을 구조 파싱과 같은 try에 넣어 포즈 0개를
        반환했다. 에너지가 없으면 affinity를 None으로 두되 포즈는 남긴다."""
        text = DLG.replace(
            "DOCKED: USER    Estimated Free Energy of Binding    =   -6.75 kcal/mol\n",
            "")
        poses = parse_dlg(text)
        self.assertEqual(len(poses), 2)
        self.assertIsNone(poses[0]["affinity"])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(parse_dlg(""), [])


class TestBuildCommand(unittest.TestCase):
    def test_uses_the_map_field_file_and_pins_the_seed(self):
        cmd = build_command(fld="docking/v2/maps/receptor.maps.fld",
                            ligand="lig.pdbqt", out_prefix="out/lig", seed=42)
        self.assertIn("--ffile", cmd)
        self.assertIn("docking/v2/maps/receptor.maps.fld", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("42", cmd)

    def test_requests_enough_poses_for_a_geometry_filter(self):
        """게이트는 점수가 아니라 상호작용으로 포즈를 고른다. 소수의
        에너지 상위 포즈만 받으면 그 설계가 무너진다."""
        cmd = build_command(fld="f.fld", ligand="l.pdbqt",
                            out_prefix="o", seed=42)
        self.assertIn("--nrun", cmd)
        self.assertGreaterEqual(int(cmd[cmd.index("--nrun") + 1]), 20)

    def test_nrun_constant_is_not_lowered(self):
        """NRUN=20은 브리프에서 못박은 값. 코드 어딘가가 이 상수를 낮추면
        위 게이트 테스트는 통과하지만 실제 배치 호출은 조용히 20 미만을 쓴다."""
        self.assertGreaterEqual(NRUN, 20)


class TestFindFld(unittest.TestCase):
    def test_locates_the_single_maps_fld_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "receptor.maps.fld").write_text("x")
            self.assertEqual(find_fld(tmp),
                             str(Path(tmp) / "receptor.maps.fld"))

    def test_raises_when_no_fld_file_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                find_fld(tmp)

    def test_raises_when_multiple_fld_files_present(self):
        """맵 디렉터리가 여러 리셉터를 섞어 담고 있으면 조용히 하나를 골라
        엉뚱한 그리드에 도킹하는 대신 바로 실패해야 한다."""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.maps.fld").write_text("x")
            (Path(tmp) / "b.maps.fld").write_text("x")
            with self.assertRaises(ValueError):
                find_fld(tmp)


class TestWriteFilelist(unittest.TestCase):
    """write_filelist의 배치 파일 레이아웃 - UNVERIFIED, 실제 바이너리로 두
    리간드짜리 시험 실행을 거쳐 확인해야 한다 (모듈 docstring/체크리스트 참고)."""

    def test_fld_first_then_ligand_and_resnam_alternating(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = write_filelist("receptor.maps.fld",
                                  ["a.pdbqt", "b.pdbqt"],
                                  str(tmp_path / "out"),
                                  str(tmp_path / "filelist_0.txt"))
            lines = Path(path).read_text().splitlines()
            self.assertEqual(lines[0], str(Path("receptor.maps.fld").resolve()))
            self.assertEqual(lines[1], str(Path("a.pdbqt").resolve()))
            self.assertEqual(lines[2], str((tmp_path / "out" / "a").resolve()))
            self.assertEqual(lines[3], str(Path("b.pdbqt").resolve()))
            self.assertEqual(lines[4], str((tmp_path / "out" / "b").resolve()))

    def test_every_written_path_is_absolute(self):
        """DEFECT 1, observed verbatim on the cluster: AutoDock-GPU resolves a
        relative path inside the filelist against the FILELIST'S OWN
        directory, not the working directory. docking/v2/maps/meeko_receptor
        .maps.fld became docking/ligand_shards/docking/v2/maps/... and failed
        with "Can't open fld file". All three entry kinds must be absolute."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "shards").mkdir()
            path = write_filelist("docking/v2/maps/meeko_receptor.maps.fld",
                                  ["docking/ligands/a.pdbqt",
                                   "docking/ligands/b.pdbqt"],
                                  "docking/out",
                                  str(tmp_path / "shards" / "filelist_0.txt"))
            for line in Path(path).read_text().splitlines():
                self.assertTrue(Path(line).is_absolute(),
                                f"non-absolute path written into filelist: {line}")


class TestBuildFilelistCommand(unittest.TestCase):
    def test_uses_filelist_not_lfile(self):
        """CHANGE 1: 배치 경로는 리간드 하나당 프로세스가 아니라 샤드당
        --filelist 배치 파일 하나를 쓴다."""
        cmd = build_filelist_command("shard0.txt", devnum=1, seed=42)
        self.assertIn("--filelist", cmd)
        self.assertIn("shard0.txt", cmd)
        self.assertNotIn("--lfile", cmd)

    def test_devnum_counts_from_one(self):
        """CHANGE 2: --devnum은 1부터 센다 (CUDA_VISIBLE_DEVICES와 다르게).
        여기서 어긋나면 조용히 엉뚱한 GPU로 가거나, GPU가 더 적은 장비에서
        실패한다."""
        cmd = build_filelist_command("shard0.txt", devnum=1, seed=42)
        self.assertIn("--devnum", cmd)
        self.assertEqual(cmd[cmd.index("--devnum") + 1], "1")

    def test_nrun_20_on_every_filelist_command(self):
        cmd = build_filelist_command("shard0.txt", devnum=1, seed=42)
        self.assertIn("--nrun", cmd)
        self.assertEqual(cmd[cmd.index("--nrun") + 1], "20")


class TestRunBatchSplitting(unittest.TestCase):
    """run_batch의 리간드 분배/샤드별 filelist/명령 구성 로직만 검증한다 -
    dry_run=True라서 autodock_gpu 바이너리는 전혀 호출되지 않는다 (이 머신에는
    설치돼 있지 않다)."""

    def _make_index(self, tmp: Path, n: int) -> str:
        index = tmp / "ligands.txt"
        index.write_text("\n".join(f"lig_{i:03d}.pdbqt" for i in range(n)) + "\n")
        return str(index)

    def _filelist_ligands(self, filelist_path: str) -> list[str]:
        lines = Path(filelist_path).read_text().splitlines()
        # line 0 is the fld path; ligand paths are the odd-position entries.
        return [Path(lines[i]).name for i in range(1, len(lines), 2)]

    def test_round_robin_not_contiguous(self):
        """v1의 근거: 리간드가 small/medium/large 크기 클래스로 나뉘어(4627/3128/8,
        torsion 상한 8/16/20) 연속 분할은 large를 한 샤드에 몰아 그 샤드가
        나머지보다 한참 늦게 끝난다. 인덱스 순서를 유지한 채 라운드로빈으로
        나누면 각 샤드가 매 gpus번째 리간드를 받는다."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "maps").mkdir()
            (tmp_path / "maps" / "receptor.maps.fld").write_text("x")
            index = self._make_index(tmp_path, 9)

            plan = run_batch(index, str(tmp_path / "maps"), str(tmp_path / "out"),
                             gpus=3, shard_dir=str(tmp_path / "shards"),
                             dry_run=True)

            shard0_ligs = self._filelist_ligands(plan["shards"][0]["filelist"])
            shard1_ligs = self._filelist_ligands(plan["shards"][1]["filelist"])
            self.assertEqual(shard0_ligs, ["lig_000.pdbqt", "lig_003.pdbqt",
                                           "lig_006.pdbqt"])
            self.assertEqual(shard1_ligs, ["lig_001.pdbqt", "lig_004.pdbqt",
                                           "lig_007.pdbqt"])

    def test_every_ligand_gets_into_exactly_one_shards_filelist(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "maps").mkdir()
            (tmp_path / "maps" / "receptor.maps.fld").write_text("x")
            index = self._make_index(tmp_path, 11)

            plan = run_batch(index, str(tmp_path / "maps"), str(tmp_path / "out"),
                             gpus=4, shard_dir=str(tmp_path / "shards"),
                             dry_run=True)

            all_ligs = [lig for info in plan["shards"].values()
                       for lig in self._filelist_ligands(info["filelist"])]
            self.assertEqual(sorted(all_ligs),
                             sorted(f"lig_{i:03d}.pdbqt" for i in range(11)))

    def test_every_shard_command_uses_nrun_20_filelist_and_the_discovered_fld(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "maps").mkdir()
            fld = tmp_path / "maps" / "receptor.maps.fld"
            fld.write_text("x")
            index = self._make_index(tmp_path, 5)

            plan = run_batch(index, str(tmp_path / "maps"), str(tmp_path / "out"),
                             gpus=2, shard_dir=str(tmp_path / "shards"),
                             dry_run=True)

            self.assertEqual(plan["fld"], str(fld))
            for info in plan["shards"].values():
                cmd = info["command"]
                self.assertIn("--filelist", cmd)
                self.assertNotIn("--lfile", cmd)
                self.assertEqual(int(cmd[cmd.index("--nrun") + 1]), 20)
                # fld path lives inside the shard's filelist, not on the cmdline.
                self.assertEqual(Path(info["filelist"]).read_text().splitlines()[0],
                                 str(fld))

    def test_shard_0_gets_devnum_1_and_shard_3_gets_devnum_4(self):
        """CHANGE 2, pinned at the run_batch level: shard index 0 (0-based,
        matching dock_unidock.split_index) must map to --devnum 1 and shard
        index 3 to --devnum 4. An off-by-one here silently docks on the wrong
        GPU or fails on a machine with fewer devices."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "maps").mkdir()
            (tmp_path / "maps" / "receptor.maps.fld").write_text("x")
            index = self._make_index(tmp_path, 40)

            plan = run_batch(index, str(tmp_path / "maps"), str(tmp_path / "out"),
                             gpus=4, shard_dir=str(tmp_path / "shards"),
                             dry_run=True)

            cmd0 = plan["shards"][0]["command"]
            cmd3 = plan["shards"][3]["command"]
            self.assertEqual(cmd0[cmd0.index("--devnum") + 1], "1")
            self.assertEqual(cmd3[cmd3.index("--devnum") + 1], "4")
            self.assertEqual(plan["shards"][0]["devnum"], 1)
            self.assertEqual(plan["shards"][3]["devnum"], 4)


class _FakeProcess:
    """Stands in for subprocess.Popen so tests never invoke the real
    autodock_gpu binary (not installed on this machine)."""

    def __init__(self, status: int):
        self._status = status

    def wait(self) -> int:
        return self._status


class TestRunBatchFailureDetection(unittest.TestCase):
    """DEFECT 2, observed on the cluster: a shard whose autodock_gpu process
    exited nonzero (or exited zero but wrote no .dlg) was swallowed silently
    -- run_batch returned a plan that looked fine, with zero .dlg files on
    disk. That is the same failure shape as three earlier real regressions
    in this project (a truncated CCD residue name zeroing every affinity, a
    naive RMSD reporting 0.63 A as 6.13, an Open Babel mistyping promoting the
    4th-best pose to 1st): the pipeline runs to completion and produces a
    plausible-looking number. A printed warning would not have caught any of
    those either -- run_batch must raise."""

    def _make_index(self, tmp: Path, n: int) -> str:
        index = tmp / "ligands.txt"
        index.write_text("\n".join(f"lig_{i:03d}.pdbqt" for i in range(n)) + "\n")
        return str(index)

    def test_nonzero_exit_status_raises_with_log_tail_in_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "maps").mkdir()
            (tmp_path / "maps" / "receptor.maps.fld").write_text("x")
            index = self._make_index(tmp_path, 2)

            def fake_popen(command, stdout, stderr):
                stdout.write("Error: Can't open fld file some/bad/path.fld.\n"
                             "Error: get_gridinfo failed with fld file "
                             "specified in file list.\n")
                stdout.flush()
                return _FakeProcess(status=1)

            with patch("dock_autodock_gpu.subprocess.Popen", side_effect=fake_popen):
                with self.assertRaises(RuntimeError) as ctx:
                    run_batch(index, str(tmp_path / "maps"), str(tmp_path / "out"),
                             gpus=1, shard_dir=str(tmp_path / "shards"))
            self.assertIn("Can't open fld file", str(ctx.exception))

    def test_zero_exit_but_no_dlg_output_still_raises(self):
        """The exit status can lie: a shard can exit 0 having produced
        nothing. That must be treated as a failure too."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "maps").mkdir()
            (tmp_path / "maps" / "receptor.maps.fld").write_text("x")
            index = self._make_index(tmp_path, 2)

            def fake_popen(command, stdout, stderr):
                stdout.write("AutoDock-GPU version: v1.6-20-gbe06a13\n")
                stdout.flush()
                return _FakeProcess(status=0)  # exits "successfully"

            with patch("dock_autodock_gpu.subprocess.Popen", side_effect=fake_popen):
                with self.assertRaises(RuntimeError) as ctx:
                    run_batch(index, str(tmp_path / "maps"), str(tmp_path / "out"),
                             gpus=1, shard_dir=str(tmp_path / "shards"))
            self.assertIn("0/2", str(ctx.exception))

    def test_dlg_output_present_with_zero_exit_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "maps").mkdir()
            (tmp_path / "maps" / "receptor.maps.fld").write_text("x")
            out_dir = tmp_path / "out"
            out_dir.mkdir()
            index = self._make_index(tmp_path, 2)

            def fake_popen(command, stdout, stderr):
                stdout.write("done\n")
                stdout.flush()
                # simulate the real binary having written .dlg outputs
                (out_dir / "lig_000.dlg").write_text("DOCKED: MODEL 1\n")
                (out_dir / "lig_001.dlg").write_text("DOCKED: MODEL 1\n")
                return _FakeProcess(status=0)

            with patch("dock_autodock_gpu.subprocess.Popen", side_effect=fake_popen):
                plan = run_batch(index, str(tmp_path / "maps"), str(out_dir),
                                 gpus=1, shard_dir=str(tmp_path / "shards"))
            self.assertEqual(plan["results"][0][0]["status"], 0)


class TestPreflight(unittest.TestCase):
    """shutil.which only says the file exists. The binary is linked against a
    newer libstdc++ than the compute nodes ship, so on an interactive shell
    without LD_LIBRARY_PATH it is found, executed, and dies with
    "GLIBCXX_3.4.26 not found" -- which reached the user as a shard failure
    after the batch had been planned and the filelists written."""

    def _fake_binary(self, tmp, body):
        path = Path(tmp) / "autodock_gpu"
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
        return str(tmp)

    def _with_path(self, directory):
        return unittest.mock.patch.dict(
            os.environ, {"PATH": directory + os.pathsep + os.environ["PATH"]})

    def test_missing_binary_is_reported_with_where_to_look(self):
        with unittest.mock.patch.dict(os.environ, {"PATH": "/nonexistent"}):
            problem = preflight()
        self.assertIsNotNone(problem)
        self.assertIn("not on PATH", problem)

    def test_a_working_binary_returns_none(self):
        with TemporaryDirectory() as tmp:
            directory = self._fake_binary(tmp, "exit 0\n")
            with self._with_path(directory):
                self.assertIsNone(preflight())

    def test_glibcxx_failure_names_the_ld_library_path_fix(self):
        with TemporaryDirectory() as tmp:
            directory = self._fake_binary(
                tmp,
                "echo \"autodock_gpu: /lib64/libstdc++.so.6: version "
                "\\`GLIBCXX_3.4.26' not found\" >&2\nexit 1\n")
            with self._with_path(directory):
                problem = preflight()
        self.assertIsNotNone(problem)
        self.assertIn("LD_LIBRARY_PATH", problem)
        self.assertIn("CONDA_PREFIX", problem)

    def test_a_non_glibcxx_failure_does_not_claim_a_library_problem(self):
        with TemporaryDirectory() as tmp:
            directory = self._fake_binary(tmp, "echo 'no CUDA device' >&2\nexit 2\n")
            with self._with_path(directory):
                problem = preflight()
        self.assertIn("no CUDA device", problem)
        self.assertNotIn("LD_LIBRARY_PATH", problem)



if __name__ == "__main__":
    unittest.main()
