import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dock_autodock_gpu import NRUN, build_command, find_fld, parse_dlg, run_batch

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


class TestRunBatchSplitting(unittest.TestCase):
    """run_batch의 리간드 분배/명령 구성 로직만 검증한다 - dry_run=True라서
    autodock_gpu 바이너리는 전혀 호출되지 않는다 (이 머신에는 설치돼 있지 않다)."""

    def _make_index(self, tmp: Path, n: int) -> str:
        index = tmp / "ligands.txt"
        index.write_text("\n".join(f"lig_{i:03d}.pdbqt" for i in range(n)) + "\n")
        return str(index)

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

            shard0_ligs = [Path(cmd[cmd.index("--lfile") + 1]).name
                          for cmd in plan["shards"][0]["commands"]]
            shard1_ligs = [Path(cmd[cmd.index("--lfile") + 1]).name
                          for cmd in plan["shards"][1]["commands"]]
            self.assertEqual(shard0_ligs, ["lig_000.pdbqt", "lig_003.pdbqt",
                                           "lig_006.pdbqt"])
            self.assertEqual(shard1_ligs, ["lig_001.pdbqt", "lig_004.pdbqt",
                                           "lig_007.pdbqt"])

    def test_every_ligand_gets_a_command_and_none_are_dropped_or_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "maps").mkdir()
            (tmp_path / "maps" / "receptor.maps.fld").write_text("x")
            index = self._make_index(tmp_path, 11)

            plan = run_batch(index, str(tmp_path / "maps"), str(tmp_path / "out"),
                             gpus=4, shard_dir=str(tmp_path / "shards"),
                             dry_run=True)

            all_ligs = [Path(cmd[cmd.index("--lfile") + 1]).name
                       for info in plan["shards"].values()
                       for cmd in info["commands"]]
            self.assertEqual(sorted(all_ligs),
                             sorted(f"lig_{i:03d}.pdbqt" for i in range(11)))

    def test_every_command_uses_nrun_20_and_the_discovered_fld(self):
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
                for cmd in info["commands"]:
                    self.assertIn(str(fld), cmd)
                    self.assertEqual(int(cmd[cmd.index("--nrun") + 1]), 20)


if __name__ == "__main__":
    unittest.main()
