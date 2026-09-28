import subprocess
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

TEMPLATE = Path("configs/tl.toml.in")


def render(mode):
    """run_d3.sh의 치환을 거친 config를 받아 파싱한다."""
    out = subprocess.run(
        ["bash", "scripts/run_d3.sh", "--print-config", "--mode", mode],
        capture_output=True, text=True, check=True,
    )
    return tomllib.loads(out.stdout)


class TestProductionSweepLogging(unittest.TestCase):
    """v1 §4가 기록한 함정: 'validation NLL is only computed on save epochs'.
    프로덕션 스윕에서 save_every_n_epochs를 크게 두면 체크포인트에 val NLL이
    남지 않아, epoch 선택 근거가 사라진다."""

    def test_production_sweep_saves_every_20_epochs(self):
        cfg = render("production-sweep")
        self.assertEqual(cfg["parameters"]["save_every_n_epochs"], 20)

    def test_production_sweep_runs_to_200_epochs(self):
        cfg = render("production-sweep")
        self.assertEqual(cfg["parameters"]["num_epochs"], 200)

    def test_tb_logdir_is_set_so_wandb_can_mirror_it(self):
        cfg = render("production-sweep")
        self.assertTrue(cfg["tb_logdir"])

    def test_randomize_all_smiles_is_on(self):
        """기본값은 파일 읽기 시 한 번만 무작위화해 증강이 전혀 없다.
        n=25에서는 선택이 아니라 필수다."""
        cfg = render("production-sweep")
        self.assertTrue(cfg["parameters"]["randomize_all_smiles"])

    def test_no_isomeric_smiles_key(self):
        """v4.5.11의 TL SectionParameters에는 이 필드가 없고 GlobalConfig는
        extra='forbid'다. 포함하면 하드 기동 오류다."""
        cfg = render("production-sweep")
        self.assertNotIn("isomeric_smiles", cfg["parameters"])

    def test_diagnostic_mode_saves_every_epoch(self):
        cfg = render("diagnostic")
        self.assertEqual(cfg["parameters"]["save_every_n_epochs"], 1)
        self.assertEqual(cfg["parameters"]["num_epochs"], 20)


if __name__ == "__main__":
    unittest.main()
