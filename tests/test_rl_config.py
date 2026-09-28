import unittest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ARMS = {"A-prime": "configs/rl_A_prime.toml", "C": "configs/rl_C.toml"}


def load(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


class TestInceptionRemoved(unittest.TestCase):
    """spec §4.1: inception 메모리는 점수 내림차순으로 잘리고(inception.py:86-90)
    매 스텝 배치 128개가 추가되며(279행), 축출된 시드는 재진입이 영구 차단된다
    (41·57행). core 25개 중 A1AFA를 뺀 전부가 TPSA 창 밖이라 총점 0.06 이하이고,
    에이전트의 총점 중앙값은 1.00이다. 시드는 첫 배치에서 밀려난다."""

    def test_no_inception_block_in_either_arm(self):
        for arm, path in ARMS.items():
            with self.subTest(arm=arm):
                cfg = load(path)
                stage = cfg["stage"][0]
                self.assertNotIn("inception", stage)
                self.assertNotIn("inception", cfg)


class TestArmsDifferOnlyInPrior(unittest.TestCase):
    """단일 변수 절제: prior 외의 모든 설정이 같아야 한다."""

    def test_priors_differ(self):
        a = load(ARMS["A-prime"])
        c = load(ARMS["C"])
        self.assertNotEqual(a["parameters"]["agent_file"],
                            c["parameters"]["agent_file"])

    def test_scoring_and_learning_settings_are_identical(self):
        a = load(ARMS["A-prime"])
        c = load(ARMS["C"])
        self.assertEqual(a["stage"][0]["scoring"], c["stage"][0]["scoring"])
        self.assertEqual(a["learning_strategy"], c["learning_strategy"])
        self.assertEqual(a["parameters"]["batch_size"],
                         c["parameters"]["batch_size"])

    def test_max_steps_is_1000(self):
        """문서상 '600 steps'는 오류다. 실행된 값은 1000이다."""
        for arm, path in ARMS.items():
            with self.subTest(arm=arm):
                self.assertEqual(load(path)["stage"][0]["max_steps"], 1000)

    def test_diversity_filter_applies_throughout(self):
        for arm, path in ARMS.items():
            with self.subTest(arm=arm):
                df = load(path)["stage"][0]["diversity_filter"]
                self.assertEqual(df["type"], "IdenticalMurckoScaffold")
                self.assertEqual(df["bucket_size"], 25)
                self.assertEqual(df["minscore"], 0.4)


class TestConfigValidatesAgainstReinvent4Schema(unittest.TestCase):
    """Parses both configs through REINVENT4's own pydantic RLConfig
    (extra="forbid"), so an unknown key or shape error surfaces now instead
    of at submission. This does not require a GPU or the agent_file to
    exist - priors/focused_A_prime.prior is expected to be missing until the
    Task 5 TL sweep is run and select_epoch() picks an epoch."""

    def test_rl_config_schema(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "REINVENT4"))
        try:
            from reinvent.validation import ReinventConfig
            from reinvent.runmodes.RL.validation import RLConfig
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                f"REINVENT4 runtime deps not importable here ({exc}); "
                "run in the 'reinvent' conda env for this check"
            )

        for arm, path in ARMS.items():
            with self.subTest(arm=arm):
                cfg = load(path)
                # Same top-level schema reinvent/Reinvent.py:74 applies on startup.
                ReinventConfig(**cfg)
                # Deep-validates parameters/stage/learning_strategy/diversity_filter
                # shapes, which ReinventConfig leaves as untyped dict/list.
                rl_cfg = {k: v for k, v in cfg.items()
                          if k not in ("run_type", "device", "tb_logdir")}
                RLConfig(**rl_cfg)


if __name__ == "__main__":
    unittest.main()
