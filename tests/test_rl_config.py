import sys
import unittest
from itertools import combinations
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_rl_config import (  # noqa: E402
    FRAG,
    RL_DROP_COMPONENTS,
    RL_TRANSFORM_OVERRIDES,
    build_config,
)

ARMS = {
    "A-prime": "configs/rl_A_prime.toml",
    "B": "configs/rl_B.toml",
    "C": "configs/rl_C.toml",
}


def load(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def components(config):
    """{component_type: [endpoint, ...]} for one RL config's single stage."""
    out = {}
    for block in config["stage"][0]["scoring"]["component"]:
        for component_type, body in block.items():
            out.setdefault(component_type, []).extend(body["endpoint"])
    return out


class TestInceptionRemoved(unittest.TestCase):
    """spec §4.1: inception 메모리는 점수 내림차순으로 잘리고(inception.py:86-90)
    매 스텝 배치 128개가 추가되며(279행), 축출된 시드는 재진입이 영구 차단된다
    (41·57행). core 25개 중 A1AFA를 뺀 전부가 TPSA 창 밖이라 총점 0.06 이하이고,
    에이전트의 총점 중앙값은 1.00이다. 시드는 첫 배치에서 밀려난다."""

    def test_no_inception_block_in_any_arm(self):
        for arm, path in ARMS.items():
            with self.subTest(arm=arm):
                cfg = load(path)
                self.assertNotIn("inception", cfg["stage"][0])
                self.assertNotIn("inception", cfg)


class TestArmsDifferOnlyInPrior(unittest.TestCase):
    """단일 변수 절제: prior와 arm 이름이 붙은 출력 경로 외의 모든 설정이 같아야
    한다. v4는 arm B 하나만 돌리지만(설계 §1.1), TL-C는 §4.2 체크포인트 선택의
    대체 경로로 남아 있으므로 같은 목적함수를 공유해야 비교가 오염되지 않는다."""

    def test_every_arm_has_a_distinct_prior(self):
        agents = [load(path)["parameters"]["agent_file"] for path in ARMS.values()]
        self.assertEqual(len(set(agents)), len(agents), agents)

    def test_scoring_and_learning_settings_are_identical_across_arms(self):
        for left, right in combinations(sorted(ARMS), 2):
            with self.subTest(arms=(left, right)):
                a, b = load(ARMS[left]), load(ARMS[right])
                self.assertEqual(a["stage"][0]["scoring"], b["stage"][0]["scoring"])
                self.assertEqual(a["learning_strategy"], b["learning_strategy"])
                self.assertEqual(a["parameters"]["batch_size"],
                                 b["parameters"]["batch_size"])

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


class TestGeneratedConfigsAreNotHandEdited(unittest.TestCase):
    """THE TEST THAT WAS MISSING. v4's re-derived TPSA window was hand-edited into
    rl_C.toml and rl_A_prime.toml while configs/_rl_scoring.frag kept the old
    bounds, so the next regeneration -- the first `--arm B` run -- would have
    silently restored the window v4 exists to replace, under a header reading
    'do not edit by hand'. Byte-comparing each file with its generator output is
    what makes that header true."""

    def test_each_config_is_exactly_what_the_generator_emits(self):
        for arm, path in ARMS.items():
            with self.subTest(arm=arm):
                self.assertEqual(Path(path).read_text(), build_config(arm),
                                 f"{path} differs from `make_rl_config.py --arm "
                                 f"{arm}`; regenerate instead of editing it")


class TestV4Deviations(unittest.TestCase):
    """The RL objective's two declared departures from the blueprint fragment."""

    def test_sascore_is_absent_from_every_arm(self):
        self.assertIn("SAScore", RL_DROP_COMPONENTS)
        for arm, path in ARMS.items():
            with self.subTest(arm=arm):
                self.assertNotIn("SAScore", components(load(path)))

    def test_tpsa_window_comes_from_the_single_source(self):
        want = RL_TRANSFORM_OVERRIDES["TPSA"]
        self.assertEqual(want, {"low": 60.0, "high": 140.0},
                         "results/v4_tpsa_window.md derived (60, 140)")
        for arm, path in ARMS.items():
            with self.subTest(arm=arm):
                transform = components(load(path))["TPSA"][0]["transform"]
                self.assertEqual(transform["low"], want["low"])
                self.assertEqual(transform["high"], want["high"])
                # Only the two bounds move - the shape coefficients are held so
                # the window is the single explanation for any score change.
                self.assertEqual(transform["coef_div"], 120.0)
                self.assertEqual(transform["coef_si"], 20.0)
                self.assertEqual(transform["coef_se"], 20.0)

    def test_dropping_sascore_renormalises_every_remaining_exponent_to_one_half(self):
        """REINVENT normalizes geometric-mean weights by their sum
        (reinvent/scoring/aggregators/means.py: `scores ** (weights/sum_weights)`),
        so removing SAScore's 0.5 is not score-neutral: the divisor goes 2.5 -> 2.0
        and each surviving exponent 0.4 -> 0.5. Consequences, all verified against
        that formula: totals fall (TPSA 0.5 gives 0.7579 -> 0.7071), a missing
        carboxylate costs 1e-8**0.5 = 1.0e-04 instead of 1e-8**0.4 = 6.31e-04 --
        which is the measured number in the fragment's own comment, and what
        confirms this model rather than assuming it -- and diversity_filter
        minscore 0.4 now needs a TPSA component of 0.4**2 = 0.160 rather than
        0.4**2.5 = 0.101."""
        for arm, path in ARMS.items():
            with self.subTest(arm=arm):
                scored = [endpoint
                          for component_type, endpoints in components(load(path)).items()
                          if component_type != "CustomAlerts"
                          for endpoint in endpoints]
                weights = [endpoint["weight"] for endpoint in scored]
                self.assertEqual(sorted(weights), [1.0, 1.0])
                self.assertEqual(sum(weights), 2.0)
                self.assertAlmostEqual(1e-8 ** (1.0 / sum(weights)), 1.0e-04)
                self.assertAlmostEqual(0.4 ** sum(weights), 0.16)


class TestFragmentStaysTheBlueprintRecord(unittest.TestCase):
    """The deviations live in the generator BECAUSE the fragment is not RL-only.

    It is also the source of configs/score_panel.toml (make_scoring_config.py),
    the column names scripts/panel_report.py reads, and the tail that
    reinvent4_avb1_scoring_config_sketch.toml is asserted to end with
    (tests/test_config.py). v2/v3's reported panel numbers rest on that chain, so
    editing the fragment to carry v4's objective would rewrite a pre-registration
    document and drop the panel's SA column. If a future change moves the window
    into the fragment on purpose, this test is the place to record that decision --
    it should fail loudly first."""

    def test_fragment_still_holds_the_blueprint_window_and_sascore(self):
        frag = Path(FRAG).read_text()
        self.assertIn("transform.low = 40.0", frag)
        self.assertIn("transform.high = 115.0", frag)
        self.assertIn("[stage.scoring.component.SAScore]", frag)


class TestConfigValidatesAgainstReinvent4Schema(unittest.TestCase):
    """Parses every config through REINVENT4's own pydantic RLConfig
    (extra="forbid"), so an unknown key or shape error surfaces now instead
    of at submission. This does not require a GPU or the agent_file to exist."""

    def test_rl_config_schema(self):
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
