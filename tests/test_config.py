import sys
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from objective import ALERT_SMARTS, CARBOXYLATE_SMARTS

SKETCH = "reinvent4_avb1_scoring_config_sketch.toml"


def load():
    with open(SKETCH, "rb") as handle:
        return tomllib.load(handle)


def endpoints(config):
    """Flatten every component endpoint into {name: (component_type, endpoint)}."""
    out = {}
    for block in config["stage"][0]["scoring"]["component"]:
        for component_type, body in block.items():
            for endpoint in body["endpoint"]:
                out[endpoint["name"]] = (component_type, endpoint)
    return out


class TestSingleStage(unittest.TestCase):
    def test_exactly_one_stage_of_600_steps(self):
        config = load()
        self.assertEqual(len(config["stage"]), 1)
        self.assertEqual(config["stage"][0]["max_steps"], 600)

    def test_common_parameters(self):
        config = load()
        self.assertEqual(config["learning_strategy"]["type"], "dap")
        self.assertEqual(config["learning_strategy"]["sigma"], 128)
        self.assertEqual(config["parameters"]["batch_size"], 128)
        self.assertEqual(config["diversity_filter"]["type"],
                         "IdenticalMurckoScaffold")
        self.assertEqual(config["diversity_filter"]["bucket_size"], 25)
        self.assertEqual(config["diversity_filter"]["minscore"], 0.4)
        self.assertEqual(config["stage"][0]["scoring"]["type"], "geometric_mean")

    def test_diversity_filter_is_not_per_stage(self):
        """A per-stage filter would apply to part of the run, not the whole of it."""
        config = load()
        self.assertIn("diversity_filter", config)
        self.assertNotIn("diversity_filter", config["stage"][0])

    def test_inception_seeds_the_training_actives(self):
        config = load()
        self.assertEqual(config["inception"]["smiles_file"],
                         "data/actives_core.smi")
        self.assertEqual(config["inception"]["memory_size"], 100)
        self.assertEqual(config["inception"]["sample_size"], 10)


class TestComponents(unittest.TestCase):
    def test_component_set_is_exactly_four(self):
        config = load()
        names = set(endpoints(config))
        self.assertEqual(
            names,
            {"carboxylate MIDAS anchor", "TPSA", "SA score", "unwanted groups"},
        )
        # endpoints() keys by name, so a duplicate name would be swallowed and the
        # set above would still match. Count the raw blocks too.
        self.assertEqual(len(config["stage"][0]["scoring"]["component"]), 4)

    def test_removed_components_are_absent(self):
        types = {t for t, _ in endpoints(load()).values()}
        for removed in ("TanimotoSimilarity", "MatchingSubstructure", "SlogP",
                        "Qed", "MolecularWeight", "NumRotBond", "HBondDonors"):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, types)

    def test_carboxylate_is_a_groupcount_gate(self):
        component_type, endpoint = endpoints(load())["carboxylate MIDAS anchor"]
        self.assertEqual(component_type, "GroupCount")
        self.assertEqual(endpoint["params"]["smarts"], [CARBOXYLATE_SMARTS])
        self.assertEqual(endpoint["transform"]["type"], "right_step")
        self.assertEqual(endpoint["transform"]["high"], 1)
        self.assertEqual(endpoint["weight"], 1.0)

    def test_tpsa_window(self):
        _, endpoint = endpoints(load())["TPSA"]
        transform = endpoint["transform"]
        self.assertEqual(transform["type"], "double_sigmoid")
        self.assertEqual(transform["low"], 40.0)
        self.assertEqual(transform["high"], 115.0)
        self.assertEqual(transform["coef_div"], 120.0)
        self.assertEqual(transform["coef_si"], 20.0)
        self.assertEqual(transform["coef_se"], 20.0)
        self.assertEqual(endpoint["weight"], 1.0)

    def test_sascore_guard_rail_window(self):
        _, endpoint = endpoints(load())["SA score"]
        transform = endpoint["transform"]
        self.assertEqual(transform["type"], "reverse_sigmoid")
        self.assertEqual(transform["low"], 6.0)
        self.assertEqual(transform["high"], 8.0)
        self.assertEqual(transform["k"], 0.5)
        self.assertEqual(endpoint["weight"], 0.5)

    def test_alerts_include_narrow_aniline_and_exclude_the_broad_one(self):
        _, endpoint = endpoints(load())["unwanted groups"]
        smarts = endpoint["params"]["smarts"]
        self.assertEqual(smarts, ALERT_SMARTS)
        self.assertNotIn("[NH2,NH][c]", smarts)


def load_reinvent_transforms():
    """Load REINVENT's transform registry WITHOUT importing the reinvent package.

    `import reinvent.scoring.transforms` pulls in torch, which is not installed in
    the base conda environment. The transform modules themselves need only numpy, so
    we load them under a synthetic package name; their relative `from .transform
    import Transform` then resolves inside it.
    """
    import importlib.util
    import types

    directory = (Path(__file__).resolve().parent.parent
                 / "REINVENT4" / "reinvent" / "scoring" / "transforms")
    package = types.ModuleType("rvt")
    package.__path__ = [str(directory)]
    sys.modules["rvt"] = package

    def load_module(name):
        spec = importlib.util.spec_from_file_location(
            f"rvt.{name}", directory / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"rvt.{name}"] = module
        spec.loader.exec_module(module)
        return module

    transform = load_module("transform")
    for name in ("sigmoid_functions", "sigmoids", "steps", "double_sigmoid"):
        load_module(name)
    return transform.get_transform


class TestReinventAcceptsTheNames(unittest.TestCase):
    """Review Focus 5: REINVENT resolves transform names and forbids extra keys.

    get_transform lowercases and strips underscores, so "right_step" -> "rightstep"
    -> RightStep. The parameter dataclasses reject unknown keys, so a typo in the
    TOML is a hard startup error rather than a silently ignored setting.
    """

    def test_transform_names_resolve(self):
        get_transform = load_reinvent_transforms()
        expected = {
            "right_step": "RightStep",
            "double_sigmoid": "DoubleSigmoid",
            "reverse_sigmoid": "ReverseSigmoid",
        }
        for name, class_name in expected.items():
            with self.subTest(name=name):
                cls, _ = get_transform(name)
                self.assertEqual(cls.__name__, class_name)

    def test_config_transform_blocks_instantiate(self):
        """Every transform table in the sketch must construct its REINVENT class."""
        get_transform = load_reinvent_transforms()
        for endpoint_name in ("carboxylate MIDAS anchor", "TPSA", "SA score"):
            with self.subTest(endpoint=endpoint_name):
                _, endpoint = endpoints(load())[endpoint_name]
                table = dict(endpoint["transform"])
                cls, param_cls = get_transform(table["type"])
                cls(param_cls(**table))

    def test_right_step_gate_is_binary(self):
        get_transform = load_reinvent_transforms()
        cls, param_cls = get_transform("right_step")
        transform = cls(param_cls(type="right_step", high=1))
        self.assertEqual(list(transform([0, 1, 2])), [0.0, 1.0, 1.0])


COMPONENT_MODULES = {
    "GroupCount": "reinvent_plugins.components.RDKit.comp_group_count",
    "TPSA": "reinvent_plugins.components.RDKit.comp_tpsa",
    "SAScore": "reinvent_plugins.components.SAScore.comp_sascore",
    "CustomAlerts": "reinvent_plugins.components.comp_custom_alerts",
}


class TestReinventAcceptsTheComponents(unittest.TestCase):
    """The transform tests cover transform tables. This covers the components.

    REINVENT's plugin components live in `reinvent_plugins`, a package separate
    from `reinvent`, so they import without torch. Each declares a pydantic
    `Parameters` dataclass; a `params` key it does not declare is a startup
    failure. Without this, a typo like `param.smarts` or a renamed component
    class would reach a GPU run unchallenged.
    """

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                               / "REINVENT4"))

    def test_every_component_class_resolves(self):
        for name, module_path in COMPONENT_MODULES.items():
            with self.subTest(component=name):
                module = __import__(module_path, fromlist=[name])
                self.assertIsNotNone(getattr(module, name, None))

    def test_every_params_key_is_declared_by_its_component(self):
        config = load()
        seen = set()
        for block in config["stage"][0]["scoring"]["component"]:
            for component_type, body in block.items():
                seen.add(component_type)
                module = __import__(COMPONENT_MODULES[component_type],
                                    fromlist=["Parameters"])
                declared = getattr(module, "Parameters", None)
                allowed = (set(declared.__dataclass_fields__)
                           if declared is not None else set())
                for endpoint in body["endpoint"]:
                    with self.subTest(component=component_type,
                                      endpoint=endpoint["name"]):
                        self.assertLessEqual(set(endpoint.get("params", {})),
                                             allowed)
        self.assertEqual(seen, set(COMPONENT_MODULES))


class TestFragments(unittest.TestCase):
    def test_stage_fragments_are_gone(self):
        self.assertFalse(Path("configs/_stage1_scoring.frag").exists())
        self.assertFalse(Path("configs/_stage2_scoring.frag").exists())

    def test_sketch_ends_with_the_fragment_verbatim(self):
        """The sketch is header + fragment, so the two cannot drift apart.

        endswith, not assertIn: containment would also pass if the fragment were
        duplicated or buried mid-file with unrelated content after it.
        """
        fragment = Path("configs/_rl_scoring.frag").read_text()
        sketch = Path(SKETCH).read_text()
        self.assertTrue(sketch.rstrip().endswith(fragment.strip()))


if __name__ == "__main__":
    unittest.main()
