import unittest
from pathlib import Path

from vlm_diagnosis.scripts.validate_experiment_configs import (
    EXPECTED_BUDGETS,
    VALID_RUN_KINDS,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "experiments" / "configs"


class ExperimentConfigTest(unittest.TestCase):
    def test_all_stage_configs_are_structurally_valid(self):
        paths = sorted(CONFIGS.glob("m*.yaml"))
        self.assertEqual(len(paths), 9)
        for path in paths:
            errors, _ = validate_config(path)
            self.assertEqual(errors, [], f"{path}: {errors}")

    def test_budgeted_stages_use_the_shared_grid(self):
        import yaml

        for name in ("m2a.yaml", "m3.yaml", "m4.yaml", "m2b.yaml", "m5.yaml", "m6.yaml"):
            with (CONFIGS / name).open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            self.assertEqual(config["budgets_keep"], EXPECTED_BUDGETS)

    def test_all_configs_have_seed_and_matching_output_class(self):
        import yaml

        for path in sorted(CONFIGS.glob("m*.yaml")):
            with path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            self.assertIsInstance(config["seed"], int)
            self.assertIn(config["run_kind"], VALID_RUN_KINDS)
            self.assertTrue(
                config["output"].startswith(f"results/{config['run_kind']}/"),
                path,
            )

    def test_missing_planned_resources_are_reported_as_unresolved(self):
        # m4는 runner(m4_information_types)와 PCTD manifest가 아직 없는 PLANNED 단계다.
        # 해당 runner가 구현되면 아직 미구현인 다른 단계로 픽스처를 옮긴다.
        errors, unresolved = validate_config(CONFIGS / "m4.yaml")
        self.assertEqual(errors, [])
        self.assertTrue(any(item.startswith("resource:data.") for item in unresolved))
        self.assertTrue(any(item.startswith("resource:runner=") for item in unresolved))


if __name__ == "__main__":
    unittest.main()
