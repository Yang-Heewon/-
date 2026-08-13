import unittest
from pathlib import Path

from vlm_diagnosis.scripts.validate_experiment_configs import (
    EXPECTED_BUDGETS,
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


if __name__ == "__main__":
    unittest.main()
