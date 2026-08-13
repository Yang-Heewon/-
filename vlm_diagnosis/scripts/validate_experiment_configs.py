"""Validate staged experiment contracts and list unresolved user decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = ROOT / "experiments" / "configs"
EXPECTED_BUDGETS = [0.2, 0.4, 0.6, 0.8]
VALID_STATES = {"READY", "PARTIAL", "PLANNED", "BLOCKED", "COMPLETE"}


def unresolved_values(value: Any, path: str = "") -> Iterator[str]:
    if value is None:
        yield path or "<root>"
        return
    if isinstance(value, str) and "TBD" in value.upper():
        yield path or "<root>"
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from unresolved_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from unresolved_values(child, f"{path}[{index}]")


def validate_config(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["root must be a mapping"], []
    if not config.get("stage"):
        errors.append("missing stage")
    state = config.get("status")
    if state not in VALID_STATES:
        errors.append(f"invalid status: {state!r}")
    budgets = config.get("budgets_keep")
    if budgets is not None and budgets != EXPECTED_BUDGETS:
        errors.append(f"budgets_keep must equal {EXPECTED_BUDGETS}, got {budgets}")
    decisions = config.get("required_decisions")
    if not isinstance(decisions, list) or not decisions:
        errors.append("required_decisions must be a non-empty list")
    unresolved = sorted(set(unresolved_values(config)))
    return errors, unresolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="report TBD/null values without returning a failing exit status",
    )
    args = parser.parse_args()
    paths = args.paths or sorted(DEFAULT_CONFIG_DIR.glob("m*.yaml"))
    any_errors = False
    any_unresolved = False
    for path in paths:
        errors, unresolved = validate_config(path)
        print(f"[{path.name}] errors={len(errors)} unresolved={len(unresolved)}")
        for error in errors:
            print(f"  ERROR {error}")
        for item in unresolved:
            print(f"  DECIDE {item}")
        any_errors = any_errors or bool(errors)
        any_unresolved = any_unresolved or bool(unresolved)
    if any_errors or (any_unresolved and not args.allow_unresolved):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

