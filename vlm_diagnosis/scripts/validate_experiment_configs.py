"""Validate experiment contracts, decisions, runners, manifests, and output paths."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
from typing import Any, Iterator

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = ROOT / "experiments" / "configs"
DECISIONS_PATH = ROOT / "experiments" / "DECISIONS.md"
EXPECTED_BUDGETS = [0.2, 0.4, 0.6, 0.8]
VALID_STATES = {"READY", "PARTIAL", "PLANNED", "BLOCKED", "COMPLETE"}
VALID_RUN_KINDS = {"smoke", "discovery", "confirmation"}
MANIFEST_REQUIRED_FIELDS = {
    "dataset",
    "dataset_revision",
    "split",
    "sample_id",
    "selection_seed",
}


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


def decision_ids(path: Path = DECISIONS_PATH) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\|\s*((?:G\d{2})|(?:M[0-9A-Z]+-\d{2}))\s*\|", text))


def _resource_values(config: dict[str, Any], suffix: str) -> Iterator[tuple[str, str]]:
    def walk(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key.endswith(suffix) and isinstance(child, str):
                    yield child_path, child
                yield from walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, f"{path}[{index}]")

    yield from walk(config)


def _validate_manifest(path: Path) -> list[str]:
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}: line {line_number} invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"{path}: line {line_number} must be an object")
                continue
            missing = sorted(MANIFEST_REQUIRED_FIELDS.difference(record))
            if missing:
                errors.append(f"{path}: line {line_number} missing fields {missing}")
    return errors


def _check_resource(
    *,
    label: str,
    exists: bool,
    state: str,
    errors: list[str],
    unresolved: list[str],
) -> None:
    if exists:
        return
    message = f"resource:{label}"
    if state in {"READY", "COMPLETE"}:
        errors.append(f"missing required {message}")
    else:
        unresolved.append(message)


def validate_config(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    errors: list[str] = []
    resource_unresolved: list[str] = []
    if not isinstance(config, dict):
        return ["root must be a mapping"], []

    if not config.get("schema_version"):
        errors.append("missing schema_version")
    if not config.get("stage"):
        errors.append("missing stage")
    state = config.get("status")
    if state not in VALID_STATES:
        errors.append(f"invalid status: {state!r}")
    run_kind = config.get("run_kind")
    if run_kind not in VALID_RUN_KINDS:
        errors.append(f"invalid run_kind: {run_kind!r}")
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append("seed must be an integer")

    output = config.get("output")
    if not isinstance(output, str):
        errors.append("output must be a path string")
    elif run_kind in VALID_RUN_KINDS:
        expected_prefix = f"results/{run_kind}/"
        if not output.startswith(expected_prefix):
            errors.append(f"output for run_kind={run_kind} must start with {expected_prefix}")

    budgets = config.get("budgets_keep")
    if budgets is not None and budgets != EXPECTED_BUDGETS:
        errors.append(f"budgets_keep must equal {EXPECTED_BUDGETS}, got {budgets}")

    decisions = config.get("required_decisions")
    if not isinstance(decisions, list) or not decisions:
        errors.append("required_decisions must be a non-empty list")
    else:
        known = decision_ids()
        unknown = sorted(set(decisions).difference(known))
        if unknown:
            errors.append(f"unknown required_decisions: {unknown}")

    for key_path, module_name in _resource_values(config, "runner"):
        if "TBD" in module_name.upper():
            continue
        _check_resource(
            label=f"{key_path}={module_name}",
            exists=importlib.util.find_spec(module_name) is not None,
            state=state,
            errors=errors,
            unresolved=resource_unresolved,
        )

    for key_path, manifest_name in _resource_values(config, "manifest"):
        if "TBD" in manifest_name.upper():
            continue
        manifest_path = ROOT / manifest_name
        exists = manifest_path.is_file() and manifest_path.stat().st_size > 0
        _check_resource(
            label=f"{key_path}={manifest_name}",
            exists=exists,
            state=state,
            errors=errors,
            unresolved=resource_unresolved,
        )
        if exists:
            errors.extend(_validate_manifest(manifest_path))

    unresolved = sorted(set(unresolved_values(config)) | set(resource_unresolved))
    return errors, unresolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="report unresolved values/resources without returning a failing exit status",
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
