"""Fail-closed audit for strict D0 source-denial evaluations.

The audit ties together four independently useful pieces of evidence:

* an image-only write manifest, used to inventory forbidden source files;
* a question-only read manifest, checked recursively for source path keys;
* an ``openat`` strace from the first reader invocation; and
* first/repeat result JSONL files, checked for coverage and exact prediction
  determinism.

An arm is supplied as ``NAME=FIRST,TRACE,REPEAT`` (``::`` may replace commas).
The result loaders deliberately tolerate heterogeneous metadata and result
schemas.  They discover common sample/question identifiers, ignore metadata
records, and only require package byte/hash fields when an arm reports package
storage fields.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ID_KEYS = ("sample_id", "memory_id", "image_id", "episode_id")
QUESTION_ID_KEYS = ("question_id", "query_id", "qa_id", "qid")
SOURCE_PATH_KEYS = (
    "image",
    "image_path",
    "source_image",
    "source_image_path",
    "source_path",
    "pixel_path",
)
WRITE_LEAK_KEYS = {
    "question",
    "questions",
    "question_id",
    "query",
    "query_id",
    "answer",
    "answers",
    "gold",
}
PRIMARY_BYTE_KEYS = (
    "package_bytes",
    "used_bytes",
    "payload_bytes",
    "stored_bytes",
    "artifact_bytes",
)
PRIMARY_HASH_KEYS = (
    "package_sha256",
    "used_sha256",
    "payload_sha256",
    "materialized_payload_sha256",
    "memory_sha256",
    "artifact_sha256",
)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PID_PREFIX_RE = re.compile(r"^(?:(?P<plain>\d+)|\[pid\s+(?P<bracket>\d+)\])\s+")
OPENAT_RE = re.compile(
    r"\bopenat(?:2)?\((?P<dirfd>[^,]+),\s*(?P<quoted>\"(?:\\.|[^\"\\])*\")"
)
EXIT_RE = re.compile(r"\+\+\+ exited with (?P<status>\d+) \+\+\+")


@dataclass(frozen=True)
class ArmPaths:
    name: str
    first: Path
    trace: Path
    repeat: Path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: JSONL is empty")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _resolve_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def _walk_keys(value: Any, location: str = "$") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            yield str(key), child_location, child
            yield from _walk_keys(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{location}[{index}]")


def _is_read_path_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized in {
        "image",
        "images",
        "image_path",
        "image_paths",
        "source_image",
        "source_images",
        "source_path",
        "source_paths",
        "pixel_path",
        "pixel_paths",
        "pixel_values",
        "pixels",
    }:
        return True
    if "path" in normalized and any(
        token in normalized for token in ("image", "source", "pixel")
    ):
        return True
    return normalized.endswith(("image_file", "source_file", "pixel_file"))


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _infer_dataset_root(source: Path, repo_root: Path) -> Path:
    """Infer ``.../data/DATASET`` when possible, otherwise use the parent."""

    try:
        relative = source.relative_to(repo_root)
    except ValueError:
        return source.parent
    parts = relative.parts
    if "data" in parts:
        index = parts.index("data")
        if index + 1 < len(parts) - 1:
            return (repo_root.joinpath(*parts[: index + 2])).resolve(strict=False)
    return source.parent


def inspect_manifests(
    write_manifest: Path,
    read_manifest: Path,
    repo_root: Path,
    explicit_dataset_roots: list[Path],
) -> tuple[dict[str, Any], set[tuple[str, str]], set[Path], set[str], set[Path]]:
    errors: list[str] = []
    write_rows = _load_jsonl(write_manifest)
    read_rows = _load_jsonl(read_manifest)

    source_paths: set[Path] = set()
    source_basenames: set[str] = set()
    write_leaks: list[dict[str, Any]] = []
    duplicate_sources: list[str] = []
    seen_source_rows: Counter[Path] = Counter()
    for row_index, row in enumerate(write_rows):
        raw_path = _first_present(row, SOURCE_PATH_KEYS)
        if not isinstance(raw_path, str) or not raw_path.strip():
            errors.append(f"write row {row_index} has no recognized source image path")
        else:
            source = _resolve_path(raw_path, repo_root)
            source_paths.add(source)
            source_basenames.add(source.name)
            seen_source_rows[source] += 1
        for key, location, _ in _walk_keys(row):
            if key.lower().replace("-", "_") in WRITE_LEAK_KEYS:
                write_leaks.append({"row": row_index, "location": location, "key": key})
    duplicate_sources = [
        _display_path(path, repo_root)
        for path, count in sorted(seen_source_rows.items(), key=lambda item: str(item[0]))
        if count > 1
    ]
    if write_leaks:
        errors.append("write manifest contains future question/answer keys")

    forbidden_read_keys: list[dict[str, Any]] = []
    expected_keys: set[tuple[str, str]] = set()
    duplicate_question_keys: list[list[str]] = []
    for row_index, row in enumerate(read_rows):
        for key, location, value in _walk_keys(row):
            if _is_read_path_key(key):
                forbidden_read_keys.append(
                    {
                        "row": row_index,
                        "location": location,
                        "key": key,
                        "value_type": type(value).__name__,
                    }
                )
        sample_id = _first_present(row, SAMPLE_ID_KEYS)
        questions = row.get("questions")
        candidates: list[dict[str, Any]]
        if isinstance(questions, list):
            candidates = [question for question in questions if isinstance(question, dict)]
            if len(candidates) != len(questions):
                errors.append(f"read row {row_index} contains a non-object question")
        else:
            candidates = [row]
        for question_index, question in enumerate(candidates):
            question_id = _first_present(question, QUESTION_ID_KEYS)
            if sample_id is None or question_id is None:
                errors.append(
                    f"read row {row_index}, question {question_index} lacks sample/question id"
                )
                continue
            key = (str(sample_id), str(question_id))
            if key in expected_keys:
                duplicate_question_keys.append(list(key))
            expected_keys.add(key)
    if forbidden_read_keys:
        errors.append("read manifest contains image/source/pixel path keys")
    if duplicate_question_keys:
        errors.append("read manifest contains duplicate sample/question keys")
    if not expected_keys:
        errors.append("read manifest contains no recognized questions")

    if explicit_dataset_roots:
        dataset_roots = set(explicit_dataset_roots)
        root_origin = "explicit"
    else:
        dataset_roots = {_infer_dataset_root(path, repo_root) for path in source_paths}
        root_origin = "inferred"

    basenames_to_paths: dict[str, list[str]] = {}
    for path in source_paths:
        basenames_to_paths.setdefault(path.name, []).append(_display_path(path, repo_root))
    basename_collisions = {
        basename: sorted(paths)
        for basename, paths in sorted(basenames_to_paths.items())
        if len(paths) > 1
    }

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "write": {
            "path": _display_path(write_manifest, repo_root),
            "sha256": _sha256(write_manifest),
            "n_rows": len(write_rows),
            "image_only": not write_leaks,
            "question_or_answer_key_occurrences": write_leaks,
            "duplicate_source_rows": duplicate_sources,
        },
        "read": {
            "path": _display_path(read_manifest, repo_root),
            "sha256": _sha256(read_manifest),
            "n_rows": len(read_rows),
            "n_expected_questions": len(expected_keys),
            "question_only_no_source_paths": not forbidden_read_keys,
            "forbidden_path_key_occurrences": forbidden_read_keys,
            "duplicate_question_keys": duplicate_question_keys,
        },
        "source_inventory": {
            "n_exact_source_paths": len(source_paths),
            "exact_source_paths": sorted(_display_path(path, repo_root) for path in source_paths),
            "source_basenames": sorted(source_basenames),
            "basename_collisions": basename_collisions,
            "dataset_root_origin": root_origin,
            "dataset_roots": sorted(_display_path(path, repo_root) for path in dataset_roots),
        },
        "errors": errors,
    }
    return summary, expected_keys, source_paths, source_basenames, dataset_roots


def _metadata_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record_type") == "run_metadata":
            metadata.append(row)
        elif row.get("mode") == "read" and row.get("process_id") is not None:
            metadata.append(row)
    return metadata


def _extract_reader_pid(
    rows: list[dict[str, Any]], errors: list[str], label: str
) -> int | None:
    metadata_pids = {
        int(row["process_id"])
        for row in _metadata_rows(rows)
        if isinstance(row.get("process_id"), int) and not isinstance(row.get("process_id"), bool)
    }
    result_pids = {
        int(row["reader_pid"])
        for row in rows
        if isinstance(row.get("reader_pid"), int) and not isinstance(row.get("reader_pid"), bool)
    }
    if len(metadata_pids) > 1:
        errors.append(f"{label}: multiple metadata process_id values")
    if len(result_pids) > 1:
        errors.append(f"{label}: multiple result reader_pid values")
    if metadata_pids and result_pids and metadata_pids != result_pids:
        errors.append(f"{label}: metadata process_id does not match result reader_pid")
    candidates = metadata_pids or result_pids
    if len(candidates) != 1:
        errors.append(f"{label}: cannot identify one reader PID")
        return None
    return next(iter(candidates))


def _result_key(
    row: dict[str, Any], expected_samples: set[str]
) -> tuple[str, str] | None:
    question_id = _first_present(row, QUESTION_ID_KEYS)
    if question_id is None:
        return None
    sample_id = _first_present(row, SAMPLE_ID_KEYS)
    if sample_id is None:
        return None
    # ``memory_id`` and ``image_id`` are accepted only when they identify an
    # expected sample.  This avoids silently treating unrelated metadata IDs as
    # question keys.
    if str(sample_id) not in expected_samples:
        return str(sample_id), str(question_id)
    return str(sample_id), str(question_id)


def _valid_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_results(
    path: Path,
    expected_keys: set[tuple[str, str]],
    read_manifest: Path,
    repo_root: Path,
    label: str,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], int | None]:
    errors: list[str] = []
    limitations: list[str] = []
    rows = _load_jsonl(path)
    expected_samples = {sample_id for sample_id, _ in expected_keys}
    results: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_keys: list[list[str]] = []
    unkeyed_question_rows = 0
    for row in rows:
        if _first_present(row, QUESTION_ID_KEYS) is None:
            continue
        key = _result_key(row, expected_samples)
        if key is None:
            unkeyed_question_rows += 1
            continue
        if key in results:
            duplicate_keys.append(list(key))
        else:
            results[key] = row
    if duplicate_keys:
        errors.append(f"{label}: duplicate question result keys")
    if unkeyed_question_rows:
        errors.append(f"{label}: question rows without a recognized sample id")

    result_keys = set(results)
    missing = sorted(expected_keys - result_keys)
    extra = sorted(result_keys - expected_keys)
    if missing:
        errors.append(f"{label}: missing expected question results")
    if extra:
        errors.append(f"{label}: extra question results")

    feasible_count = 0
    infeasible_count = 0
    prediction_count = 0
    package_applicable_count = 0
    package_signature_by_sample: dict[str, tuple[tuple[str, Any], ...]] = {}
    package_field_issues: list[dict[str, Any]] = []
    for key, row in results.items():
        feasible = row.get("feasible") is not False
        if feasible:
            feasible_count += 1
            if "prediction" in row and row.get("prediction") is not None:
                prediction_count += 1
            else:
                errors.append(f"{label}: feasible result {key} has no prediction")
        else:
            infeasible_count += 1
            if not row.get("infeasible_reason"):
                errors.append(f"{label}: infeasible result {key} has no reason")

        if row.get("source_path_in_read_manifest") is True:
            errors.append(f"{label}: result {key} reports source path in read manifest")
        if row.get("pixel_values_used") is True:
            errors.append(f"{label}: result {key} reports pixel_values_used=true")

        byte_items = [(field, row[field]) for field in PRIMARY_BYTE_KEYS if field in row]
        hash_items = [(field, row[field]) for field in PRIMARY_HASH_KEYS if field in row]
        if byte_items or hash_items:
            package_applicable_count += 1
        if feasible and (byte_items or hash_items):
            issues: list[str] = []
            if not byte_items:
                issues.append("hash reported without a recognized byte field")
            if not hash_items:
                issues.append("byte field reported without a recognized payload hash")
            for field, value in byte_items:
                if not _valid_positive_integer(value):
                    issues.append(f"{field} is not a positive integer")
            for field, value in hash_items:
                if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                    issues.append(f"{field} is not a 64-hex SHA-256")
            if issues:
                package_field_issues.append({"key": list(key), "issues": issues})

            signature = tuple(byte_items + hash_items)
            previous = package_signature_by_sample.get(key[0])
            if previous is None:
                package_signature_by_sample[key[0]] = signature
            elif previous != signature:
                package_field_issues.append(
                    {
                        "key": list(key),
                        "issues": ["package byte/hash signature changes within one sample"],
                    }
                )
    if package_field_issues:
        errors.append(f"{label}: invalid or inconsistent package byte/hash fields")
    if package_applicable_count == 0:
        limitations.append(
            "This schema reports no recognized package byte/hash fields; package fields were not applicable."
        )

    actual_manifest_hash = _sha256(read_manifest)
    metadata = _metadata_rows(rows)
    metadata_contract_checked = bool(metadata)
    for row in metadata:
        if row.get("source_path_available") is True:
            errors.append(f"{label}: metadata reports source_path_available=true")
        if row.get("pixel_values_available") is True:
            errors.append(f"{label}: metadata reports pixel_values_available=true")
        declared_hash = row.get("manifest_sha256")
        if declared_hash is not None and declared_hash != actual_manifest_hash:
            errors.append(f"{label}: metadata manifest_sha256 does not match read manifest")
        declared_path = row.get("manifest")
        if isinstance(declared_path, str):
            if _resolve_path(declared_path, repo_root) != read_manifest.resolve(strict=False):
                errors.append(f"{label}: metadata manifest path is not the supplied read manifest")
    if not metadata:
        limitations.append(
            "No run_metadata row was found; source_path_available and declared manifest hash were not checked."
        )

    reader_pid = _extract_reader_pid(rows, errors, label)
    summary = {
        "status": "PASS" if not errors else "FAIL",
        "path": _display_path(path, repo_root),
        "sha256": _sha256(path),
        "n_jsonl_rows": len(rows),
        "n_expected_questions": len(expected_keys),
        "n_question_results": len(results),
        "n_feasible_results": feasible_count,
        "n_infeasible_results": infeasible_count,
        "n_feasible_predictions": prediction_count,
        "missing_keys": [list(key) for key in missing],
        "extra_keys": [list(key) for key in extra],
        "duplicate_keys": duplicate_keys,
        "reader_pid": reader_pid,
        "metadata_contract_checked": metadata_contract_checked,
        "package_fields": {
            "applicable_result_count": package_applicable_count,
            "valid": not package_field_issues,
            "issues": package_field_issues,
        },
        "errors": errors,
        "limitations": limitations,
    }
    return summary, results, reader_pid


def _parse_trace_pid(line: str) -> int | None:
    match = PID_PREFIX_RE.match(line)
    if match is None:
        return None
    return int(match.group("plain") or match.group("bracket"))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_open_path(path: Path, repo_root: Path) -> Path:
    """Normalize a traced path without resolving every system-library open.

    Resolving tens of thousands of ``/usr`` and model-cache paths through the
    filesystem is needlessly slow.  Source and package candidates live under
    ``repo_root`` in this protocol, so only repository paths need symlink-aware
    resolution; external paths are normalized lexically.
    """

    normalized = Path(os.path.normpath(str(path)))
    if _is_within(normalized, repo_root):
        return normalized.resolve(strict=False)
    return normalized


def audit_trace(
    trace_path: Path,
    reader_pid: int | None,
    source_paths: set[Path],
    source_basenames: set[str],
    dataset_roots: set[Path],
    repo_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    traced_pids: set[int] = set()
    pid_openat_counts: Counter[int] = Counter()
    exit_statuses: dict[int, int] = {}
    parsed_openat = 0
    unparsed_openat = 0
    unresolved_relative_dirfd = 0
    source_hits: list[dict[str, Any]] = []
    dataset_hits: list[dict[str, Any]] = []
    basename_hits: list[dict[str, Any]] = []
    resolved_cache: dict[tuple[str, str], Path | None] = {}

    with trace_path.open(encoding="utf-8", errors="replace") as handle:
        lines = list(handle)
    for line_number, line in enumerate(lines, 1):
        pid = _parse_trace_pid(line)
        if pid is not None:
            traced_pids.add(pid)
        exit_match = EXIT_RE.search(line)
        if pid is not None and exit_match is not None:
            exit_statuses[pid] = int(exit_match.group("status"))

        if "openat(" not in line and "openat2(" not in line:
            continue
        match = OPENAT_RE.search(line)
        if match is None:
            unparsed_openat += 1
            continue
        parsed_openat += 1
        if pid is not None:
            pid_openat_counts[pid] += 1
        quoted_path = match.group("quoted")
        try:
            # Most strace paths are also valid JSON strings; the C-backed JSON
            # parser is materially faster on large traces.  ``ast`` remains a
            # fallback for C-style escapes accepted by strace but not JSON.
            raw_path = json.loads(quoted_path)
        except json.JSONDecodeError:
            try:
                raw_path = ast.literal_eval(quoted_path)
            except (SyntaxError, ValueError):
                unparsed_openat += 1
                continue
        if not isinstance(raw_path, str):
            unparsed_openat += 1
            continue
        dirfd = match.group("dirfd").strip()
        cache_key = (dirfd, raw_path)
        if cache_key in resolved_cache:
            resolved = resolved_cache[cache_key]
            path = Path(raw_path)
            resolvable = path.is_absolute() or dirfd == "AT_FDCWD"
        else:
            path = Path(raw_path)
            resolvable = path.is_absolute() or dirfd == "AT_FDCWD"
            if path.is_absolute():
                resolved = _resolve_open_path(path, repo_root)
            elif dirfd == "AT_FDCWD":
                resolved = _resolve_open_path(repo_root / path, repo_root)
            else:
                resolved = None
            resolved_cache[cache_key] = resolved
        if not resolvable:
            unresolved_relative_dirfd += 1

        hit = {
            "line": line_number,
            "pid": pid,
            "raw_path": raw_path,
            "resolved_path": _display_path(resolved, repo_root) if resolved else None,
            "dirfd": dirfd,
        }
        if resolved in source_paths:
            source_hits.append(hit)
        if resolved is not None and any(_is_within(resolved, root) for root in dataset_roots):
            dataset_hits.append(hit)
        if path.name in source_basenames:
            basename_hits.append({**hit, "path_resolvable": resolvable})

    if unparsed_openat:
        errors.append("one or more openat/openat2 lines could not be parsed")
    if reader_pid is None:
        errors.append("reader PID is unavailable, so trace coverage cannot be established")
    else:
        if reader_pid not in traced_pids:
            errors.append("reader PID is absent from the strace")
        if pid_openat_counts[reader_pid] == 0:
            errors.append("strace contains no parsed openat call for the reader PID")
        if reader_pid not in exit_statuses:
            errors.append("strace has no terminal exit record for the reader PID")
        elif exit_statuses[reader_pid] != 0:
            errors.append("strace reports a non-zero reader exit status")
    if source_hits:
        errors.append("strace opened or attempted an exact source image path")
    if dataset_hits:
        errors.append("strace opened or attempted a path inside a forbidden dataset root")
    # A basename match is deliberately fail-closed.  It catches relative opens
    # whose true cwd cannot be reconstructed from an openat-only trace.
    if basename_hits:
        errors.append("strace opened or attempted a source-image basename")

    return {
        "status": "PASS" if not errors else "FAIL",
        "path": _display_path(trace_path, repo_root),
        "sha256": _sha256(trace_path),
        "n_lines": len(lines),
        "n_traced_pids": len(traced_pids),
        "reader_pid": reader_pid,
        "reader_pid_present": reader_pid in traced_pids if reader_pid is not None else False,
        "reader_openat_count": pid_openat_counts[reader_pid] if reader_pid is not None else 0,
        "reader_exit_status": exit_statuses.get(reader_pid) if reader_pid is not None else None,
        "parsed_openat_calls": parsed_openat,
        "unparsed_openat_calls": unparsed_openat,
        "unresolved_relative_dirfd_calls": unresolved_relative_dirfd,
        "exact_source_hits": source_hits,
        "dataset_root_hits": dataset_hits,
        "source_basename_hits": basename_hits,
        "errors": errors,
    }


def compare_repeats(
    first: dict[tuple[str, str], dict[str, Any]],
    repeat: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    first_keys = set(first)
    repeat_keys = set(repeat)
    missing = sorted(first_keys - repeat_keys)
    extra = sorted(repeat_keys - first_keys)
    prediction_mismatches: list[dict[str, Any]] = []
    feasibility_mismatches: list[dict[str, Any]] = []
    package_mismatches: list[dict[str, Any]] = []
    for key in sorted(first_keys & repeat_keys):
        first_prediction = first[key].get("prediction")
        repeat_prediction = repeat[key].get("prediction")
        if first_prediction != repeat_prediction:
            prediction_mismatches.append(
                {
                    "key": list(key),
                    "first": first_prediction,
                    "repeat": repeat_prediction,
                }
            )
        first_feasible = first[key].get("feasible") is not False
        repeat_feasible = repeat[key].get("feasible") is not False
        if first_feasible != repeat_feasible:
            feasibility_mismatches.append(
                {"key": list(key), "first": first_feasible, "repeat": repeat_feasible}
            )
        compared_fields = [
            field
            for field in PRIMARY_BYTE_KEYS + PRIMARY_HASH_KEYS
            if field in first[key] or field in repeat[key]
        ]
        differing = {
            field: {"first": first[key].get(field), "repeat": repeat[key].get(field)}
            for field in compared_fields
            if first[key].get(field) != repeat[key].get(field)
        }
        if differing:
            package_mismatches.append({"key": list(key), "fields": differing})
    if missing or extra:
        errors.append("first and repeat key sets differ")
    if prediction_mismatches:
        errors.append("first and repeat predictions are not exactly deterministic")
    if feasibility_mismatches:
        errors.append("first and repeat feasibility decisions differ")
    if package_mismatches:
        errors.append("first and repeat package byte/hash reports differ")
    return {
        "status": "PASS" if not errors else "FAIL",
        "n_common_questions": len(first_keys & repeat_keys),
        "missing_in_repeat": [list(key) for key in missing],
        "extra_in_repeat": [list(key) for key in extra],
        "prediction_mismatches": prediction_mismatches,
        "feasibility_mismatches": feasibility_mismatches,
        "package_field_mismatches": package_mismatches,
        "exact_prediction_determinism": not prediction_mismatches and not missing and not extra,
        "errors": errors,
    }


def parse_arm(value: str, repo_root: Path) -> ArmPaths:
    if "=" not in value:
        raise ValueError("arm must be NAME=FIRST,TRACE,REPEAT")
    name, payload = value.split("=", 1)
    separator = "::" if "::" in payload else ","
    parts = [part.strip() for part in payload.split(separator)]
    if not name.strip() or len(parts) != 3 or any(not part for part in parts):
        raise ValueError("arm must be NAME=FIRST,TRACE,REPEAT")
    return ArmPaths(
        name=name.strip(),
        first=_resolve_path(parts[0], repo_root),
        trace=_resolve_path(parts[1], repo_root),
        repeat=_resolve_path(parts[2], repo_root),
    )


def audit_arm(
    arm: ArmPaths,
    expected_keys: set[tuple[str, str]],
    source_paths: set[Path],
    source_basenames: set[str],
    dataset_roots: set[Path],
    read_manifest: Path,
    repo_root: Path,
) -> dict[str, Any]:
    first_summary, first, first_pid = validate_results(
        arm.first, expected_keys, read_manifest, repo_root, f"{arm.name}/first"
    )
    repeat_summary, repeat, _ = validate_results(
        arm.repeat, expected_keys, read_manifest, repo_root, f"{arm.name}/repeat"
    )
    trace_summary = audit_trace(
        arm.trace,
        first_pid,
        source_paths,
        source_basenames,
        dataset_roots,
        repo_root,
    )
    determinism = compare_repeats(first, repeat)
    errors = (
        first_summary["errors"]
        + repeat_summary["errors"]
        + trace_summary["errors"]
        + determinism["errors"]
    )
    return {
        "status": "PASS" if not errors else "FAIL",
        "first_results": first_summary,
        "first_reader_trace": trace_summary,
        "repeat_results": repeat_summary,
        "repeat_determinism": determinism,
        "errors": errors,
        "limitations": [
            "Only the first reader run has a supplied syscall trace; the repeat run is checked for outputs, not source access.",
        ],
    }


def run_audit(
    write_manifest: Path,
    read_manifest: Path,
    arms: list[ArmPaths],
    repo_root: Path,
    dataset_roots: list[Path] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=False)
    write_manifest = write_manifest.resolve(strict=False)
    read_manifest = read_manifest.resolve(strict=False)
    explicit_roots = [path.resolve(strict=False) for path in (dataset_roots or [])]
    manifest_summary, expected, sources, basenames, forbidden_roots = inspect_manifests(
        write_manifest, read_manifest, repo_root, explicit_roots
    )
    arm_summaries: dict[str, Any] = {}
    duplicate_arm_names: list[str] = []
    for arm in arms:
        if arm.name in arm_summaries:
            duplicate_arm_names.append(arm.name)
            continue
        arm_summaries[arm.name] = audit_arm(
            arm,
            expected,
            sources,
            basenames,
            forbidden_roots,
            read_manifest,
            repo_root,
        )
    errors = list(manifest_summary["errors"])
    if not arms:
        errors.append("no arms were supplied")
    if duplicate_arm_names:
        errors.append(f"duplicate arm names: {sorted(set(duplicate_arm_names))}")
    for name, summary in arm_summaries.items():
        if summary["status"] != "PASS":
            errors.append(f"arm {name!r} failed one or more checks")

    limitations = [
        "This is an openat/openat2 trace audit, not a proof against pre-opened file descriptors, IPC, network transfer, shared memory, or source bytes embedded before tracing.",
        "Relative AT_FDCWD paths are resolved against repo_root because chdir is not present in an openat-only trace; source basename matches therefore fail closed.",
        "Package byte/hash validation checks reported field shape and within-sample/repeat consistency; it cannot independently hash artifacts when result rows do not expose artifact paths.",
        "Dataset roots are inferred from source paths unless --dataset-root is supplied; explicit roots are preferable for a paper audit.",
        "Exact repeat equality tests deterministic decoded predictions, not deterministic hidden states or floating-point kernels.",
    ]
    return {
        "record_type": "strict_source_denial_audit",
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "passed": not errors,
        "repo_root": str(repo_root),
        "manifests": manifest_summary,
        "arms": arm_summaries,
        "duplicate_arm_names": sorted(set(duplicate_arm_names)),
        "errors": errors,
        "limitations": limitations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", required=True, help="image-only JSONL")
    parser.add_argument("--read-manifest", required=True, help="question-only JSONL")
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=FIRST,TRACE,REPEAT",
        help="repeatable arm specification; :: is also accepted as a separator",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        help="explicit forbidden dataset root; repeat for multiple roots",
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--out", required=True, help="JSON audit output")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve(strict=False)
    write_manifest = _resolve_path(args.write_manifest, repo_root)
    read_manifest = _resolve_path(args.read_manifest, repo_root)
    arms = [parse_arm(value, repo_root) for value in args.arm]
    dataset_roots = [_resolve_path(value, repo_root) for value in args.dataset_root]
    result = run_audit(
        write_manifest,
        read_manifest,
        arms,
        repo_root,
        dataset_roots=dataset_roots,
    )
    out = _resolve_path(args.out, repo_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{result['status']}] {_display_path(out, repo_root)}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
