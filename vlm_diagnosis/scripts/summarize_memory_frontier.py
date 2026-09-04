"""Aggregate heterogeneous source-denial smoke outputs on one question set.

The script is deliberately descriptive: it computes no confidence interval
and cannot promote a smoke result into a paper claim.  It aligns every arm to
a designated source-image result, reports missing/extra questions, source-
correct conditional retention, exact prediction agreement, one package byte
per memory item, and timing fields without pretending unlike timing scopes are
equivalent.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from vlm_diagnosis.core.metrics import normalize_text


ROOT = Path(__file__).resolve().parents[2]
TIMING_FIELDS = (
    "first_token_seconds",
    "prefill_seconds",
    "answer_seconds",
    "package_load_seconds",
    "reconstruction_seconds",
    "decode_seconds",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_question_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load one result per (sample, question), rejecting hidden duplicates."""

    results: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _jsonl(path):
        if row.get("question_id") is None or row.get("prediction") is None:
            continue
        key = (str(row["sample_id"]), str(row["question_id"]))
        if key in results:
            raise ValueError(f"duplicate question result in {path}: {key}")
        results[key] = row
    if not results:
        raise ValueError(f"no question results in {path}")
    return results


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _package_bytes_by_sample(
    rows: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for (sample_id, _), row in rows.items():
        value = row.get("package_bytes")
        if value is None:
            continue
        size = int(value)
        if size < 0:
            raise ValueError("package_bytes cannot be negative")
        if sample_id in sizes and sizes[sample_id] != size:
            raise ValueError(
                f"package_bytes changes within sample {sample_id}: "
                f"{sizes[sample_id]} != {size}"
            )
        sizes[sample_id] = size
    return sizes


def summarize_arm(
    reference: dict[tuple[str, str], dict[str, Any]],
    arm: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    reference_keys = set(reference)
    arm_keys = set(arm)
    common = sorted(reference_keys & arm_keys)
    source_correct = [key for key in common if float(reference[key].get("em", 0)) == 1.0]
    em_values = [float(arm[key]["em"]) for key in common if arm[key].get("em") is not None]
    anls_values = [
        float(arm[key]["anls"]) for key in common if arm[key].get("anls") is not None
    ]
    retention = [float(arm[key]["em"]) for key in source_correct
                 if arm[key].get("em") is not None]
    exact_agreement = [
        float(str(arm[key]["prediction"]) == str(reference[key]["prediction"]))
        for key in common
    ]
    normalized_agreement = [
        float(normalize_text(str(arm[key]["prediction"]))
              == normalize_text(str(reference[key]["prediction"])))
        for key in common
    ]
    arm_sizes = _package_bytes_by_sample(arm)
    source_sizes = _package_bytes_by_sample(reference)
    shared_size_samples = sorted(set(arm_sizes) & set(source_sizes))
    ratios = [arm_sizes[sample] / source_sizes[sample] for sample in shared_size_samples]

    timing: dict[str, dict[str, float | int | None]] = {}
    for field in TIMING_FIELDS:
        values = [float(arm[key][field]) for key in common
                  if arm[key].get(field) is not None]
        if values:
            timing[field] = {
                "n": len(values),
                "mean": _mean(values),
                "median": _median(values),
            }

    return {
        "n_reference_questions": len(reference_keys),
        "n_arm_questions": len(arm_keys),
        "n_common_questions": len(common),
        "missing_question_keys": [list(key) for key in sorted(reference_keys - arm_keys)],
        "extra_question_keys": [list(key) for key in sorted(arm_keys - reference_keys)],
        "em": _mean(em_values),
        "anls": _mean(anls_values),
        "source_correct_n": len(source_correct),
        "conditional_retention": _mean(retention),
        "prediction_agreement_exact": _mean(exact_agreement),
        "prediction_agreement_normalized": _mean(normalized_agreement),
        "package": {
            "n_memories": len(arm_sizes),
            "bytes_by_sample": arm_sizes,
            "mean_bytes": _mean(list(arm_sizes.values())),
            "median_bytes": _median(list(arm_sizes.values())),
            "mean_ratio_to_source_container": _mean(ratios),
            "median_ratio_to_source_container": _median(ratios),
        },
        "timing_by_reported_scope": timing,
    }


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("arm must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise ValueError("arm must contain non-empty NAME and PATH")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return name.strip(), path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="source-image JSONL")
    parser.add_argument("--arm", action="append", default=[], help="NAME=JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--claim-scope", default="implementation smoke only")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source_path = source_path.resolve()
    reference = load_question_results(source_path)
    arms: dict[str, Any] = {}
    for value in args.arm:
        name, path = parse_named_path(value)
        if name in arms:
            raise ValueError(f"duplicate arm name {name!r}")
        arms[name] = {
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            **summarize_arm(reference, load_question_results(path)),
        }

    output = {
        "record_type": "memory_frontier_descriptive_summary",
        "claim_scope": args.claim_scope,
        "source_path": (
            str(source_path.relative_to(ROOT))
            if source_path.is_relative_to(ROOT) else str(source_path)
        ),
        "source": summarize_arm(reference, reference),
        "arms": arms,
        "warnings": [
            "No confidence intervals: this summary cannot support a paper claim.",
            "Timing fields with different names/scopes are not directly interchangeable.",
            "Package bytes exclude a shared external retrieval index unless its arm stores one.",
        ],
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
