"""Strict descriptive summary for the physical recurrent-session JSONL.

Usage:
  python -m vlm_diagnosis.scripts.recurrent_session_analysis \
    --input results/smoke/recurrent_session.jsonl \
    --out results/smoke/recurrent_session_summary.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("full", "image_static", "recurrent")
SCALARS = (
    "em", "anls", "full_em", "full_anls", "loyalty", "entered_tokens", "cold_kv_bytes",
    "active_history_kv_bytes", "peak_active_kv_bytes", "selector_state_bytes",
    "combined_kv_and_state_bytes", "load_seconds", "ttft_seconds", "turn_seconds",
    "image_prefill_seconds", "h2d_kv_bytes", "d2h_new_kv_bytes", "historical_tokens",
    "active_history_tokens", "next_active_history_tokens", "peak_active_kv_tokens",
    "new_session_tokens", "initial_prefix_tokens", "budget_tokens",
)
INTEGER_SCALARS = {
    "entered_tokens", "cold_kv_bytes", "active_history_kv_bytes", "peak_active_kv_bytes",
    "selector_state_bytes", "combined_kv_and_state_bytes", "h2d_kv_bytes",
    "d2h_new_kv_bytes", "historical_tokens", "active_history_tokens",
    "next_active_history_tokens", "peak_active_kv_tokens", "new_session_tokens",
    "initial_prefix_tokens", "budget_tokens",
}
V11_INTEGERS = (
    "retained_kv_tokens", "retained_kv_bytes", "resident_gpu_kv_bytes",
    "logical_history_tokens_after", "initial_deleted_tokens", "deleted_tokens_this_turn",
    "deleted_image_tokens_this_turn", "session_metadata_bytes",
    "persistent_session_tensor_bytes", "compaction_peak_kv_bytes_upper_bound",
)


def _number(value: Any, name: str, *, integer: bool = False,
            optional: bool = False, high: float | None = None) -> float | int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (high is not None and result > high):
        raise ValueError(f"{name} is out of range")
    if integer and not result.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(result) if integer else result


def load_jsonl(path: str | Path) -> tuple[dict, list[dict]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_no}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no} is not a JSON object")
            records.append(record)
    metadata = [r for r in records if r.get("record_type") == "run_metadata"]
    rows = [r for r in records if r.get("record_type") != "run_metadata"]
    if len(metadata) != 1 or not rows:
        raise ValueError("input must contain exactly one metadata row and data rows")
    validate(metadata[0], rows)
    return metadata[0], rows


def _storage_mode(metadata: dict) -> str:
    version = metadata.get("schema_version")
    if version not in {"1.0", "1.1"} or metadata.get("stage") != "RECURRENT_SESSION":
        raise ValueError("unsupported metadata schema or stage")
    if version == "1.0":
        explicit = metadata.get("storage_mode", metadata.get("storage"))
        if explicit not in {None, "offload"}:
            raise ValueError("schema 1.0 is the legacy offload format")
        return "offload"
    mode = metadata.get("storage_mode")
    if mode not in {"delete", "offload"}:
        raise ValueError("schema 1.1 metadata requires storage_mode delete or offload")
    if metadata.get("storage") not in {None, mode}:
        raise ValueError("metadata storage fields disagree")
    return mode


def _id_list(value: Any, name: str, expected: int) -> list[int]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"{name} must contain exactly {expected} IDs")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{name} must contain nonnegative integer IDs")
    if value != sorted(set(value)):
        raise ValueError(f"{name} IDs must be sorted and unique")
    return value


def _validate_v11_row(row: dict, prefix: str, mode: str) -> None:
    if row.get("storage_mode") != mode or not isinstance(row.get("compression_applied"), bool):
        raise ValueError(f"{prefix} storage metadata disagree")
    for field in V11_INTEGERS:
        row[field] = _number(row.get(field), f"{prefix}.{field}", integer=True)
    row["retained_kv_fraction_of_initial"] = _number(
        row.get("retained_kv_fraction_of_initial"),
        f"{prefix}.retained_kv_fraction_of_initial")
    row["initial_cache_setup_seconds"] = _number(
        row.get("initial_cache_setup_seconds"), f"{prefix}.initial_cache_setup_seconds")
    before = row.get("selection_before")
    if not isinstance(before, dict):
        raise ValueError(f"{prefix}.selection_before must be an object")
    before_image = _number(before.get("selected_image_tokens"),
                           f"{prefix}.selection_before.selected_image_tokens", integer=True)
    active = _id_list(row.get("active_indices"), f"{prefix}.active_indices",
                      row["active_history_tokens"])
    next_active = _id_list(row.get("next_active_indices"), f"{prefix}.next_active_indices",
                           row["next_active_history_tokens"])
    logical = row["historical_tokens"] + row["new_session_tokens"]
    if row["logical_history_tokens_after"] != logical:
        raise ValueError(f"{prefix} logical history accounting mismatch")
    if row["peak_active_kv_tokens"] != row["active_history_tokens"] + row["new_session_tokens"]:
        raise ValueError(f"{prefix} active-cache peak token accounting mismatch")
    if row["active_history_tokens"] <= 0:
        raise ValueError(f"{prefix} active history must be nonempty")
    token_bytes, remainder = divmod(row["active_history_kv_bytes"], row["active_history_tokens"])
    if remainder or token_bytes <= 0:
        raise ValueError(f"{prefix} invalid per-token KV byte accounting")
    exact_bytes = {
        "peak_active_kv_bytes": row["peak_active_kv_tokens"] * token_bytes,
        "retained_kv_bytes": row["retained_kv_tokens"] * token_bytes,
    }
    if any(row[field] != expected for field, expected in exact_bytes.items()):
        raise ValueError(f"{prefix} KV token/byte accounting mismatch")
    if row["next_active_history_tokens"] != row["selection_after"]["kept_count"]:
        raise ValueError(f"{prefix} next active set and selection disagree")
    if row["entered_tokens"] != len(set(next_active) - set(active)):
        raise ValueError(f"{prefix} entered_tokens does not match physical IDs")
    if any(item >= row["historical_tokens"] for item in active):
        raise ValueError(f"{prefix} active IDs exceed prior logical history")
    if any(item >= logical for item in next_active):
        raise ValueError(f"{prefix} next IDs exceed logical history")
    condition = row["condition_id"]
    compressed = condition != "full"
    expected_active = row["budget_tokens"] if compressed else logical
    if row["next_active_history_tokens"] != expected_active:
        raise ValueError(f"{prefix} persistent cache is not fixed-budget/full as declared")
    expected_initial_deleted = (
        row["initial_prefix_tokens"] - row["budget_tokens"]
        if mode == "delete" and compressed else 0)
    if row["initial_deleted_tokens"] != expected_initial_deleted:
        raise ValueError(f"{prefix} initial deletion accounting mismatch")
    if (mode == "delete" and row["deleted_image_tokens_this_turn"]
            != before_image - row["selection_after"]["selected_image_tokens"]):
        raise ValueError(f"{prefix} deleted image accounting mismatch")
    expected_fraction = row["retained_kv_bytes"] / (row["initial_prefix_tokens"] * token_bytes)
    if not math.isclose(row["retained_kv_fraction_of_initial"], expected_fraction, rel_tol=1e-9):
        raise ValueError(f"{prefix} retained KV fraction mismatch")

    if mode == "delete":
        if any(row[field] != 0 for field in ("cold_kv_bytes", "h2d_kv_bytes", "d2h_new_kv_bytes")):
            raise ValueError(f"{prefix} delete mode cannot retain/transfer cold KV")
        allowed = set(active) | set(range(row["historical_tokens"], logical))
        if not set(next_active) <= allowed:
            raise ValueError(f"{prefix} delete mode resurrected an evicted logical ID")
        if row["retained_kv_tokens"] != expected_active:
            raise ValueError(f"{prefix} delete mode retained the wrong KV count")
        if row["resident_gpu_kv_bytes"] != row["retained_kv_bytes"]:
            raise ValueError(f"{prefix} delete mode retained KV outside the GPU cache")
        deleted = 0 if condition == "full" else row["peak_active_kv_tokens"] - expected_active
        # ``compact_cache`` creates index_select outputs, then the installed
        # DynamicCache constructor copies them once more: old peak + 2B.
        compaction = (row["peak_active_kv_tokens"]
                      + (2 * expected_active if compressed else 0)) * token_bytes
    else:
        if row["retained_kv_tokens"] != logical or row["retained_kv_bytes"] != row["cold_kv_bytes"]:
            raise ValueError(f"{prefix} offload mode must retain full logical history on CPU")
        if row["resident_gpu_kv_bytes"] != 0 or row["h2d_kv_bytes"] != row["active_history_kv_bytes"]:
            raise ValueError(f"{prefix} offload residency/transfer accounting mismatch")
        if row["d2h_new_kv_bytes"] != row["new_session_tokens"] * token_bytes:
            raise ValueError(f"{prefix} offload writeback accounting mismatch")
        deleted = 0
        compaction = row["peak_active_kv_bytes"]
    if row["compression_applied"] != (mode == "delete" and compressed):
        raise ValueError(f"{prefix} compression_applied mismatch")
    if row["deleted_tokens_this_turn"] != deleted:
        raise ValueError(f"{prefix} per-turn deletion accounting mismatch")
    if mode == "offload" and row["deleted_image_tokens_this_turn"] != 0:
        raise ValueError(f"{prefix} offload mode cannot delete image KV")
    if row["compaction_peak_kv_bytes_upper_bound"] != compaction:
        raise ValueError(f"{prefix} compaction peak accounting mismatch")
    persistent = (row["retained_kv_bytes"] + row["selector_state_bytes"]
                  + row["session_metadata_bytes"])
    if row["persistent_session_tensor_bytes"] != persistent:
        raise ValueError(f"{prefix} persistent tensor accounting mismatch")
    combined = row["cold_kv_bytes"] + compaction + row["selector_state_bytes"]
    if row["combined_kv_and_state_bytes"] != combined:
        raise ValueError(f"{prefix} combined transient accounting mismatch")


def validate(metadata: dict, rows: list[dict]) -> None:
    for key in ("run_id", "model"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"metadata missing {key}")
    mode = _storage_mode(metadata)
    version = metadata["schema_version"]
    declared = metadata.get("conditions")
    declared = declared.split(",") if isinstance(declared, str) else declared
    if (not isinstance(declared, list) or len(declared) != len(CONDITIONS)
            or set(declared) != set(CONDITIONS)):
        raise ValueError("metadata must declare full,image_static,recurrent")

    seen, by_question, warm, setup = set(), defaultdict(dict), {}, {}
    for index, row in enumerate(rows, 1):
        prefix = f"data row {index}"
        if row.get("run_id") != metadata["run_id"] or row.get("model") != metadata["model"]:
            raise ValueError("mixed run_id/model in input")
        for key in ("sample_id", "question_id", "dataset", "condition_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{prefix} missing {key}")
        condition = row["condition_id"]
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition}")
        key = (row["sample_id"], row["question_id"], condition)
        if key in seen:
            raise ValueError(f"duplicate sample/question/condition: {key}")
        seen.add(key)
        question_key = key[:2]
        by_question[question_key][condition] = row
        row["step"] = _number(row.get("step"), f"{prefix}.step", integer=True)
        if row["step"] < 1:
            raise ValueError(f"{prefix}.step must be positive")
        for field in SCALARS:
            row[field] = _number(row.get(field), f"{prefix}.{field}",
                                 integer=field in INTEGER_SCALARS,
                                 high=1.0 if field in {"em", "anls", "full_em", "full_anls", "loyalty"} else None)
        row["full_correct_retained"] = _number(
            row.get("full_correct_retained"), f"{prefix}.full_correct_retained",
            optional=True, high=1.0)
        row["peak_gpu_allocated_bytes"] = _number(
            row.get("peak_gpu_allocated_bytes"), f"{prefix}.peak_gpu_allocated_bytes",
            integer=True, optional=True)
        selection = row.get("selection_after")
        if not isinstance(selection, dict):
            raise ValueError(f"{prefix}.selection_after must be an object")
        for field in ("selected_image_tokens", "selected_history_text_tokens",
                      "selected_prefix_control_tokens", "kept_count"):
            selection[field] = _number(selection.get(field),
                                       f"{prefix}.selection_after.{field}", integer=True)
        composed = (selection["selected_image_tokens"]
                    + selection["selected_history_text_tokens"]
                    + selection["selected_prefix_control_tokens"])
        if composed != selection["kept_count"]:
            raise ValueError(f"{prefix} selection composition does not equal kept_count")
        for field in ("image_weight", "history_weight"):
            selection[field] = _number(selection.get(field), f"{prefix}.selection_after.{field}",
                                       optional=True, high=1.0)
        if selection["image_weight"] is not None and selection["history_weight"] is not None:
            if not math.isclose(selection["image_weight"] + selection["history_weight"], 1.0,
                                abs_tol=1e-6):
                raise ValueError(f"{prefix} image/history weights do not sum to one")
        if version == "1.1":
            _validate_v11_row(row, prefix, mode)
        else:
            if row.get("storage_mode") not in {None, "offload"}:
                raise ValueError(f"{prefix} schema 1.0 must use legacy offload")
            row["storage_mode"] = "offload"
            logical = row["historical_tokens"] + row["new_session_tokens"]
            row.update({
                "retained_kv_tokens": logical,
                "retained_kv_bytes": row["cold_kv_bytes"],
                "resident_gpu_kv_bytes": 0,
                "logical_history_tokens_after": logical,
                "initial_deleted_tokens": 0,
                "deleted_tokens_this_turn": 0,
                "deleted_image_tokens_this_turn": 0,
                "session_metadata_bytes": None,
                "persistent_session_tensor_bytes": None,
                "compaction_peak_kv_bytes_upper_bound": None,
                "initial_cache_setup_seconds": None,
            })
        sample = row["sample_id"]
        if sample in warm and warm[sample] != row["image_prefill_seconds"]:
            raise ValueError(f"inconsistent initial warm timing for sample {sample}")
        warm[sample] = row["image_prefill_seconds"]
        if version == "1.1":
            setup_key = (sample, condition)
            if (setup_key in setup
                    and setup[setup_key] != row["initial_cache_setup_seconds"]):
                raise ValueError(f"inconsistent initial cache setup timing for {setup_key}")
            setup[setup_key] = row["initial_cache_setup_seconds"]

    for question, group in by_question.items():
        if set(group) != set(CONDITIONS):
            raise ValueError(f"incomplete condition set for {question}")
        steps = {row["step"] for row in group.values()}
        if len(steps) != 1:
            raise ValueError(f"mixed steps for {question}")
        full = group["full"]
        for row in group.values():
            if not math.isclose(row["full_em"], full["em"], abs_tol=1e-9):
                raise ValueError(f"inconsistent full_em for {question}")
            if not math.isclose(row["full_anls"], full["anls"], abs_tol=1e-9):
                raise ValueError(f"inconsistent full_anls for {question}")
            expected = row["em"] if full["em"] == 1.0 else None
            retained = row.get("full_correct_retained")
            if retained != expected:
                raise ValueError(f"inconsistent full_correct_retained for {question}")


def _mean(rows: list[dict], getter) -> float | None:
    values = [getter(row) for row in rows]
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _aggregate(rows: list[dict]) -> dict[str, Any]:
    retained = [row["full_correct_retained"] for row in rows
                if row["full_correct_retained"] is not None]
    selection = lambda row, key: row["selection_after"].get(key)
    return {
        "n": len(rows), "em": _mean(rows, lambda r: r["em"]),
        "anls": _mean(rows, lambda r: r["anls"]),
        "loyalty": _mean(rows, lambda r: r["loyalty"]),
        "retention": sum(retained) / len(retained) if retained else None,
        "retention_n": len(retained),
        "entered_mean": _mean(rows, lambda r: r["entered_tokens"]),
        "entered_total": sum(r["entered_tokens"] for r in rows),
        "history_text": _mean(rows, lambda r: selection(r, "selected_history_text_tokens")),
        "selected_image": _mean(rows, lambda r: selection(r, "selected_image_tokens")),
        "prefix_control": _mean(rows, lambda r: selection(r, "selected_prefix_control_tokens")),
        "kept_count": _mean(rows, lambda r: selection(r, "kept_count")),
        "image_weight": _mean(rows, lambda r: selection(r, "image_weight")),
        "history_weight": _mean(rows, lambda r: selection(r, "history_weight")),
        **{field: _mean(rows, lambda r, f=field: r.get(f)) for field in (
            "cold_kv_bytes", "active_history_kv_bytes", "peak_active_kv_bytes",
            "peak_gpu_allocated_bytes", "selector_state_bytes", "combined_kv_and_state_bytes",
            "retained_kv_tokens", "retained_kv_bytes", "resident_gpu_kv_bytes",
            "logical_history_tokens_after", "initial_deleted_tokens",
            "deleted_tokens_this_turn", "deleted_image_tokens_this_turn",
            "session_metadata_bytes", "persistent_session_tensor_bytes",
            "compaction_peak_kv_bytes_upper_bound", "h2d_kv_bytes", "d2h_new_kv_bytes",
            "load_seconds", "ttft_seconds", "turn_seconds")},
    }


def summarize(metadata: dict, rows: list[dict]) -> dict[str, Any]:
    # Public callers get the same fail-closed contract as the CLI loader.
    validate(metadata, rows)
    samples = sorted({row["sample_id"] for row in rows})
    questions = {(row["sample_id"], row["question_id"]) for row in rows}
    warm = {sample: next(row["image_prefill_seconds"] for row in rows
                         if row["sample_id"] == sample) for sample in samples}
    setup = {condition: [next(row["initial_cache_setup_seconds"] for row in rows
                              if row["sample_id"] == sample
                              and row["condition_id"] == condition)
                         for sample in samples] for condition in CONDITIONS}
    overall = {condition: _aggregate([r for r in rows if r["condition_id"] == condition])
               for condition in CONDITIONS}
    turns = {step: {condition: _aggregate([
        r for r in rows if r["step"] == step and r["condition_id"] == condition])
        for condition in CONDITIONS} for step in sorted({r["step"] for r in rows})}
    return {"run_id": metadata["run_id"], "model": metadata["model"],
            "schema_version": metadata["schema_version"],
            "storage_mode": _storage_mode(metadata),
            "datasets": sorted({r["dataset"] for r in rows}), "samples": len(samples),
            "questions": len(questions), "rows": len(rows), "overall": overall,
            "turns": turns, "warm_total": sum(warm.values()),
            "warm_mean": sum(warm.values()) / len(warm),
            "setup_mean": {condition: (sum(values) / len(values) if values[0] is not None else None)
                           for condition, values in setup.items()}}


def _fmt(value: float | None, scale: float = 1.0, digits: int = 3) -> str:
    return "—" if value is None else f"{value / scale:.{digits}f}"


def render_markdown(summary: dict[str, Any], input_name: str = "") -> str:
    lines = ["# Recurrent session summary", "",
             f"- Input: `{input_name}`" if input_name else "- Input: in-memory rows",
             f"- Run/model: `{summary['run_id']}` / `{summary['model']}`",
             f"- Schema/storage: `{summary['schema_version']}` / `{summary['storage_mode']}`",
             f"- Datasets: {', '.join(summary['datasets'])}",
             f"- Samples/questions/rows: {summary['samples']} / {summary['questions']} / {summary['rows']}",
             f"- Initial shared image-prefill warm time: total {summary['warm_total']*1000:.1f} ms; "
             f"mean {summary['warm_mean']*1000:.1f} ms over {summary['samples']} samples"]
    if summary["schema_version"] == "1.1":
        setup = ", ".join(f"{condition}={_fmt(value, 0.001, 1)} ms"
                          for condition, value in summary["setup_mean"].items())
        lines += [f"- Initial cache setup mean: {setup}"]
    lines.append("")
    if summary["samples"] == 1:
        lines += ["> **n=1 is a smoke/validation run, not efficacy evidence.**", ""]
    storage_note = (
        "> Delete mode physically retains only selected K/V for compressed conditions; evicted K/V and per-token state are irreversible. FULL retains its complete own history, and no CPU cold reservoir is kept."
        if summary["storage_mode"] == "delete" else
        "> Offload mode retains every K/V uncompressed in the CPU cold reservoir. Its selected GPU set is a working set, not total-storage compression."
    )
    if summary["schema_version"] == "1.0":
        storage_note += " Schema 1.0 is reported with derived legacy-offload storage fields."
    lines += [
        "> Each condition follows its own generated-answer history. FULL is an independent own-history baseline, not a matched-history causal ablation.",
        storage_note,
        "> Process-GPU peak is an absolute process measurement: it includes the shared model and other conditions' resident caches, so it is not method-only memory.",
        "> The recurrent gate is a training-free heuristic, not a learned LSTM.",
        "> Same-image paired bootstrap is intentionally omitted in this minimal analyzer. All values are descriptive; no statistical-significance claim is made.", "",
        "## Overall task metrics", "",
        "| Condition | n | EM | ANLS | Loyalty to FULL | Full-correct EM retention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = summary["overall"][condition]
        retained = f"{_fmt(item['retention'])} ({item['retention_n']}/{item['n']})"
        lines.append(f"| {condition} | {item['n']} | {_fmt(item['em'])} | {_fmt(item['anls'])} | "
                     f"{_fmt(item['loyalty'])} | {retained} |")
    lines += ["", "## Metrics by turn", "",
              "| Turn | Condition | n | EM | ANLS | Full-correct EM retention |",
              "|---:|---|---:|---:|---:|---:|"]
    for step, groups in summary["turns"].items():
        for condition in CONDITIONS:
            item = groups[condition]
            retained = f"{_fmt(item['retention'])} ({item['retention_n']}/{item['n']})"
            lines.append(f"| {step} | {condition} | {item['n']} | {_fmt(item['em'])} | {_fmt(item['anls'])} | {retained} |")
    lines += ["", "## Recurrent working-set composition (means)", "",
              "| Turn | n | selected image | selected history text | selected prefix control | kept count | image/history weight | entered mean/total |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for step, groups in summary["turns"].items():
        item = groups["recurrent"]
        weights = f"{_fmt(item['image_weight'])}/{_fmt(item['history_weight'])}"
        lines.append(f"| {step} | {item['n']} | {_fmt(item['selected_image'], digits=2)} | "
                     f"{_fmt(item['history_text'], digits=2)} | {_fmt(item['prefix_control'], digits=2)} | "
                     f"{_fmt(item['kept_count'], digits=2)} | {weights} | "
                     f"{_fmt(item['entered_mean'], digits=2)}/{item['entered_total']} |")
    lines += ["", "## Persistent storage and deletion by turn (means)", "",
              "| Turn | Condition | logical tokens | retained KV tokens/MiB | image remaining | CPU cold MiB | resident GPU KV MiB | initial deleted | deleted this turn/image | persistent tensors MiB |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for step, groups in summary["turns"].items():
        for condition in CONDITIONS:
            item = groups[condition]
            retained = f"{_fmt(item['retained_kv_tokens'], digits=2)}/{_fmt(item['retained_kv_bytes'], 2**20, 2)}"
            deleted = f"{_fmt(item['deleted_tokens_this_turn'], digits=2)}/{_fmt(item['deleted_image_tokens_this_turn'], digits=2)}"
            lines.append(
                f"| {step} | {condition} | {_fmt(item['logical_history_tokens_after'], digits=2)} | {retained} | "
                f"{_fmt(item['selected_image'], digits=2)} | {_fmt(item['cold_kv_bytes'], 2**20, 2)} | "
                f"{_fmt(item['resident_gpu_kv_bytes'], 2**20, 2)} | {_fmt(item['initial_deleted_tokens'], digits=2)} | "
                f"{deleted} | {_fmt(item['persistent_session_tensor_bytes'], 2**20, 2)} |")
    lines += ["", "## Transient memory and timing by turn (means)", "",
              "| Turn | Condition | hot input MiB | active-KV peak MiB | compaction upper MiB | H2D/D2H MiB | process-GPU peak MiB* | selector/session metadata KiB | combined upper MiB | load / TTFT / turn ms |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for step, groups in summary["turns"].items():
        for condition in CONDITIONS:
            item = groups[condition]
            timing = "/".join(_fmt(item[name], 0.001, 1)
                              for name in ("load_seconds", "ttft_seconds", "turn_seconds"))
            transfer = f"{_fmt(item['h2d_kv_bytes'], 2**20, 2)}/{_fmt(item['d2h_new_kv_bytes'], 2**20, 2)}"
            metadata = f"{_fmt(item['selector_state_bytes'], 2**10, 2)}/{_fmt(item['session_metadata_bytes'], 2**10, 2)}"
            lines.append(
                f"| {step} | {condition} | {_fmt(item['active_history_kv_bytes'], 2**20, 2)} | "
                f"{_fmt(item['peak_active_kv_bytes'], 2**20, 2)} | "
                f"{_fmt(item['compaction_peak_kv_bytes_upper_bound'], 2**20, 2)} | {transfer} | "
                f"{_fmt(item['peak_gpu_allocated_bytes'], 2**20, 2)} | {metadata} | "
                f"{_fmt(item['combined_kv_and_state_bytes'], 2**20, 2)} | {timing} |")
    lines += ["", "\\* Process-GPU peak has process scope, not per-method scope."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.input) if Path(args.input).is_absolute() else ROOT / args.input
    target = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    metadata, rows = load_jsonl(source)
    text = render_markdown(summarize(metadata, rows), args.input)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(text)
    print(f"[saved] {target}")


if __name__ == "__main__":
    main()
