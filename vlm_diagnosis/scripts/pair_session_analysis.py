"""Validate and summarize schema-2.0 globally budgeted physical KV-pair runs.

This is intentionally separate from the legacy common-token/offload analyzer.
Counts cannot establish pair identity: no-resurrection is checked in cache tests.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean

from vlm_diagnosis.core.metrics import anls, exact_match, normalize_text


CONDITIONS = ("full", "image_static", "recurrent")
ROOT = Path(__file__).resolve().parents[2]


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _number(value, name, *, integer=False, high=None):
    _require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{name}: expected number")
    _require(math.isfinite(value) and value >= 0 and (high is None or value <= high), f"{name}: out of range")
    _require(not integer or isinstance(value, int), f"{name}: expected integer")
    return value


def _snapshot(s, count, budget, logical):
    _require(isinstance(s, dict), "selection must be an object")
    for name in ("n_layers", "n_heads", "groups", "resident_pairs", "counted_pairs",
                 "budget_pairs", "distinct_logical_tokens", "state_bytes"):
        _number(s.get(name), name, integer=True)
    layers, heads = s["n_layers"], s["n_heads"]
    _require(layers > 0 and heads > 0 and s["groups"] == layers * heads, "invalid layer/head dimensions")
    counts = s.get("pairs_by_group")
    _require(isinstance(counts, list) and len(counts) == layers * heads, "invalid head-count vector")
    for value in counts:
        _number(value, "head count", integer=True, high=logical)
    _require(sum(counts) == count == s["resident_pairs"] == s["counted_pairs"], "pair-count mismatch")
    _require(s["budget_pairs"] == budget and s["state_bytes"] == count * 34, "selector-state accounting mismatch")
    matrix = [counts[i:i + heads] for i in range(0, len(counts), heads)]
    _require(s.get("pairs_by_layer_head") == matrix, "layer/head matrix mismatch")
    _require(s.get("pairs_by_layer") == list(map(sum, matrix)), "layer totals mismatch")
    _require(s.get("pairs_by_head") == [sum(col) for col in zip(*matrix)], "head totals mismatch")
    modalities = s.get("modality_pair_counts")
    _require(isinstance(modalities, dict) and all(isinstance(k, str) for k in modalities), "invalid modalities")
    for value in modalities.values():
        _number(value, "modality pair count", integer=True)
    _require(sum(modalities.values()) == count, "modality totals mismatch")
    _require(max(counts) <= s["distinct_logical_tokens"] <= min(logical, count), "invalid distinct-token count")
    return counts


def validate(metadata, rows):
    _require(metadata.get("schema_version") == "2.0" and metadata.get("stage") == "RECURRENT_PAIRS",
             "requires schema 2.0 RECURRENT_PAIRS; token/offload logs use recurrent_session_analysis")
    _require(metadata.get("granularity") == "kv_pair" and metadata.get("storage_mode") == "delete",
             "requires physically deleted KV pairs")
    conditions = metadata.get("conditions", "").split(",")
    _require(len(conditions) == 3 and set(conditions) == set(CONDITIONS), "comparison requires all three conditions")
    ratio = _number(metadata.get("budget"), "budget", high=1)
    _require(ratio > 0 and rows, "empty run or zero budget")
    _require(isinstance(metadata.get("run_id"), str) and metadata["run_id"]
             and isinstance(metadata.get("model"), str) and metadata["model"], "missing run/model identity")
    paired, sessions = defaultdict(dict), defaultdict(list)
    integer_fields = (
        "step", "initial_prefix_tokens", "initial_kv_pairs", "budget_pairs", "active_history_pairs",
        "retained_kv_pairs", "retained_kv_bytes", "peak_active_kv_pairs", "peak_active_kv_bytes",
        "cache_storage_peak_bytes_upper_bound", "new_session_tokens", "logical_history_tokens_after",
        "initial_deleted_pairs", "deleted_pairs_this_turn", "entered_pairs", "evicted_pairs",
        "selector_state_bytes", "session_metadata_bytes", "persistent_session_tensor_bytes",
        "cold_kv_bytes", "h2d_kv_bytes", "d2h_new_kv_bytes",
    )
    for r in rows:
        c = r.get("condition_id")
        _require(c in CONDITIONS and r.get("run_id") == metadata.get("run_id") and r.get("model") == metadata.get("model"),
                 "condition/run/model mismatch")
        _require(r.get("granularity") == "kv_pair" and r.get("storage_mode") == "delete"
                 and r.get("compression_applied") is (c != "full"), "row storage semantics mismatch")
        for field in integer_fields:
            _number(r.get(field), field, integer=True)
        for field in ("em", "anls", "full_em", "full_anls", "loyalty"):
            _number(r.get(field), field, high=1)
        _number(r.get("retained_kv_fraction_of_initial"), "retained fraction")
        p, b, logical, fresh = (r[k] for k in ("initial_prefix_tokens", "budget_pairs",
                                             "logical_history_tokens_after", "new_session_tokens"))
        old, kept, peak = (r[k] for k in ("active_history_pairs", "retained_kv_pairs", "peak_active_kv_pairs"))
        _require(p > 0 and b > 0 and kept > 0 and r["step"] > 0 and logical >= p + fresh, "invalid session lengths")
        before = _snapshot(r.get("selection_before"), old, b, logical - fresh)
        after = _snapshot(r.get("selection_after"), kept, b, logical)
        _require((r["selection_before"]["n_layers"], r["selection_before"]["n_heads"])
                 == (r["selection_after"]["n_layers"], r["selection_after"]["n_heads"]), "head dimensions changed")
        groups = len(before)
        _require(r["initial_kv_pairs"] == p * groups and b == round(p * groups * ratio), "global budget mismatch")
        _require(peak == old + groups * fresh and r["deleted_pairs_this_turn"] == peak - kept, "deletion accounting mismatch")
        _require(kept == old + r["entered_pairs"] - r["evicted_pairs"]
                 and r["entered_pairs"] <= groups * fresh and r["evicted_pairs"] <= old,
                 "turnover accounting mismatch")
        _require(all(y <= x + fresh for x, y in zip(before, after)), "head gains exceed fresh candidate pairs")
        pair_bytes, rem = divmod(r["retained_kv_bytes"], kept)
        _require(pair_bytes > 0 and rem == 0 and r["peak_active_kv_bytes"] == peak * pair_bytes,
                 "KV byte accounting mismatch")
        _require(r["cache_storage_peak_bytes_upper_bound"] == 2 * peak * pair_bytes, "cache peak bound mismatch")
        _require(r["selector_state_bytes"] == kept * 34 and r["session_metadata_bytes"] >= kept * 8,
                 "state/metadata byte accounting mismatch")
        _require(r["persistent_session_tensor_bytes"] == r["retained_kv_bytes"] + r["selector_state_bytes"] + r["session_metadata_bytes"],
                 "persistent byte accounting mismatch")
        _require(all(r[k] == 0 for k in ("cold_kv_bytes", "h2d_kv_bytes", "d2h_new_kv_bytes")), "hidden cold/transfer KV")
        _require(math.isclose(r["retained_kv_fraction_of_initial"], kept / (p * groups)), "retained fraction mismatch")
        _require(r["initial_deleted_pairs"] == (0 if c == "full" else p * groups - b), "initial deletion mismatch")
        if c == "full":
            _require(before == [logical - fresh] * groups and after == [logical] * groups
                     and r["evicted_pairs"] == 0, "FULL is not a full cache")
        else:
            _require(old == kept == b, "compressed cache violates exact global pair budget")
        if c == "image_static":
            _require(r["selection_before"] == r["selection_after"]
                     and r["entered_pairs"] == r["evicted_pairs"] == 0, "static selection changed")
        _require(isinstance(r.get("sample_id"), str) and isinstance(r.get("dataset"), str)
                 and isinstance(r.get("question_id"), (str, int)), "invalid sample/question identity")
        _require(isinstance(r.get("prediction"), str) and isinstance(r.get("gold"), list)
                 and r["gold"] and all(isinstance(g, str) for g in r["gold"]), "invalid prediction/gold")
        _require(r["em"] == exact_match(r["prediction"], r["gold"])
                 and math.isclose(r["anls"], anls(r["prediction"], r["gold"])), "answer metrics mismatch")
        key = (r["dataset"], r["sample_id"], r["step"])
        _require(c not in paired[key], "duplicate condition/turn")
        paired[key][c] = r
        sessions[(r["dataset"], r["sample_id"], c)].append(r)
    for group in paired.values():
        _require(set(group) == set(CONDITIONS), "missing paired condition")
        full = group["full"]
        for r in group.values():
            for field in ("question_id", "gold", "initial_prefix_tokens", "initial_kv_pairs", "budget_pairs"):
                _require(r[field] == full[field], f"paired {field} mismatch")
            _require(r["retained_kv_bytes"] // r["retained_kv_pairs"]
                     == full["retained_kv_bytes"] // full["retained_kv_pairs"], "paired KV-pair byte cost mismatch")
            _require(r["full_em"] == full["em"] and r["full_anls"] == full["anls"], "FULL metric reference mismatch")
            _require("full_correct_retained" in r and r["full_correct_retained"] == (r["em"] if full["em"] == 1 else None),
                     "FULL-correct retention mismatch")
            _require(r["loyalty"] == float(normalize_text(r["prediction"]) == normalize_text(full["prediction"])),
                     "answer agreement mismatch")
    for group in sessions.values():
        group.sort(key=lambda r: r["step"])
        _require([r["step"] for r in group] == list(range(1, len(group) + 1)), "noncontiguous session steps")
        _require(group[0]["logical_history_tokens_after"] - group[0]["new_session_tokens"] == group[0]["initial_prefix_tokens"],
                 "initial logical clock mismatch")
        for previous, current in zip(group, group[1:]):
            for field in ("initial_prefix_tokens", "initial_kv_pairs", "budget_pairs"):
                _require(previous[field] == current[field], f"session {field} changed")
            _require(previous["retained_kv_bytes"] // previous["retained_kv_pairs"]
                     == current["retained_kv_bytes"] // current["retained_kv_pairs"], "session KV-pair byte cost changed")
            _require(previous["selection_after"] == current["selection_before"], "selection-state discontinuity")
            _require(previous["logical_history_tokens_after"] + current["new_session_tokens"] == current["logical_history_tokens_after"],
                     "logical clock discontinuity")


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    _require(all(isinstance(r, dict) for r in records), "records must be objects")
    metadata = [r for r in records if r.get("record_type") == "run_metadata"]
    rows = [r for r in records if r.get("record_type") != "run_metadata"]
    _require(len(metadata) == 1, "requires exactly one metadata row")
    validate(metadata[0], rows)
    return metadata[0], rows


def render(metadata, rows):
    validate(metadata, rows)
    samples = {(r["dataset"], r["sample_id"]) for r in rows}
    lines = ["# Physical global KV-pair session validation", "",
             f"Model: {metadata['model']}; images: {len(samples)}; paired turns: {len(rows) // 3}.",
             "Unit: one (layer, KV head, logical token) K/V vector pair. Head quotas are not fixed.", "",
             "| Condition | EM | ANLS | FULL-correct retention | Answer agreement | Mean stored KV MiB |",
             "|---|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        group = [r for r in rows if r["condition_id"] == c]
        retained = [r["full_correct_retained"] for r in group if r["full_correct_retained"] is not None]
        retention = f"{mean(retained):.3f}" if retained else "N/A"
        lines.append(f"| {c} | {mean(r['em'] for r in group):.3f} | {mean(r['anls'] for r in group):.3f} | "
                     f"{retention} | {mean(r['loyalty'] for r in group):.3f} | "
                     f"{mean(r['retained_kv_bytes'] for r in group) / 2**20:.3f} |")
    lines += ["", "## Recurrent allocation by turn", "",
              "| Image | Turn | Global B | Head min–max | Distinct token IDs | Image pairs | Text pairs | Deleted this turn | Persistent tensors MiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["condition_id"] != "recurrent":
            continue
        s = r["selection_after"]
        counts, modalities = s["pairs_by_group"], s["modality_pair_counts"]
        label = (r["dataset"] + ":" + r["sample_id"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {label} | {r['step']} | {r['budget_pairs']:,} | {min(counts)}–{max(counts)} | "
                     f"{s['distinct_logical_tokens']:,} | {modalities.get('image', 0):,} | {modalities.get('text', 0):,} | "
                     f"{r['deleted_pairs_this_turn']:,} | {r['persistent_session_tensor_bytes'] / 2**20:.3f} |")
    lines += ["", "## Scope and limits", "",
              "- Exact global budgets, head/modality totals, physical-byte accounting, zero cold KV and turn continuity pass validation.",
              "- Logs contain counts, not every retained pair ID. Irreversible deletion and non-shared storage are checked by cache/session tests, not inferred from these counts.",
              "- The retained fraction is relative to the initial full prefix, not the growing FULL cache. Current-turn KV and compaction copies require extra temporary storage; the initial prefill is full.",
              "- Persistent tensors include KV, 34-byte per-pair selector state, 8-byte per-pair cache IDs and template tensors; not model weights, activations, Python objects or allocator reserve.",
              "- GPU allocation peaks include the shared model and other condition caches. They are not isolated per-method memory measurements.",
              "- All conditions use the same Python ragged eager reference backend. This is not a fused-kernel throughput claim.",
              "- Each condition generates its own answer history. This is not a matched-history selector ablation or a KVZIP comparison.",
              "- Small smoke results establish execution only, not general accuracy retention, statistical significance or novelty.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metadata, rows = load_jsonl(ROOT / args.input)
    report = render(metadata, rows)
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(f"[saved] {target}")


if __name__ == "__main__":
    main()
