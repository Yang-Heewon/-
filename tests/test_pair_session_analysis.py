"""Self-contained fixtures for the physical-pair report contract."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from vlm_diagnosis.scripts.pair_session_analysis import load_jsonl, render, validate


def _snapshot(counts, logical, text=0):
    total = sum(counts)
    return {
        "n_layers": 2, "n_heads": 2, "groups": 4,
        "resident_pairs": total, "counted_pairs": total, "budget_pairs": 8,
        "pairs_by_group": counts,
        "pairs_by_layer_head": [counts[:2], counts[2:]],
        "pairs_by_layer": [sum(counts[:2]), sum(counts[2:])],
        "pairs_by_head": [counts[0] + counts[2], counts[1] + counts[3]],
        "distinct_logical_tokens": logical,
        "modality_pair_counts": {"image": total - text, "text": text},
        "state_bytes": 34 * total,
    }


def fixture():
    metadata = {
        "record_type": "run_metadata", "schema_version": "2.0", "stage": "RECURRENT_PAIRS",
        "granularity": "kv_pair", "storage_mode": "delete", "budget": .5,
        "conditions": "full,image_static,recurrent", "model": "tiny", "run_id": "fixture",
    }
    rows = []
    prior = {"full": _snapshot([4] * 4, 4),
             "image_static": _snapshot([3, 1, 2, 2], 4),
             "recurrent": _snapshot([3, 1, 2, 2], 4)}
    for step in (1, 2):
        for c in ("full", "image_static", "recurrent"):
            before = prior[c]
            if c == "full":
                after = _snapshot([4 + step] * 4, 4 + step, text=4 * step)
            elif c == "image_static":
                after = deepcopy(before)
            else:
                after = _snapshot([3 - step, 1 + step, 2, 2], 4 + step, text=step)
            old, kept = before["resident_pairs"], after["resident_pairs"]
            rows.append({
                "run_id": "fixture", "model": "tiny", "condition_id": c,
                "granularity": "kv_pair", "storage_mode": "delete", "compression_applied": c != "full",
                "dataset": "synthetic", "sample_id": "one", "question_id": str(step), "step": step,
                "initial_prefix_tokens": 4, "initial_kv_pairs": 16, "budget_pairs": 8,
                "active_history_pairs": old, "retained_kv_pairs": kept, "retained_kv_bytes": kept * 64,
                "peak_active_kv_pairs": old + 4, "peak_active_kv_bytes": (old + 4) * 64,
                "cache_storage_peak_bytes_upper_bound": 2 * (old + 4) * 64,
                "new_session_tokens": 1, "logical_history_tokens_after": 4 + step,
                "initial_deleted_pairs": 0 if c == "full" else 8,
                "deleted_pairs_this_turn": old + 4 - kept,
                "entered_pairs": 4 if c == "full" else int(c == "recurrent"),
                "evicted_pairs": int(c == "recurrent"),
                "selector_state_bytes": kept * 34, "session_metadata_bytes": kept * 8 + 64,
                "persistent_session_tensor_bytes": kept * (64 + 34 + 8) + 64,
                "cold_kv_bytes": 0, "h2d_kv_bytes": 0, "d2h_new_kv_bytes": 0,
                "retained_kv_fraction_of_initial": kept / 16,
                "selection_before": deepcopy(before), "selection_after": deepcopy(after),
                "prediction": "yes", "gold": ["yes"], "em": 1., "anls": 1.,
                "full_em": 1., "full_anls": 1., "loyalty": 1., "full_correct_retained": 1.,
            })
            prior[c] = after
    return metadata, rows


class PairSessionAnalysisTest(unittest.TestCase):
    def test_valid_reallocated_heads_and_report(self):
        metadata, rows = fixture()
        validate(metadata, rows)
        report = render(metadata, rows)
        self.assertIn("paired turns: 2", report)
        self.assertIn("2–2", report)
        self.assertIn("1–3", report)
        self.assertIn("not a matched-history selector ablation", report)

    def test_load_jsonl_round_trip(self):
        metadata, rows = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in [metadata, *rows]), encoding="utf-8")
            self.assertEqual(load_jsonl(path), (metadata, rows))

    def test_rejects_legacy_or_incomplete_comparison(self):
        for field, value in (("schema_version", "1.1"), ("granularity", "token"),
                             ("storage_mode", "offload"), ("conditions", "recurrent")):
            with self.subTest(field=field):
                metadata, rows = fixture()
                metadata[field] = value
                with self.assertRaises(ValueError):
                    validate(metadata, rows)
        metadata, rows = fixture()
        with self.assertRaisesRegex(ValueError, "missing paired condition"):
            validate(metadata, rows[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate(metadata, rows + [deepcopy(rows[0])])

    def test_rejects_corrupt_pair_and_storage_accounting(self):
        for field, value in (("budget_pairs", 9), ("retained_kv_pairs", 9),
                             ("retained_kv_bytes", 513), ("deleted_pairs_this_turn", 0),
                             ("entered_pairs", 7), ("cold_kv_bytes", 1),
                             ("selector_state_bytes", 8 * 18), ("persistent_session_tensor_bytes", 0),
                             ("retained_kv_fraction_of_initial", float("nan")), ("step", True)):
            with self.subTest(field=field):
                metadata, rows = fixture()
                rows[2][field] = value
                with self.assertRaises(ValueError):
                    validate(metadata, rows)

    def test_rejects_head_totals_and_static_movement(self):
        for field, value in (("pairs_by_group", [1, 1, 2, 2]), ("pairs_by_layer", [8, 0]),
                             ("pairs_by_head", [8, 0]), ("pairs_by_layer_head", [[8], [0]]),
                             ("modality_pair_counts", {"image": 7}), ("distinct_logical_tokens", 1)):
            with self.subTest(field=field):
                metadata, rows = fixture()
                rows[2]["selection_after"][field] = value
                with self.assertRaises(ValueError):
                    validate(metadata, rows)
        metadata, rows = fixture()
        rows[1]["selection_after"] = _snapshot([2, 2, 2, 2], 4)
        with self.assertRaisesRegex(ValueError, "static selection changed"):
            validate(metadata, rows)

    def test_rejects_temporal_discontinuity(self):
        metadata, rows = fixture()
        # Internally consistent counts, but they do not match the preceding turn.
        rows[5]["selection_before"] = _snapshot([3, 1, 2, 2], 5, text=1)
        rows[5]["selection_after"] = _snapshot([2, 2, 2, 2], 6, text=2)
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            validate(metadata, rows)

    def test_rejects_wrong_full_references_and_metrics(self):
        for field, value in (("full_em", 0), ("full_correct_retained", None), ("loyalty", 0),
                             ("em", 0), ("anls", .5), ("question_id", "different")):
            with self.subTest(field=field):
                metadata, rows = fixture()
                rows[2][field] = value
                with self.assertRaises(ValueError):
                    validate(metadata, rows)

    def test_no_full_correct_denominator_is_na(self):
        metadata, rows = fixture()
        for row in rows:
            row.update(prediction="no", em=0., anls=0., full_em=0., full_anls=0., full_correct_retained=None)
        self.assertIn("| N/A |", render(metadata, rows))


if __name__ == "__main__":
    unittest.main()
