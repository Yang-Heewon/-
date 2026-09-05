import copy
import json
from pathlib import Path
import tempfile
import unittest

from vlm_diagnosis.scripts.recurrent_session_analysis import (
    load_jsonl,
    render_markdown,
    summarize,
)


META = {"record_type": "run_metadata", "schema_version": "1.0",
        "stage": "RECURRENT_SESSION", "run_id": "run-1", "model": "qwen25vl",
        "conditions": "full,image_static,recurrent"}
META_DELETE = {**META, "schema_version": "1.1", "storage": "delete",
               "storage_mode": "delete"}


def row(question, step, condition, em, anls, full_em, history=0, entered=0):
    weight = {"full": (None, None), "image_static": (1.0, 0.0),
              "recurrent": (0.5, 0.5)}[condition]
    historical = 10 + 2 * (step - 1)
    active = historical if condition == "full" else 4
    next_active = historical + 2 if condition == "full" else 4
    return {"run_id": "run-1", "model": "qwen25vl", "sample_id": "sample-1",
            "dataset": "ScreenQA", "question_id": question, "condition_id": condition,
            "step": step, "em": em, "anls": anls, "full_em": full_em,
            "full_anls": 1.0 if full_em == 1 else 0.2,
            "loyalty": 1.0 if condition == "full" else 0.5,
            "full_correct_retained": em if full_em == 1 else None,
            "entered_tokens": entered, "cold_kv_bytes": 8 * 2**20,
            "active_history_kv_bytes": 2 * 2**20, "peak_active_kv_bytes": 3 * 2**20,
            "peak_gpu_allocated_bytes": 10 * 2**20, "selector_state_bytes": 2**10,
            "combined_kv_and_state_bytes": 11 * 2**20 + 2**10,
            "h2d_kv_bytes": 2 * 2**20, "d2h_new_kv_bytes": 2**20,
            "historical_tokens": historical, "active_history_tokens": active,
            "next_active_history_tokens": next_active,
            "peak_active_kv_tokens": active + 2, "new_session_tokens": 2,
            "initial_prefix_tokens": 10, "budget_tokens": 4,
            "load_seconds": 0.01, "ttft_seconds": 0.02, "turn_seconds": 0.03,
            "image_prefill_seconds": 0.1,
            "selection_after": {"selected_image_tokens": 6 - history,
                                "selected_history_text_tokens": history,
                                "selected_prefix_control_tokens": 4, "kept_count": 10,
                                "image_weight": weight[0], "history_weight": weight[1]}}


def fixture_rows():
    rows = []
    for condition, em, anls in (("full", 1, 1), ("image_static", 1, 0.8),
                                ("recurrent", 0, 0.4)):
        rows.append(row("q1", 1, condition, em, anls, 1,
                        history=2 if condition == "recurrent" else 0,
                        entered=2 if condition == "recurrent" else 0))
    for condition, em, anls in (("full", 0, 0.2), ("image_static", 1, 0.7),
                                ("recurrent", 1, 0.9)):
        rows.append(row("q2", 2, condition, em, anls, 0,
                        history=4 if condition == "recurrent" else 0,
                        entered=1 if condition == "recurrent" else 0))
    return rows


def v11_rows(mode="delete"):
    rows = []
    for step, question in ((1, "q1"), (2, "q2")):
        historical, new = 10 + 2 * (step - 1), 2
        for condition in ("full", "image_static", "recurrent"):
            if condition == "full":
                active, next_active = list(range(historical)), list(range(historical + new))
                before_image, after_image, control = 3, 3, 7
            elif condition == "image_static":
                active = next_active = [0, 1, 2, 3]
                before_image = after_image = 3
                control = 1
            elif step == 1:
                active, next_active = [0, 1, 2, 3], [0, 1, 10, 11]
                before_image, after_image, control = 3, 1, 1
            else:
                active, next_active = [0, 1, 10, 11], [0, 10, 12, 13]
                before_image, after_image, control = 1, 0, 1
            full_em = 1 if step == 1 else 0
            em = full_em if condition == "full" else int(condition == "image_static")
            anls = (1.0 if step == 1 else 0.2) if condition == "full" else 0.8
            result = row(question, step, condition, em, anls, full_em)
            logical = historical + new
            peak = len(active) + new
            retained_tokens = ((logical if condition == "full" else 4)
                               if mode == "delete" else logical)
            token_bytes = 100
            state_bytes, metadata_bytes = 44, 128
            retained_bytes = retained_tokens * token_bytes
            deleted = peak - retained_tokens if mode == "delete" and condition != "full" else 0
            compaction = ((peak + 2 * retained_tokens) * token_bytes
                          if mode == "delete" and condition != "full"
                          else peak * token_bytes)
            result.update({
                "storage_mode": mode,
                "compression_applied": mode == "delete" and condition != "full",
                "historical_tokens": historical, "new_session_tokens": new,
                "logical_history_tokens_after": logical,
                "active_history_tokens": len(active),
                "next_active_history_tokens": len(next_active),
                "peak_active_kv_tokens": peak,
                "active_history_kv_bytes": len(active) * token_bytes,
                "peak_active_kv_bytes": peak * token_bytes,
                "retained_kv_tokens": retained_tokens,
                "retained_kv_bytes": retained_bytes,
                "retained_kv_fraction_of_initial": retained_tokens / 10,
                "cold_kv_bytes": retained_bytes if mode == "offload" else 0,
                "resident_gpu_kv_bytes": retained_bytes if mode == "delete" else 0,
                "h2d_kv_bytes": len(active) * token_bytes if mode == "offload" else 0,
                "d2h_new_kv_bytes": new * token_bytes if mode == "offload" else 0,
                "initial_deleted_tokens": 6 if mode == "delete" and condition != "full" else 0,
                "deleted_tokens_this_turn": deleted,
                "deleted_image_tokens_this_turn": (before_image - after_image
                                                    if mode == "delete" else 0),
                "selector_state_bytes": state_bytes,
                "session_metadata_bytes": metadata_bytes,
                "persistent_session_tensor_bytes": retained_bytes + state_bytes + metadata_bytes,
                "compaction_peak_kv_bytes_upper_bound": compaction,
                "combined_kv_and_state_bytes": ((0 if mode == "delete" else retained_bytes)
                                                + compaction + state_bytes),
                "initial_cache_setup_seconds": 0.01,
                "active_indices": active, "next_active_indices": next_active,
                "entered_tokens": len(set(next_active) - set(active)),
                "selection_before": {"selected_image_tokens": before_image},
            })
            history = len([item for item in next_active if item >= 10])
            result["selection_after"].update({
                "selected_image_tokens": after_image,
                "selected_prefix_control_tokens": control,
                "selected_history_text_tokens": history,
                "kept_count": len(next_active),
            })
            rows.append(result)
    return rows


class RecurrentSessionAnalysisTest(unittest.TestCase):
    def test_metrics_dynamics_warm_timing_and_required_caveats(self):
        summary = summarize(copy.deepcopy(META), fixture_rows())
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["questions"], 2)
        self.assertAlmostEqual(summary["warm_total"], 0.1)
        self.assertAlmostEqual(summary["overall"]["full"]["em"], 0.5)
        self.assertAlmostEqual(summary["overall"]["recurrent"]["anls"], 0.65)
        self.assertAlmostEqual(summary["overall"]["recurrent"]["loyalty"], 0.5)
        self.assertEqual(summary["overall"]["image_static"]["retention"], 1.0)
        self.assertEqual(summary["overall"]["recurrent"]["retention"], 0.0)
        self.assertEqual(summary["turns"][1]["recurrent"]["entered_total"], 2)
        self.assertEqual(summary["turns"][2]["recurrent"]["history_text"], 4.0)
        markdown = render_markdown(summary, "fixture.jsonl")
        for phrase in ("n=1 is a smoke/validation run, not efficacy evidence",
                       "own-history baseline", "retains every K/V uncompressed",
                       "training-free heuristic", "no statistical-significance claim"):
            self.assertIn(phrase, markdown)
        self.assertIn("Initial shared image-prefill warm time", markdown)
        self.assertIn("combined upper MiB", markdown)
        self.assertIn("Recurrent working-set composition", markdown)
        self.assertIn("selected prefix control", markdown)
        self.assertIn("image/history weight", markdown)

    def test_schema_11_delete_storage_and_irreversibility_summary(self):
        rows = v11_rows("delete")
        summary = summarize(copy.deepcopy(META_DELETE), rows)
        self.assertEqual(summary["storage_mode"], "delete")
        recurrent = summary["turns"][1]["recurrent"]
        self.assertEqual(recurrent["retained_kv_tokens"], 4.0)
        self.assertEqual(recurrent["deleted_tokens_this_turn"], 2.0)
        self.assertEqual(recurrent["deleted_image_tokens_this_turn"], 2.0)
        text = render_markdown(summary, "delete.jsonl")
        self.assertIn("evicted K/V and per-token state are irreversible", text)
        self.assertIn("Persistent storage and deletion", text)
        self.assertIn("not method-only memory", text)

    def test_schema_11_offload_allows_active_image_change_without_deletion(self):
        metadata = {**META, "schema_version": "1.1", "storage": "offload",
                    "storage_mode": "offload"}
        rows = v11_rows("offload")
        summary = summarize(metadata, rows)
        self.assertEqual(summary["storage_mode"], "offload")
        self.assertEqual(summary["turns"][1]["recurrent"]["deleted_image_tokens_this_turn"], 0.0)
        self.assertIn("retains every K/V uncompressed", render_markdown(summary))

    def test_schema_11_delete_rejects_cold_storage_and_resurrection(self):
        cold = v11_rows("delete")
        cold[0]["cold_kv_bytes"] = 1
        with self.assertRaises(ValueError):
            summarize(copy.deepcopy(META_DELETE), cold)
        resurrected = v11_rows("delete")
        recurrent_step2 = next(r for r in resurrected
                               if r["condition_id"] == "recurrent" and r["step"] == 2)
        recurrent_step2["next_active_indices"] = [0, 2, 12, 13]
        with self.assertRaises(ValueError):
            summarize(copy.deepcopy(META_DELETE), resurrected)

    def test_loader_and_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            records = [META, *fixture_rows()]
            path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            metadata, rows = load_jsonl(path)
            self.assertEqual(metadata["run_id"], "run-1")
            self.assertEqual(len(rows), 6)

            cases = []
            duplicate = fixture_rows(); duplicate.append(copy.deepcopy(duplicate[0])); cases.append(duplicate)
            mixed_model = fixture_rows(); mixed_model[0]["model"] = "qwen3vl"; cases.append(mixed_model)
            mixed_run = fixture_rows(); mixed_run[0]["run_id"] = "run-2"; cases.append(mixed_run)
            incomplete = fixture_rows()[:-1]; cases.append(incomplete)
            for invalid_rows in cases:
                with self.subTest(case=len(invalid_rows), first=invalid_rows[0]["run_id"]):
                    path.write_text("\n".join(json.dumps(record) for record in
                                               [META, *invalid_rows]), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_jsonl(path)
            bad_metadata = {**META, "stage": "WRONG_STAGE"}
            path.write_text("\n".join(json.dumps(record) for record in
                                       [bad_metadata, *fixture_rows()]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
