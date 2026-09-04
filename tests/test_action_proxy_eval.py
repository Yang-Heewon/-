import builtins
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vlm_diagnosis.exps import action_proxy_eval as evaluator
from vlm_diagnosis.exps.action_proxy_eval import (
    DEFAULT_ARMS,
    bbox_iou,
    build_messages,
    expand_multi_image_placeholders,
    inject_multi_projected_visual_tokens,
    iter_trial_specs,
    load_package_index,
    normalize_operational_arguments,
    parse_action_prediction,
    point_in_bbox,
    prepare_manifests,
    resolve_projected_package,
    score_action_prediction,
    spatial_target_diagnostics,
    split_episode_row,
)
from vlm_diagnosis.exps.source_denial_kv import assert_source_free
from vlm_diagnosis.scripts.gen_action_proxy import build_episode


class ActionProxyEvalContractTest(unittest.TestCase):
    def setUp(self):
        self.row = build_episode(1, 42, "data/action_fixture")[1]  # type action

    def test_split_is_question_free_for_writer_and_source_free_for_reader(self):
        write_rows, read_row = split_episode_row(self.row)
        self.assertEqual(len(write_rows), 2)
        self.assertEqual(
            [item["role"] for item in read_row["observation_metadata"]],
            ["old_state", "current_state"],
        )
        assert_source_free(read_row)
        self.assertNotIn("observations", read_row)
        self.assertNotIn("state_facts", json.dumps(read_row))
        self.assertFalse(read_row["real_trajectory_claim_allowed"])

        serialized_read = json.dumps(read_row)
        for observation in self.row["observations"]:
            self.assertNotIn(observation["image"], serialized_read)
        for write_row in write_rows:
            self.assertEqual(
                set(write_row),
                {
                    "dataset", "dataset_revision", "split", "episode_id",
                    "sample_id", "image", "image_sha256",
                },
            )
            serialized = json.dumps(write_row).lower()
            self.assertNotIn("task_goal", serialized)
            self.assertNotIn("gold_action", serialized)
            self.assertNotIn("question", serialized)

    def test_prepare_emits_two_writes_and_one_read_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
            write_path = root / "write.jsonl"
            read_path = root / "read.jsonl"
            self.assertEqual(
                prepare_manifests(source, write_path, read_path),
                (2, 1),
            )
            self.assertEqual(len(write_path.read_text().splitlines()), 2)
            read_row = json.loads(read_path.read_text())
            assert_source_free(read_row)
            self.assertEqual(len(read_row["available_observation_ids"]), 2)

    def test_trial_arms_have_exact_memory_and_history_inputs(self):
        _, read_row = split_episode_row(self.row)
        specs = iter_trial_specs(read_row)
        self.assertEqual([spec.arm for spec in specs], list(DEFAULT_ARMS))
        by_arm = {spec.arm: spec for spec in specs}
        old_id, current_id = read_row["available_observation_ids"]
        self.assertEqual(
            by_arm["ordered_history"].observation_ids,
            (old_id, current_id),
        )
        self.assertTrue(by_arm["ordered_history"].include_event_history)
        self.assertEqual(by_arm["current_only"].observation_ids, (current_id,))
        self.assertEqual(by_arm["old_only"].observation_ids, (old_id,))
        self.assertEqual(by_arm["no_memory"].observation_ids, ())
        self.assertTrue(all(
            not spec.include_event_history
            for arm, spec in by_arm.items() if arm != "ordered_history"
        ))

    def test_prompts_have_expected_images_without_text_gold_leakage(self):
        _, read_row = split_episode_row(self.row)
        secret = read_row["gold_action"]["arguments"]["text"]
        expected_images = {
            "ordered_history": 2,
            "current_only": 1,
            "old_only": 1,
            "no_memory": 0,
        }
        for spec in iter_trial_specs(read_row):
            messages = build_messages(read_row, spec)
            content = messages[0]["content"]
            self.assertEqual(
                sum(item["type"] == "image" for item in content),
                expected_images[spec.arm],
            )
            model_visible_text = " ".join(
                item["text"] for item in content if item["type"] == "text"
            )
            self.assertNotIn(secret, model_visible_text)
            self.assertNotIn(read_row["gold_action"]["target_element_id"], model_visible_text)
            self.assertIn('click -> {"button":"left"}', model_visible_text)
            self.assertIn(
                'type -> {"text":"<exact saved token>","replace_existing":false}',
                model_visible_text,
            )
            self.assertIn(
                'scroll -> {"direction":"up|down","amount":"one_viewport"}',
                model_visible_text,
            )
            self.assertIn("Do not rename text to token", model_visible_text)
            self.assertIn("exactly four numbers", model_visible_text)
            if spec.arm == "ordered_history":
                self.assertIn("completed action", model_visible_text)
                self.assertIn("must not be replayed", model_visible_text)

    def test_multi_image_placeholder_expansion_and_injection(self):
        raw = torch.tensor([[1, 99, 2, 99, 3]])
        expanded = expand_multi_image_placeholders(raw, 99, [2, 3])
        self.assertEqual(expanded.tolist(), [[1, 99, 99, 2, 99, 99, 99, 3]])
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            expand_multi_image_placeholders(raw, 99, [2])

        embeddings = torch.zeros(1, 8, 2)
        old = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16)
        current = torch.tensor([[5, 6], [7, 8], [9, 10]], dtype=torch.float16)
        injected = inject_multi_projected_visual_tokens(
            expanded, embeddings, [old, current], 99
        )
        self.assertEqual(
            injected[0, expanded[0] == 99].tolist(),
            torch.cat([old, current]).float().tolist(),
        )
        text_only = torch.tensor([[1, 2, 3]])
        text_embeddings = torch.ones(1, 3, 2)
        self.assertTrue(torch.equal(
            inject_multi_projected_visual_tokens(text_only, text_embeddings, [], 99),
            text_embeddings,
        ))

    def test_strict_action_and_component_scoring(self):
        _, read_row = split_episode_row(self.row)
        gold = read_row["gold_action"]
        exact = json.dumps({
            "action_type": gold["action_type"],
            "target_label": gold["target_label"],
            "target_bbox": gold["target_bbox"],
            "arguments": gold["arguments"],
        })
        score = score_action_prediction(
            f"```json\n{exact}\n```", gold, read_row["invalidated_actions"]
        )
        self.assertTrue(score["prediction_json_valid"])
        self.assertEqual(score["action_type_em"], 1.0)
        self.assertEqual(score["arguments_em"], 1.0)
        self.assertEqual(score["action_type_arguments_em"], 1.0)
        self.assertEqual(score["target_bbox_iou"], 1.0)
        self.assertEqual(score["full_action_em"], 1.0)
        self.assertEqual(score["stale_action_replay"], 0.0)
        self.assertEqual(parse_action_prediction("not json"), None)

        wrong_case_token = json.dumps({
            "action_type": "type",
            "target_label": gold["target_label"],
            "target_bbox": gold["target_bbox"],
            "arguments": {"text": gold["arguments"]["text"].lower(),
                          "replace_existing": False},
        })
        wrong_score = score_action_prediction(
            wrong_case_token, gold, read_row["invalidated_actions"]
        )
        self.assertEqual(wrong_score["action_type_em"], 1.0)
        self.assertEqual(wrong_score["arguments_em"], 0.0)
        self.assertEqual(wrong_score["full_action_em"], 0.0)

        # A semantic UI role is accepted only when it uniquely identifies the
        # interactive target (true for the single token field, not click lists).
        role_prediction = json.dumps({
            "action_type": "type",
            "target_label": "text_field",
            "arguments": gold["arguments"],
        })
        role_score = score_action_prediction(
            role_prediction, gold, read_row["invalidated_actions"]
        )
        self.assertEqual(role_score["target_role_em"], 1.0)
        self.assertEqual(role_score["full_action_em"], 1.0)

        click_row = split_episode_row(build_episode(0, 42, "data/action_fixture")[1])[1]
        click_gold = click_row["gold_action"]
        generic_button = score_action_prediction(
            json.dumps({
                "action_type": "click",
                "target_label": "button",
                "arguments": click_gold["arguments"],
            }),
            click_gold,
            click_row["invalidated_actions"],
        )
        self.assertEqual(generic_button["target_role_em"], 0.0)
        self.assertEqual(generic_button["full_action_em"], 0.0)

        stale = score_action_prediction(
            json.dumps({
                "action_type": "click", "target_label": "OPEN",
                "target_bbox": [442, 470, 676, 520],
                "arguments": {"button": "left"},
            }),
            gold,
            read_row["invalidated_actions"],
        )
        self.assertEqual(stale["stale_action_replay"], 1.0)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_operational_metric_uses_only_declared_aliases_and_defaults(self):
        click = normalize_operational_arguments("click", {})
        self.assertTrue(click["valid"])
        self.assertEqual(click["normalized_arguments"], {"button": "left"})
        self.assertEqual(click["applied_defaults"], ["button"])

        token = "GKB4TCD"
        typed = normalize_operational_arguments("type", {"token": token})
        self.assertTrue(typed["valid"])
        self.assertEqual(
            typed["normalized_arguments"],
            {"text": token, "replace_existing": False},
        )
        self.assertEqual(typed["applied_aliases"], ["token->text"])
        self.assertEqual(typed["applied_defaults"], ["replace_existing"])

        scroll = normalize_operational_arguments("scroll", {"direction": "up"})
        self.assertTrue(scroll["valid"])
        self.assertEqual(
            scroll["normalized_arguments"],
            {"direction": "up", "amount": "one_viewport"},
        )
        self.assertFalse(normalize_operational_arguments("scroll", {})["valid"])
        self.assertFalse(
            normalize_operational_arguments(
                "type", {"token": token, "text": "DIFFERENT"}
            )["valid"]
        )
        self.assertFalse(
            normalize_operational_arguments("click", {"button": "left", "x": 1})[
                "valid"
            ]
        )

        _, type_row = split_episode_row(self.row)
        gold = type_row["gold_action"]
        alias_prediction = json.dumps({
            "action_type": "type",
            "target_label": gold["target_label"],
            "target_bbox": gold["target_bbox"],
            "arguments": {"token": gold["arguments"]["text"]},
        })
        score = score_action_prediction(
            alias_prediction, gold, type_row["invalidated_actions"]
        )
        # Raw strict fields are unchanged and remain visible beside the
        # explicitly named secondary operational fields.
        self.assertEqual(score["arguments_em"], 0.0)
        self.assertEqual(score["argument_component_em"]["text"], 0.0)
        self.assertEqual(score["operational_arguments_em"], 1.0)
        self.assertEqual(score["operational_action_type_arguments_em"], 1.0)
        self.assertEqual(score["operational_full_action_em"], 1.0)
        self.assertIn("operational_argument_normalization", score["component_metrics"])

    def test_first_run_schema_artifact_is_separate_from_scroll_model_failure(self):
        click_row = split_episode_row(build_episode(0, 42, "data/action_fixture")[1])[1]
        click_gold = click_row["gold_action"]
        old_prompt_style_click = score_action_prediction(
            json.dumps({
                "action_type": "click",
                "target_label": click_gold["target_label"],
                "target_bbox": [300, 600, 400, 630],
                "arguments": {},
            }),
            click_gold,
            click_row["invalidated_actions"],
        )
        self.assertEqual(old_prompt_style_click["arguments_em"], 0.0)
        self.assertEqual(old_prompt_style_click["operational_arguments_em"], 1.0)
        self.assertEqual(old_prompt_style_click["target_label_em"], 1.0)
        self.assertEqual(old_prompt_style_click["operational_full_action_em"], 1.0)

        scroll_row = split_episode_row(build_episode(2, 42, "data/action_fixture")[1])[1]
        scroll_gold = scroll_row["gold_action"]
        stale_open = score_action_prediction(
            json.dumps({
                "action_type": "click",
                "target_label": "OPEN",
                "target_bbox": [518, 514],
                "arguments": {},
            }),
            scroll_gold,
            scroll_row["invalidated_actions"],
        )
        self.assertEqual(stale_open["action_type_em"], 0.0)
        self.assertEqual(stale_open["operational_arguments_em"], 0.0)
        self.assertEqual(stale_open["operational_action_type_arguments_em"], 0.0)
        self.assertEqual(stale_open["stale_action_replay"], 1.0)

    def test_spatial_metrics_keep_iou_center_and_point_separate(self):
        gold = [112, 576, 656, 662]
        low_iou_center_hit = spatial_target_diagnostics(
            {"target_bbox": [300, 600, 400, 630]}, gold
        )
        self.assertLess(low_iou_center_hit["target_bbox_iou"], 0.5)
        self.assertEqual(low_iou_center_hit["target_bbox_iou_at_0_5"], 0.0)
        self.assertEqual(low_iou_center_hit["target_bbox_center_inside_gold"], 1.0)
        self.assertEqual(low_iou_center_hit["target_point_hit"], 0.0)
        self.assertEqual(low_iou_center_hit["spatial_target_success"], 1.0)

        legacy_point = spatial_target_diagnostics(
            {"target_bbox": [322, 620]}, gold
        )
        self.assertIsNone(legacy_point["target_bbox_iou"])
        self.assertEqual(legacy_point["target_bbox_center_inside_gold"], 0.0)
        self.assertEqual(legacy_point["target_point_hit"], 1.0)
        self.assertEqual(
            legacy_point["predicted_target_point_source"],
            "target_bbox_legacy_point",
        )
        self.assertTrue(point_in_bbox([112, 576], gold))
        self.assertFalse(point_in_bbox([111, 576], gold))

    def test_package_index_and_resolver_reject_source_material(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packages.jsonl"
            record = {
                "record_type": "package",
                "representation": evaluator.REPRESENTATION,
                "observation_id": "obs1",
                "quantization": "fp16",
                "package": "results/p.pt",
                "package_bytes": 10,
                "package_sha256": "abc",
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertIn(("obs1", "fp16"), load_package_index(path))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate action package"):
                load_package_index(path)

            bad = dict(record, image_path="data/private.png")
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source-bearing key"):
                load_package_index(path)
        with self.assertRaisesRegex(ValueError, "source-data path"):
            resolve_projected_package("data/action_proxy_controlled/private.pt")

    def test_cpu_mock_reader_records_four_arms_without_importing_pil(self):
        _, read_row = split_episode_row(self.row)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "read.jsonl"
            manifest.write_text(json.dumps(read_row) + "\n", encoding="utf-8")
            package_manifest = root / "packages.jsonl"
            package_manifest.write_text("", encoding="utf-8")
            output = root / "output.jsonl"
            args = SimpleNamespace(
                manifest=str(manifest),
                package_manifest=str(package_manifest),
                package_dir="unused",
                out=str(output),
                arms=",".join(DEFAULT_ARMS),
                quantization="fp16",
                model="qwen25vl",
                device="cpu",
                shard=0,
                nshards=1,
                limit=1,
                max_new_tokens=64,
                resume=False,
            )
            fake_model = SimpleNamespace(
                config=SimpleNamespace(model_type="qwen2_5_vl")
            )
            gold = read_row["gold_action"]
            prediction = json.dumps({
                "action_type": gold["action_type"],
                "target_label": gold["target_label"],
                "target_bbox": gold["target_bbox"],
                "arguments": gold["arguments"],
            })

            def fake_packages(spec, _packages, _quantization, _model):
                records = [
                    {
                        "observation_id": observation_id,
                        "package_bytes": 10,
                        "package_sha256": f"hash-{index}",
                    }
                    for index, observation_id in enumerate(spec.observation_ids)
                ]
                return [object() for _ in spec.observation_ids], records

            original_import = builtins.__import__

            def import_without_pil(name, *import_args, **import_kwargs):
                if name == "PIL" or name.startswith("PIL."):
                    raise AssertionError("reader attempted to import PIL")
                return original_import(name, *import_args, **import_kwargs)

            with patch.object(evaluator, "load_vlm", return_value=(fake_model, object())), \
                    patch.object(evaluator, "_load_trial_packages", side_effect=fake_packages), \
                    patch.object(
                        evaluator,
                        "generate_prediction",
                        return_value=(prediction, {
                            "reconstruction_seconds": 0.01,
                            "prefill_seconds": 0.02,
                            "decode_seconds": 0.03,
                        }),
                    ), patch("builtins.__import__", side_effect=import_without_pil):
                evaluator.run_read(args)

            records = [
                json.loads(line) for line in output.read_text().splitlines()
                if json.loads(line).get("record_type") == "trial_result"
            ]
            self.assertEqual(len(records), 4)
            self.assertEqual({record["arm"] for record in records}, set(DEFAULT_ARMS))
            self.assertTrue(all(record["action_type_arguments_em"] == 1.0
                                for record in records))
            self.assertTrue(all(record["full_action_em"] == 1.0 for record in records))
            self.assertTrue(all(record["pil_used"] is False for record in records))
            self.assertTrue(all(record["pixel_values_used"] is False for record in records))
            self.assertTrue(all(record["real_trajectory_claim_allowed"] is False
                                for record in records))
            package_counts = {record["arm"]: record["n_memory_packages"] for record in records}
            self.assertEqual(package_counts, {
                "ordered_history": 2,
                "current_only": 1,
                "old_only": 1,
                "no_memory": 0,
            })


if __name__ == "__main__":
    unittest.main()
