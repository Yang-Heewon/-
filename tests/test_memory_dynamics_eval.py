import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vlm_diagnosis.exps import memory_dynamics_eval as evaluator
from vlm_diagnosis.exps.memory_dynamics_eval import (
    DEFAULT_ARMS,
    build_messages,
    iter_trial_specs,
    load_package_index,
    memory_package_path,
    package_condition,
    prediction_scores,
    prepare_manifests,
    split_episode_row,
)
from vlm_diagnosis.exps.source_denial_kv import assert_source_free
from vlm_diagnosis.scripts.gen_memory_dynamics import build_episode


class MemoryDynamicsEvalContractTest(unittest.TestCase):
    def setUp(self):
        self.row = build_episode(3, 42, "data/fixture")[1]

    def test_split_is_question_free_for_writer_and_source_free_for_reader(self):
        write_rows, read_row = split_episode_row(self.row)
        self.assertEqual(len(write_rows), 5)
        self.assertEqual(len(read_row["available_memory_ids"]), 5)
        self.assertNotIn("memories", read_row)
        assert_source_free(read_row)

        for write_row in write_rows:
            self.assertEqual(
                set(write_row),
                {
                    "dataset", "dataset_revision", "split", "episode_id",
                    "memory_id", "image", "image_sha256",
                },
            )
            serialized = json.dumps(write_row).lower()
            self.assertNotIn("question", serialized)
            self.assertNotIn("acceptable_answers", serialized)

        read_serialized = json.dumps(read_row)
        for memory in self.row["memories"]:
            self.assertNotIn(memory["image"], read_serialized)

    def test_prepare_expands_memories_but_keeps_one_read_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
            write_path = root / "write.jsonl"
            read_path = root / "read.jsonl"
            counts = prepare_manifests(source, write_path, read_path)
            self.assertEqual(counts, (5, 1))
            self.assertEqual(len(write_path.read_text().splitlines()), 5)
            self.assertEqual(len(read_path.read_text().splitlines()), 1)
            assert_source_free(json.loads(read_path.read_text()))

    def test_trial_plan_covers_oracle_interference_and_four_d4_conditions(self):
        _, read_row = split_episode_row(self.row)
        specs = iter_trial_specs(read_row)
        by_arm = {}
        for spec in specs:
            by_arm.setdefault(spec.arm, []).append(spec)
        self.assertEqual(set(by_arm), set(DEFAULT_ARMS))
        self.assertEqual(len(by_arm["d3_oracle"]), 3)
        self.assertEqual(len(by_arm["d3_interference"]), 3)
        self.assertEqual(len(by_arm["d4"]), 4)

        for spec in by_arm["d3_oracle"]:
            self.assertEqual(len(spec.memory_ids), 1)
            self.assertEqual(
                len(spec.storage_memory_ids), spec.source_trial["candidate_count"]
            )
            self.assertEqual(
                spec.storage_memory_ids[spec.target_position], spec.relevant_memory_id
            )
            self.assertTrue(spec.retrieval_bypassed)
            self.assertFalse(spec.source_trial["retrieval_bypassed"])
        for spec in by_arm["d3_interference"]:
            self.assertTrue(spec.retrieval_bypassed)
            self.assertIsNotNone(spec.target_position)

        conditions = {
            spec.source_trial["memory_condition"]: spec
            for spec in by_arm["d4"]
        }
        self.assertEqual(
            set(conditions), {"old_only", "current_only", "old+current", "no_memory"}
        )
        self.assertIsNone(conditions["old_only"].target_position)
        self.assertEqual(conditions["current_only"].target_position, 0)
        self.assertEqual(conditions["old+current"].target_position, 1)
        self.assertEqual(conditions["no_memory"].memory_ids, ())

    def test_messages_interleave_exactly_one_placeholder_per_image(self):
        messages = build_messages(3, "What is the current status?")
        content = messages[0]["content"]
        self.assertEqual(sum(item["type"] == "image" for item in content), 3)
        self.assertIn("newest applicable evidence", content[-1]["text"])
        no_memory = build_messages(0, "What is the current status?")
        self.assertFalse(any(
            item["type"] == "image" for item in no_memory[0]["content"]
        ))

    def test_current_and_stale_scores_are_separate(self):
        question = {
            "acceptable_answers": ["READY"],
            "stale_answers": ["BLOCKED"],
        }
        self.assertEqual(prediction_scores("READY", question)["current_em"], 1.0)
        scores = prediction_scores("BLOCKED", question)
        self.assertEqual(scores["current_em"], 0.0)
        self.assertEqual(scores["stale_capture"], 1.0)
        self.assertEqual(
            prediction_scores("BLOCKED", {
                "acceptable_answers": ["READY"], "stale_answers": [],
            })["stale_capture"],
            0.0,
        )

    def test_condition_and_package_paths_include_physical_cap(self):
        self.assertEqual(package_condition("jpeg", 65536), "JPEG@65536B")
        self.assertEqual(package_condition("copy", None), "SOURCE_CONTAINER_COPY")
        first = memory_package_path(
            Path("packages"), "../../memory one", "webp", 65536
        )
        second = memory_package_path(
            Path("packages"), "../../memory one", "webp", 131072
        )
        self.assertEqual(first.parent, Path("packages"))
        self.assertNotIn("..", first.name)
        self.assertNotEqual(first, second)

    def test_package_index_rejects_duplicates_and_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packages.jsonl"
            record = {
                "record_type": "package",
                "memory_id": "m1",
                "condition_id": "JPEG@65536B",
                "package": "results/p.jpeg",
                "package_bytes": 10,
                "package_sha256": "abc",
                "feasible": True,
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertIn(("m1", "JPEG@65536B"), load_package_index(path))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate package"):
                load_package_index(path)

            bad = dict(record, image_path="data/private.png")
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source-bearing key"):
                load_package_index(path)

    def test_cpu_mock_reader_emits_required_d3_d4_measurements(self):
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
                out=str(output),
                codec="jpeg",
                arms=",".join(DEFAULT_ARMS),
                device="cpu",
                shard=0,
                nshards=1,
                limit=1,
                max_new_tokens=8,
                resume=False,
            )
            fake_model = SimpleNamespace(
                config=SimpleNamespace(model_type="qwen2_5_vl")
            )

            def fake_packages(spec, _packages, _codec):
                records = [
                    {
                        "memory_id": memory_id,
                        "package_bytes": 10,
                        "package_sha256": f"hash-{index}",
                    }
                    for index, memory_id in enumerate(spec.storage_memory_ids)
                ]
                return [object() for _ in spec.memory_ids], records, None

            with patch.object(evaluator, "load_vlm", return_value=(fake_model, object())), \
                    patch.object(evaluator, "_load_trial_images", side_effect=fake_packages), \
                    patch.object(
                        evaluator,
                        "generate_answer",
                        return_value=read_row["question"]["current_answer"],
                    ):
                evaluator.run_read(args)

            records = [
                json.loads(line) for line in output.read_text().splitlines()
                if json.loads(line).get("record_type") == "trial_result"
            ]
            self.assertEqual(len(records), 10)
            self.assertTrue(all(record["current_em"] == 1.0 for record in records))
            self.assertTrue(all(record["stale_capture"] == 0.0 for record in records))
            self.assertTrue(all("target_position" in record for record in records))
            self.assertTrue(all("total_package_bytes" in record for record in records))
            self.assertTrue(all(record["retrieval_bypassed"] for record in records))
            oracle_n4 = next(
                record for record in records
                if record["arm"] == "d3_oracle" and record["candidate_count"] == 4
            )
            self.assertEqual(oracle_n4["n_memory_packages"], 4)
            self.assertEqual(oracle_n4["n_inference_images"], 1)
            self.assertEqual(oracle_n4["total_package_bytes"], 40)


if __name__ == "__main__":
    unittest.main()
