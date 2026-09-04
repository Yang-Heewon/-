import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_diagnosis.exps import source_denial_text as text_eval


class TextPackageContractTest(unittest.TestCase):
    @staticmethod
    def external_descriptor(payload: bytes, relpath: str) -> dict:
        return {
            "payload_relpath": relpath,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "encoding": "utf-8",
            "file_count": 1,
            "original_utf8_bytes": len(payload),
            "byte_cap": None,
            "truncated": False,
        }

    def test_package_aliases_and_physical_record_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packages.jsonl"
            raw = (
                '{"memory_id":"mem-1","image_id":"screen-1",'
                '"plain_text":"hello"}\n'
            ).encode()
            path.write_bytes(raw)
            packages = text_eval.load_text_packages(path)

            self.assertIs(packages["mem-1"], packages["screen-1"])
            self.assertEqual(packages["mem-1"].record_bytes, len(raw))

    def test_source_path_is_rejected_in_question_or_package(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packages.jsonl"
            path.write_text(
                json.dumps({
                    "memory_id": "m1",
                    "image_path": "private/screen.png",
                    "plain_text": "secret",
                }) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "source-bearing key"):
                text_eval.load_text_packages(path)

    def test_duplicate_alias_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packages.jsonl"
            path.write_text(
                json.dumps({"memory_id": "same", "plain_text": "a"}) + "\n" +
                json.dumps({"image_id": "same", "plain_text": "b"}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate package"):
                text_eval.load_text_packages(path)

    def test_external_payload_is_loaded_only_after_size_hash_and_utf8_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = "Café 한글".encode("utf-8")
            (root / "plain.utf8").write_bytes(payload)
            path = root / "packages.jsonl"
            path.write_text(json.dumps({
                "sample_id": "m1",
                "representations": {
                    "plain": self.external_descriptor(payload, "plain.utf8")
                },
            }) + "\n")
            package = text_eval.load_text_packages(path)["m1"]
            selected = text_eval.select_memory(package, "plain", len(payload))
            self.assertEqual(selected.used_text, "Café 한글")
            self.assertEqual(selected.package_bytes, len(payload))
            self.assertEqual(
                package.external_payloads["plain"].payload_sha256,
                hashlib.sha256(payload).hexdigest(),
            )

    def test_external_package_rejects_hidden_inline_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"physical"
            (root / "plain.utf8").write_bytes(payload)
            path = root / "packages.jsonl"
            path.write_text(json.dumps({
                "sample_id": "m1",
                "plain_text": "hidden",
                "representations": {
                    "plain": {
                        **self.external_descriptor(payload, "plain.utf8"),
                        "audit": {"text": "another hidden copy"},
                    }
                },
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "hidden inline fallback"):
                text_eval.load_text_packages(path)

    def test_external_package_rejects_traversal_and_repository_data_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "packages.jsonl"
            path.write_text(json.dumps({
                "sample_id": "m1",
                "representations": {
                    "plain": self.external_descriptor(
                        b"outside", "../outside.utf8"
                    )
                },
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "unsafe external payload path"):
                text_eval.load_text_packages(path)

            data_dir = root / "data"
            data_dir.mkdir()
            payload = b"source-tree"
            (data_dir / "plain.utf8").write_bytes(payload)
            path.write_text(json.dumps({
                "sample_id": "m1",
                "representations": {
                    "plain": self.external_descriptor(
                        payload, "data/plain.utf8"
                    )
                },
            }) + "\n")
            with patch.object(text_eval, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "may not be read from data"):
                    text_eval.load_text_packages(path)


class RepresentationTest(unittest.TestCase):
    def package(self, row):
        return text_eval.TextPackage(row=row, aliases=("m",), record_bytes=100)

    def test_records_render_plain_and_layout_without_inventing_geometry(self):
        records = [
            {"text": "Settings", "bbox": [1, 2, 30, 10]},
            {"transcription": "Wi-Fi"},
        ]
        self.assertEqual(
            text_eval.text_from_records(records, "plain"),
            "Settings\nWi-Fi",
        )
        self.assertEqual(
            text_eval.text_from_records(records, "layout"),
            "[[1,2,30,10]] Settings\n[geometry=missing] Wi-Fi",
        )

    def test_utf8_byte_cap_is_physical_and_safe(self):
        selection = text_eval.select_memory(
            self.package({"plain_text": "가나다abc"}), "plain", 5
        )
        self.assertTrue(selection.available)
        self.assertEqual(selection.package_bytes, 12)
        self.assertEqual(selection.used_text, "가")
        self.assertEqual(selection.used_bytes, 3)
        self.assertTrue(selection.truncated)

    def test_layout_does_not_silently_fallback_to_plain(self):
        selection = text_eval.select_memory(
            self.package({"plain_text": "top left: menu"}), "layout", None
        )
        self.assertFalse(selection.available)
        self.assertIsNone(selection.used_bytes)
        self.assertIn("no layout", selection.reason)

    def test_paddleocr_writer_schema_is_consumed_without_relabeling(self):
        package = self.package({
            "recognized_text": "Settings\nWi-Fi",
            "layout_text": "l0 bbox=[1,2,30,10] text=\"Settings\"",
            "lines": [
                {"text": "Settings", "bbox_xyxy_px": [1, 2, 30, 10]},
            ],
        })
        plain = text_eval.select_memory(package, "plain", None)
        layout = text_eval.select_memory(package, "layout", None)
        self.assertEqual(plain.used_text, "Settings\nWi-Fi")
        self.assertEqual(
            layout.used_text, 'l0 bbox=[1,2,30,10] text="Settings"'
        )

    def test_empty_explicit_representation_is_available(self):
        selection = text_eval.select_memory(
            self.package({"plain_text": ""}), "plain", 0
        )
        self.assertTrue(selection.available)
        self.assertEqual(selection.used_bytes, 0)


class ReadRunnerTest(unittest.TestCase):
    def test_mocked_run_consumes_complete_external_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_path = root / "questions.jsonl"
            package_path = root / "packages.jsonl"
            output_path = root / "out.jsonl"
            payload = "Network: Lab WiFi".encode("utf-8")
            (root / "memory.utf8").write_bytes(payload)
            question_path.write_text(json.dumps({
                "sample_id": "screen-1",
                "questions": [{
                    "question_id": "q1",
                    "question": "Which network?",
                    "answers": ["Lab WiFi"],
                }],
            }) + "\n")
            package_path.write_text(json.dumps({
                "sample_id": "screen-1",
                "representations": {
                    "plain": TextPackageContractTest.external_descriptor(
                        payload, "memory.utf8"
                    )
                },
            }) + "\n")
            args = argparse.Namespace(
                manifest=str(question_path), package_manifest=str(package_path),
                out=str(output_path), representation="plain",
                byte_cap=len(payload), model_id="mock-qwen", device="cpu",
                shard=0, nshards=1, limit=None, questions_per_image=2,
                max_new_tokens=4, resume=False,
            )

            with patch.object(text_eval, "ROOT", root), patch.object(
                text_eval,
                "greedy_text_answer",
                return_value=("Lab WiFi", 0.01, 42),
            ):
                text_eval.run_read(args, model=object(), tokenizer=object())

            result = json.loads(output_path.read_text().splitlines()[1])
            self.assertTrue(result["feasible"])
            self.assertEqual(result["used_bytes"], len(payload))
            self.assertEqual(result["package_bytes"], len(payload))
            self.assertEqual(
                result["package_storage_kind"], "external_materialized_utf8"
            )
            self.assertEqual(
                result["budget_scope"],
                "one_complete_materialized_representation_payload",
            )
            self.assertEqual(
                result["materialized_payload_sha256"],
                hashlib.sha256(payload).hexdigest(),
            )

    def test_mocked_run_scores_and_never_opens_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_path = root / "questions.jsonl"
            package_path = root / "packages.jsonl"
            output_path = root / "out.jsonl"
            question_path.write_text(json.dumps({
                "sample_id": "screen-1",
                "questions": [{
                    "question_id": "q1",
                    "question": "Which network?",
                    "answers": ["Lab WiFi"],
                }],
            }) + "\n")
            package_path.write_text(json.dumps({
                "image_id": "screen-1",
                "plain_text": "Network: Lab WiFi",
            }) + "\n")
            args = argparse.Namespace(
                manifest=str(question_path),
                package_manifest=str(package_path),
                out=str(output_path),
                representation="plain",
                byte_cap=None,
                model_id="mock-qwen",
                device="cpu",
                shard=0,
                nshards=1,
                limit=None,
                questions_per_image=2,
                max_new_tokens=4,
                resume=False,
            )

            def fake_answer(model, tokenizer, messages, device, max_new_tokens):
                joined = "\n".join(message["content"] for message in messages)
                self.assertIn("Network: Lab WiFi", joined)
                self.assertNotIn(".png", joined)
                return "Lab WiFi", 0.01, 42

            with patch.object(text_eval, "ROOT", root), patch.object(
                text_eval, "greedy_text_answer", side_effect=fake_answer
            ):
                text_eval.run_read(args, model=object(), tokenizer=object())

            records = [json.loads(line) for line in output_path.read_text().splitlines()]
            result = records[1]
            self.assertEqual(result["em"], 1.0)
            self.assertEqual(result["anls"], 1.0)
            self.assertEqual(result["package_bytes"], len("Network: Lab WiFi"))
            self.assertEqual(result["used_bytes"], result["package_bytes"])
            self.assertTrue(result["feasible"])
            self.assertFalse(result["source_path_in_read_manifest"])
            self.assertEqual(result["first_token_seconds"], 0.01)

    def test_question_manifest_with_image_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_path = root / "questions.jsonl"
            package_path = root / "packages.jsonl"
            question_path.write_text(json.dumps({
                "sample_id": "s",
                "image": "forbidden.png",
                "questions": [],
            }) + "\n")
            package_path.write_text(json.dumps({
                "sample_id": "s", "plain_text": "x"
            }) + "\n")
            args = argparse.Namespace(
                manifest=str(question_path), package_manifest=str(package_path),
                out=str(root / "out.jsonl"), representation="plain",
                byte_cap=None, model_id="mock-qwen", device="cpu",
                shard=0, nshards=1, limit=None, questions_per_image=2,
                max_new_tokens=4, resume=False,
            )
            with patch.object(text_eval, "ROOT", root), patch.object(
                text_eval, "load_text_model"
            ) as loader:
                with self.assertRaisesRegex(ValueError, "source-bearing key"):
                    text_eval.run_read(args)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
