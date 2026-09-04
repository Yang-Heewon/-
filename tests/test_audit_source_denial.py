import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vlm_diagnosis.scripts.audit_source_denial import (
    ArmPaths,
    audit_trace,
    parse_arm,
    run_audit,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StrictSourceDenialAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "data" / "toy" / "screen.png"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"source pixels")
        self.write_manifest = self.root / "write.jsonl"
        _write_jsonl(
            self.write_manifest,
            [
                {
                    "sample_id": "s1",
                    "image": "data/toy/screen.png",
                    "image_sha256": _sha256(self.source),
                }
            ],
        )
        self.read_manifest = self.root / "read.jsonl"
        _write_jsonl(
            self.read_manifest,
            [
                {
                    "sample_id": "s1",
                    "questions": [
                        {"question_id": "q1", "question": "first?", "image_width": 8},
                        {"question_id": "q2", "question": "second?", "role": "source"},
                    ],
                }
            ],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _metadata(self, pid: int) -> dict:
        return {
            "record_type": "run_metadata",
            "mode": "read",
            "manifest": "read.jsonl",
            "manifest_sha256": _sha256(self.read_manifest),
            "source_path_available": False,
            "pixel_values_available": False,
            "process_id": pid,
        }

    @staticmethod
    def _result(question_id: str, prediction: str, pid: int) -> dict:
        return {
            "sample_id": "s1",
            "question_id": question_id,
            "feasible": True,
            "prediction": prediction,
            "package_bytes": 17,
            "package_sha256": "a" * 64,
            "source_path_in_read_manifest": False,
            "reader_pid": pid,
        }

    def _arm(
        self,
        first_rows: list[dict] | None = None,
        repeat_rows: list[dict] | None = None,
        trace_text: str | None = None,
    ) -> ArmPaths:
        first = self.root / "first.jsonl"
        repeat = self.root / "repeat.jsonl"
        trace = self.root / "reader.openat.log"
        _write_jsonl(
            first,
            first_rows
            or [self._metadata(101), self._result("q1", "A", 101), self._result("q2", "B", 101)],
        )
        _write_jsonl(
            repeat,
            repeat_rows
            or [self._metadata(202), self._result("q1", "A", 202), self._result("q2", "B", 202)],
        )
        trace.write_text(
            trace_text
            or (
                f'101 openat(AT_FDCWD, "{self.root}/packages/memory.bin", O_RDONLY) = 3\n'
                "101 +++ exited with 0 +++\n"
            ),
            encoding="utf-8",
        )
        return ArmPaths("toy", first, trace, repeat)

    def _run(self, arm: ArmPaths) -> dict:
        return run_audit(
            self.write_manifest,
            self.read_manifest,
            [arm],
            self.root,
            dataset_roots=[self.root / "data" / "toy"],
        )

    def test_complete_heterogeneous_arm_passes(self):
        first = [
            self._metadata(101),
            {
                **self._result("q1", "A", 101),
                "package_bytes": 11,
                "package_sha256": "b" * 64,
            },
            {
                **self._result("q2", "B", 101),
                "package_bytes": 11,
                "package_sha256": "b" * 64,
            },
        ]
        repeat = [
            self._metadata(202),
            {
                **self._result("q1", "A", 202),
                "package_bytes": 11,
                "package_sha256": "b" * 64,
            },
            {
                **self._result("q2", "B", 202),
                "package_bytes": 11,
                "package_sha256": "b" * 64,
            },
        ]
        result = self._run(self._arm(first, repeat))
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "PASS")
        arm = result["arms"]["toy"]
        self.assertEqual(arm["first_results"]["n_feasible_results"], 2)
        self.assertTrue(arm["repeat_determinism"]["exact_prediction_determinism"])
        self.assertTrue(arm["first_reader_trace"]["reader_pid_present"])
        self.assertEqual(arm["first_reader_trace"]["reader_exit_status"], 0)

    def test_missing_manifest_questions_fail_strictly(self):
        first = [self._metadata(101), self._result("q1", "A", 101)]
        repeat = [self._metadata(202), self._result("q1", "A", 202)]
        result = self._run(self._arm(first, repeat))
        self.assertFalse(result["passed"])
        first_summary = result["arms"]["toy"]["first_results"]
        self.assertEqual(first_summary["missing_keys"], [["s1", "q2"]])
        self.assertIn("missing expected question results", " ".join(first_summary["errors"]))

    def test_forbidden_nested_read_path_key_fails_but_dimensions_are_allowed(self):
        _write_jsonl(
            self.read_manifest,
            [
                {
                    "sample_id": "s1",
                    "questions": [
                        {
                            "question_id": "q1",
                            "question": "x?",
                            "image_width": 8,
                            "evidence": {"source_path": "data/toy/screen.png"},
                        },
                        {"question_id": "q2", "question": "y?", "image_height": 8},
                    ],
                }
            ],
        )
        # Refresh metadata hashes so the asserted failure is specifically the
        # recursively detected source path key.
        arm = self._arm()
        first = [self._metadata(101), self._result("q1", "A", 101), self._result("q2", "B", 101)]
        repeat = [self._metadata(202), self._result("q1", "A", 202), self._result("q2", "B", 202)]
        arm = self._arm(first, repeat)
        result = self._run(arm)
        manifest = result["manifests"]
        self.assertEqual(manifest["status"], "FAIL")
        occurrences = manifest["read"]["forbidden_path_key_occurrences"]
        self.assertEqual([item["key"] for item in occurrences], ["source_path"])

    def test_exact_source_or_dataset_open_fails(self):
        trace = (
            f'101 openat(AT_FDCWD, "{self.source}", O_RDONLY) = 3\n'
            "101 +++ exited with 0 +++\n"
        )
        result = self._run(self._arm(trace_text=trace))
        summary = result["arms"]["toy"]["first_reader_trace"]
        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(len(summary["exact_source_hits"]), 1)
        self.assertEqual(len(summary["dataset_root_hits"]), 1)
        self.assertEqual(len(summary["source_basename_hits"]), 1)

    def test_reader_pid_coverage_and_terminal_exit_are_required(self):
        trace = (
            f'999 openat(AT_FDCWD, "{self.root}/packages/memory.bin", O_RDONLY) = 3\n'
            "999 +++ exited with 0 +++\n"
        )
        result = self._run(self._arm(trace_text=trace))
        errors = result["arms"]["toy"]["first_reader_trace"]["errors"]
        self.assertIn("reader PID is absent from the strace", errors)
        self.assertIn("strace has no terminal exit record for the reader PID", errors)

    def test_infeasible_row_and_ocr_style_used_hash_are_schema_tolerant(self):
        first = [
            self._metadata(101),
            {
                "sample_id": "s1",
                "question_id": "q1",
                "feasible": True,
                "prediction": "A",
                "package_bytes": 9,
                "used_bytes": 9,
                "used_sha256": "c" * 64,
                "reader_pid": 101,
            },
            {
                "sample_id": "s1",
                "question_id": "q2",
                "feasible": False,
                "infeasible_reason": "budget too small",
                "reader_pid": 101,
            },
        ]
        repeat = [
            self._metadata(202),
            {**first[1], "reader_pid": 202},
            {**first[2], "reader_pid": 202},
        ]
        result = self._run(self._arm(first, repeat))
        summary = result["arms"]["toy"]["first_results"]
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["n_feasible_results"], 1)
        self.assertEqual(summary["n_infeasible_results"], 1)
        self.assertEqual(summary["n_feasible_predictions"], 1)

    def test_invalid_package_hash_and_prediction_nondeterminism_fail(self):
        first = [self._metadata(101), self._result("q1", "A", 101), self._result("q2", "B", 101)]
        first[1]["package_sha256"] = "not-a-hash"
        repeat = [self._metadata(202), self._result("q1", "changed", 202), self._result("q2", "B", 202)]
        result = self._run(self._arm(first, repeat))
        arm = result["arms"]["toy"]
        self.assertFalse(arm["first_results"]["package_fields"]["valid"])
        self.assertFalse(arm["repeat_determinism"]["exact_prediction_determinism"])

    def test_parse_arm_accepts_comma_or_double_colon(self):
        comma = parse_arm("x=a.jsonl,t.log,b.jsonl", self.root)
        colons = parse_arm("y=a.jsonl::t.log::b.jsonl", self.root)
        self.assertEqual(comma.name, "x")
        self.assertEqual(colons.trace, (self.root / "t.log").resolve())
        with self.assertRaisesRegex(ValueError, "NAME=FIRST"):
            parse_arm("broken", self.root)

    def test_unparsed_openat_is_fail_closed(self):
        trace = self.root / "bad.log"
        trace.write_text(
            "101 openat(AT_FDCWD, <unreadable>, O_RDONLY) = 3\n"
            "101 +++ exited with 0 +++\n",
            encoding="utf-8",
        )
        summary = audit_trace(
            trace,
            101,
            {self.source},
            {self.source.name},
            {self.source.parent},
            self.root,
        )
        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(summary["unparsed_openat_calls"], 1)


if __name__ == "__main__":
    unittest.main()
