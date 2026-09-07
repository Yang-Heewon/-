"""runner 의 resume 설정 검사: 같은 설정만 이어서 실행, 다른 설정·손상 파일은 거부."""
import json
import os
import tempfile
import unittest

from vlm_diagnosis.exps.context_only_kv import check_resume, RESUME_KEYS


def _meta(**over):
    m = {k: f"v_{k}" for k in RESUME_KEYS}
    m.update({"record_type": "run", "run_id": "old", "keep_ratios": [0.2], "methods": ["d"], "random_seeds": [0, 1]})
    m.update(over)
    return m


class ResumeTest(unittest.TestCase):
    def _write(self, lines):
        fd, path = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
        with open(path, "w") as f:
            for l in lines:
                f.write(l if isinstance(l, str) else json.dumps(l)); f.write("\n")
        self.addCleanup(os.remove, path)
        return path

    def test_same_settings_resume_returns_done_contexts(self):
        path = self._write([_meta(), {"record_type": "context_done", "context_id": "a"},
                            {"record_type": "answer", "context_id": "b"}])
        meta = _meta(run_id="new")
        self.assertEqual(check_resume(path, meta), {"a"})
        self.assertEqual(meta["resumed_from"], "old")

    def test_changed_setting_is_refused(self):
        for key, value in (("keep_ratios", [0.5]), ("methods", ["d", "r"]), ("eps", "other"), ("dtype", "float32"),
                           ("manifest_sha256", "x"), ("random_seeds", [0])):
            with self.subTest(key=key), self.assertRaises(SystemExit) as cm:
                check_resume(self._write([_meta()]), _meta(**{key: value}))
            self.assertIn(key, str(cm.exception))

    def test_missing_metadata_or_malformed_line_refused(self):
        with self.assertRaises(SystemExit):
            check_resume(self._write([{"record_type": "context_done", "context_id": "a"}]), _meta())
        with self.assertRaises(SystemExit):
            check_resume(self._write([_meta(), "{not json"]), _meta())


if __name__ == "__main__":
    unittest.main()
