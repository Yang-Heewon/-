import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vlm_diagnosis.scripts.frontier_paddleocr import (
    _package_line,
    assert_image_only,
    build_parser,
    build_text_views,
    parse_ocr_result,
    write_packages,
)


class FakeEngine:
    def __init__(self):
        self.calls = 0
        self.closed = False
        self._params = {
            "text_detection_model_name": "fake-det-v1",
            "text_recognition_model_name": "fake-rec-v1",
        }

    def predict(self, image_path, **kwargs):
        self.calls += 1
        assert kwargs["return_word_box"] is True
        return [
            {
                # Deliberately use a polygon for the line and xyxy word boxes.
                "rec_texts": ["Café 한글"],
                "rec_scores": [0.987654321],
                "rec_polys": [[[10, 20], [110, 20], [110, 45], [10, 45]]],
                "rec_boxes": [[10, 20, 110, 45]],
                "text_word": [["Café", "한글"]],
                "text_word_boxes": [[[10, 20, 65, 45], [66, 20, 110, 45]]],
            }
        ]

    def close(self):
        self.closed = True


class FrontierPaddleOCRTests(unittest.TestCase):
    def test_query_blind_contract_rejects_nested_questions(self):
        assert_image_only({"sample_id": "s", "image": "x.png"})
        with self.assertRaisesRegex(ValueError, "question-bearing"):
            assert_image_only(
                {"sample_id": "s", "image": "x.png", "meta": {"answers": ["x"]}}
            )

    def test_parse_result_has_line_and_word_coordinate_contract(self):
        lines = parse_ocr_result(
            FakeEngine().predict("unused", return_word_box=True)[0]
        )
        self.assertEqual(lines[0]["bbox_xyxy_px"], [10, 20, 110, 45])
        self.assertEqual(lines[0]["polygon_xy_px"][2], [110, 45])
        self.assertEqual(lines[0]["words"][1]["text"], "한글")
        self.assertEqual(
            lines[0]["words"][1]["confidence_source"],
            "inherited_from_parent_line",
        )
        recognized, layout = build_text_views(lines, 200, 100)
        self.assertEqual(recognized, "Café 한글")
        self.assertIn("width_px=200", layout)
        self.assertIn("bbox=[10,20,110,45]", layout)

    def test_package_byte_count_is_exact_for_unicode(self):
        line = _package_line({"sample_id": "한글-é", "payload": "테스트"})
        record = json.loads(line)
        self.assertTrue(line.endswith(b"\n"))
        self.assertEqual(record["package_bytes"], len(line))

    def test_write_and_resume_produce_source_free_exact_jsonl(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            image = root / "secret-source.png"
            Image.new("RGB", (200, 100), "white").save(image)
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            manifest = root / "write.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "screen-1",
                        "image": str(image),
                        "image_sha256": image_hash,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out = root / "ocr.jsonl"
            args = build_parser().parse_args(
                ["--manifest", str(manifest), "--out", str(out), "--limit", "1"]
            )
            first_engine = FakeEngine()
            write_packages(args, engine_factory=lambda _: first_engine)
            self.assertEqual(first_engine.calls, 1)
            self.assertTrue(first_engine.closed)

            raw_line = out.read_bytes()
            record = json.loads(raw_line)
            self.assertEqual(record["package_bytes"], len(raw_line))
            self.assertEqual(record["recognized_text_utf8_bytes"], len("Café 한글".encode()))
            self.assertEqual(record["source_sha256"], image_hash)
            self.assertFalse(record["source_path_stored"])
            self.assertNotIn(str(image), raw_line.decode("utf-8"))
            self.assertNotIn("image", record)
            self.assertEqual(record["coordinate_schema"]["origin"], "top_left")

            args.resume = True
            second_engine = FakeEngine()
            write_packages(args, engine_factory=lambda _: second_engine)
            self.assertEqual(second_engine.calls, 0)
            self.assertTrue(second_engine.closed)
            self.assertEqual(out.read_bytes(), raw_line)


if __name__ == "__main__":
    unittest.main()
