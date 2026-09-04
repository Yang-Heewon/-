"""Write source-free PaddleOCR memories from an image-only manifest.

This is the independent OCR arm of the byte-frontier experiments.  The
process is deliberately query blind: its input rows may contain an image and
sample metadata, but no questions or answers.  Each output JSONL line is a
self-contained text/layout memory and contains no source path.

Run with the repository-local OCR environment::

    CUDA_VISIBLE_DEVICES='' .venv/bin/python -m \
      vlm_diagnosis.scripts.frontier_paddleocr \
      --manifest results/smoke/source_denial_write.jsonl \
      --out results/smoke/paddleocr_memory.jsonl --limit 2

PaddleOCR is imported lazily so schema and accounting tests do not require the
optional OCR dependency.  The writer always requests the CPU explicitly and
hides CUDA devices before importing Paddle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "1.0"
REPRESENTATION = "PADDLEOCR_TEXT_LAYOUT"
QUESTION_KEYS = {
    "answer",
    "answers",
    "acceptable_answers",
    "content_questions",
    "future_question",
    "future_questions",
    "gold",
    "location_questions",
    "prediction",
    "question",
    "question_id",
    "questions",
}
JSON_KWARGS = {
    "ensure_ascii": False,
    "sort_keys": True,
    "separators": (",", ":"),
    "allow_nan": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, **JSON_KWARGS).encode("utf-8")


def _package_line(record: Mapping[str, Any]) -> bytes:
    """Serialize a record whose ``package_bytes`` includes its final newline.

    Since the decimal size field contributes to the size itself, update it to
    a fixed point.  Only the number of decimal digits can change, so this
    converges in a handful of iterations.
    """
    sized = dict(record)
    sized["package_bytes"] = 0
    for _ in range(16):
        line = _canonical_bytes(sized) + b"\n"
        actual = len(line)
        if sized["package_bytes"] == actual:
            return line
        sized["package_bytes"] = actual
    raise RuntimeError("package byte accounting did not converge")


def assert_image_only(value: Any, trail: str = "root") -> None:
    """Fail closed if future-query or answer material reaches the writer."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in QUESTION_KEYS or normalized.endswith("_question"):
                raise ValueError(f"question-bearing key in OCR input: {trail}.{key}")
            assert_image_only(item, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_image_only(item, f"{trail}[{index}]")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            rows.append(row)
    return rows


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _number(value: Any) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite OCR coordinate: {number}")
    rounded = round(number, 4)
    return int(rounded) if rounded.is_integer() else rounded


def _bbox(value: Any) -> list[int | float]:
    """Normalize either an xyxy box or a polygon to an enclosing xyxy box."""
    raw = _to_builtin(value)
    if not isinstance(raw, list) or not raw:
        raise ValueError("empty OCR box")
    if len(raw) == 4 and all(not isinstance(item, list) for item in raw):
        x0, y0, x1, y1 = (_number(item) for item in raw)
    else:
        points = raw
        if not all(isinstance(point, list) and len(point) >= 2 for point in points):
            raise ValueError(f"unrecognized OCR box shape: {raw!r}")
        xs = [_number(point[0]) for point in points]
        ys = [_number(point[1]) for point in points]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if x1 < x0 or y1 < y0:
        raise ValueError(f"inverted OCR box: {[x0, y0, x1, y1]}")
    return [x0, y0, x1, y1]


def _polygon(value: Any) -> list[list[int | float]]:
    raw = _to_builtin(value)
    if not isinstance(raw, list) or not raw:
        raise ValueError("empty OCR polygon")
    if len(raw) == 4 and all(not isinstance(item, list) for item in raw):
        x0, y0, x1, y1 = _bbox(raw)
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if not all(isinstance(point, list) and len(point) >= 2 for point in raw):
        raise ValueError(f"unrecognized OCR polygon shape: {raw!r}")
    return [[_number(point[0]), _number(point[1])] for point in raw]


def _result_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "json"):
        raw = result.json
        raw = raw.get("res", raw) if isinstance(raw, Mapping) else raw
    else:
        raw = result
    raw = _to_builtin(raw)
    if not isinstance(raw, dict):
        raise ValueError("PaddleOCR result is not a mapping")
    return raw


def parse_ocr_result(result: Any) -> list[dict[str, Any]]:
    """Convert a PaddleOCR 3.x result into the stable memory line schema."""
    raw = _result_payload(result)
    texts = list(raw.get("rec_texts") or [])
    scores = list(raw.get("rec_scores") or [])
    boxes = list(raw.get("rec_boxes") or [])
    polygons = list(raw.get("rec_polys") or [])
    word_texts = list(raw.get("text_word") or [])
    word_boxes = list(raw.get("text_word_boxes") or [])
    if not (len(texts) == len(scores) == len(polygons)):
        raise ValueError(
            "PaddleOCR line fields disagree: "
            f"texts={len(texts)} scores={len(scores)} polygons={len(polygons)}"
        )
    if boxes and len(boxes) != len(texts):
        raise ValueError(f"PaddleOCR rec_boxes disagree: {len(boxes)} != {len(texts)}")
    if word_texts and len(word_texts) != len(texts):
        raise ValueError(f"PaddleOCR text_word disagree: {len(word_texts)} != {len(texts)}")
    if word_boxes and len(word_boxes) != len(texts):
        raise ValueError(
            f"PaddleOCR text_word_boxes disagree: {len(word_boxes)} != {len(texts)}"
        )

    lines: list[dict[str, Any]] = []
    for index, text_value in enumerate(texts):
        text = str(text_value)
        confidence = float(scores[index])
        if not math.isfinite(confidence):
            raise ValueError(f"non-finite OCR confidence: {confidence}")
        polygon = _polygon(polygons[index])
        line_box = _bbox(boxes[index]) if boxes else _bbox(polygon)
        this_word_texts = list(word_texts[index]) if word_texts else []
        this_word_boxes = list(word_boxes[index]) if word_boxes else []
        if len(this_word_texts) != len(this_word_boxes):
            raise ValueError(
                f"word text/box mismatch on line {index}: "
                f"{len(this_word_texts)} != {len(this_word_boxes)}"
            )
        words = [
            {
                "word_id": f"l{index:04d}-w{word_index:04d}",
                "text": str(word),
                "bbox_xyxy_px": _bbox(box),
                # PaddleOCR exposes one recognition score per line, not per word.
                "confidence": round(confidence, 8),
                "confidence_source": "inherited_from_parent_line",
            }
            for word_index, (word, box) in enumerate(
                zip(this_word_texts, this_word_boxes)
            )
        ]
        lines.append(
            {
                "line_id": f"l{index:04d}",
                "text": text,
                "confidence": round(confidence, 8),
                "bbox_xyxy_px": line_box,
                "polygon_xy_px": polygon,
                "words": words,
            }
        )
    return lines


def build_text_views(
    lines: Sequence[Mapping[str, Any]], width: int, height: int
) -> tuple[str, str]:
    recognized_text = "\n".join(str(line["text"]) for line in lines)
    header = (
        f"@image width_px={width} height_px={height} "
        "origin=top_left bbox=xyxy_px"
    )
    layout_rows = [header]
    for line in lines:
        box = ",".join(str(value) for value in line["bbox_xyxy_px"])
        encoded_text = json.dumps(str(line["text"]), ensure_ascii=False)
        layout_rows.append(
            f"{line['line_id']} bbox=[{box}] confidence={line['confidence']} "
            f"text={encoded_text}"
        )
    return recognized_text, "\n".join(layout_rows)


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tree_fingerprint(path_value: str | None) -> dict[str, Any]:
    if path_value is None:
        return {"supplied": False, "tree_sha256": None}
    path = _resolve(path_value)
    if not path.is_dir():
        raise FileNotFoundError(f"OCR model directory not found: {path}")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = file_path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_count += 1
        total_bytes += size
    return {
        "supplied": True,
        "tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def writer_provenance(args: argparse.Namespace, engine: Any) -> dict[str, Any]:
    params = getattr(engine, "_params", {})
    detector_name = params.get("text_detection_model_name", args.det_model_name)
    recognizer_name = params.get("text_recognition_model_name", args.rec_model_name)
    return {
        "implementation": "vlm_diagnosis.scripts.frontier_paddleocr",
        "paddleocr_version": _installed_version("paddleocr"),
        "paddlepaddle_version": _installed_version("paddlepaddle"),
        "paddlex_version": _installed_version("paddlex"),
        "device": "cpu",
        "cuda_visible_devices": "",
        "language": args.lang,
        "ocr_version": args.ocr_version,
        "detector": {
            "model_name": detector_name,
            **_tree_fingerprint(args.det_model_dir),
        },
        "recognizer": {
            "model_name": recognizer_name,
            **_tree_fingerprint(args.rec_model_dir),
        },
        "provider_weights_revision": (
            "local_tree_sha256" if args.det_model_dir and args.rec_model_dir
            else "provider_default_unpinned"
        ),
        "weights_revision_pinned": bool(args.det_model_dir and args.rec_model_dir),
        "return_word_box": True,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": False,
        "text_rec_score_thresh": args.text_rec_score_thresh,
        "cpu_threads": args.cpu_threads,
    }


def _create_engine(args: argparse.Namespace) -> Any:
    # Do this before importing Paddle: this writer is a deliberately CPU-only arm.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    from paddleocr import PaddleOCR

    model_kwargs: dict[str, Any] = {}
    if args.det_model_name:
        model_kwargs["text_detection_model_name"] = args.det_model_name
    if args.det_model_dir:
        model_kwargs["text_detection_model_dir"] = str(_resolve(args.det_model_dir))
    if args.rec_model_name:
        model_kwargs["text_recognition_model_name"] = args.rec_model_name
    if args.rec_model_dir:
        model_kwargs["text_recognition_model_dir"] = str(_resolve(args.rec_model_dir))
    if not (args.det_model_name or args.det_model_dir or args.rec_model_name or args.rec_model_dir):
        model_kwargs.update(lang=args.lang, ocr_version=args.ocr_version)
    return PaddleOCR(
        **model_kwargs,
        device="cpu",
        cpu_threads=args.cpu_threads,
        # PaddlePaddle 3.3.1's oneDNN executor cannot lower an Array<Double>
        # attribute used by the cached PP-OCRv6 graph.  The plain CPU runner is
        # slower but portable and avoids that runtime-only failure.
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=args.text_rec_score_thresh,
        return_word_box=True,
    )


def _done_ids(
    path: Path, expected_manifest_sha256: str, expected_writer_config_sha256: str
) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if record.get("package_bytes") != len(raw_line):
                raise RuntimeError(f"bad package_bytes at output line {line_number}")
            if record.get("manifest_sha256") != expected_manifest_sha256:
                raise RuntimeError("--resume output was built from a different manifest")
            if record.get("writer_config_sha256") != expected_writer_config_sha256:
                raise RuntimeError("--resume output used a different OCR writer configuration")
            done.add(str(record["sample_id"]))
    return done


def _validate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        assert_image_only(row)
        missing = {"sample_id", "image"}.difference(row)
        if missing:
            raise ValueError(f"manifest row {index} missing {sorted(missing)}")
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in OCR manifest: {sample_id}")
        seen.add(sample_id)
        validated.append(row)
    return validated


def _write_one(
    row: dict[str, Any],
    engine: Any,
    provenance: dict[str, Any],
    manifest_sha256: str,
    writer_config_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    image_path = _resolve(row["image"])
    source_sha256 = _sha256(image_path)
    expected_hash = row.get("image_sha256")
    if expected_hash is not None and source_sha256 != str(expected_hash):
        raise RuntimeError(f"image hash mismatch: {row['sample_id']}")
    with Image.open(image_path) as image:
        width, height = image.size

    started = time.perf_counter()
    results = engine.predict(
        str(image_path),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=provenance["text_rec_score_thresh"],
        return_word_box=True,
    )
    inference_seconds = time.perf_counter() - started
    if len(results) != 1:
        raise RuntimeError(f"expected one OCR result, got {len(results)}")
    lines = parse_ocr_result(results[0])
    recognized_text, layout_text = build_text_views(lines, width, height)
    semantic_payload = {
        "recognized_text": recognized_text,
        "layout_text": layout_text,
        "lines": lines,
    }
    record = {
        "record_type": "ocr_memory_package",
        "schema_version": SCHEMA_VERSION,
        "representation": REPRESENTATION,
        "sample_id": str(row["sample_id"]),
        "source_sha256": source_sha256,
        "source_width_px": width,
        "source_height_px": height,
        "coordinate_schema": {
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "unit": "source_image_pixel",
            "bbox_order": ["x_min", "y_min", "x_max", "y_max"],
            "polygon_point_order": ["x", "y"],
            "bounds_semantics": "detector_geometry; no inclusive/exclusive pixel claim",
            "line_order": "PaddleOCR_return_order",
        },
        **semantic_payload,
        "recognized_text_utf8_bytes": len(recognized_text.encode("utf-8")),
        "layout_text_utf8_bytes": len(layout_text.encode("utf-8")),
        "semantic_payload_utf8_bytes": len(_canonical_bytes(semantic_payload)),
        "line_count": len(lines),
        "word_count": sum(len(line["words"]) for line in lines),
        "source_path_stored": False,
        "future_questions_visible": False,
        "manifest_sha256": manifest_sha256,
        "writer_config_sha256": writer_config_sha256,
        "writer": provenance,
        "writer_pid": os.getpid(),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "inference_seconds": inference_seconds,
    }
    line = _package_line(record)
    return line, json.loads(line)


def write_packages(
    args: argparse.Namespace,
    engine_factory: Callable[[argparse.Namespace], Any] = _create_engine,
) -> None:
    manifest = _resolve(args.manifest)
    rows = _validate_rows(_jsonl(manifest))
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        rows = rows[: args.limit]
    manifest_sha256 = _sha256(manifest)
    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    engine = engine_factory(args)
    try:
        provenance = writer_provenance(args, engine)
        writer_config_sha256 = hashlib.sha256(_canonical_bytes(provenance)).hexdigest()
        done = (
            _done_ids(out, manifest_sha256, writer_config_sha256)
            if args.resume else set()
        )
        mode = "ab" if args.resume else "wb"
        with out.open(mode) as handle:
            for index, row in enumerate(rows, 1):
                sample_id = str(row["sample_id"])
                if sample_id in done:
                    print(f"[OCR skip {index}/{len(rows)}] {sample_id}", flush=True)
                    continue
                line, record = _write_one(
                    row,
                    engine,
                    provenance,
                    manifest_sha256,
                    writer_config_sha256,
                )
                handle.write(line)
                handle.flush()
                print(
                    f"[OCR {index}/{len(rows)}] {sample_id} "
                    f"lines={record['line_count']} words={record['word_count']} "
                    f"bytes={record['package_bytes']}",
                    flush=True,
                )
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="image-only JSONL manifest")
    parser.add_argument("--out", required=True, help="source-free OCR package JSONL")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--ocr-version", default="PP-OCRv6")
    parser.add_argument("--det-model-name")
    parser.add_argument("--det-model-dir")
    parser.add_argument("--rec-model-name")
    parser.add_argument("--rec-model-dir")
    parser.add_argument("--text-rec-score-thresh", type=float, default=0.0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cpu_threads < 1:
        raise SystemExit("--cpu-threads must be positive")
    explicit_models = any(
        (args.det_model_name, args.det_model_dir, args.rec_model_name, args.rec_model_dir)
    )
    if explicit_models and not all(
        (args.det_model_name or args.det_model_dir, args.rec_model_name or args.rec_model_dir)
    ):
        raise SystemExit("explicit model selection requires both detector and recognizer")
    write_packages(args)


if __name__ == "__main__":
    main()
