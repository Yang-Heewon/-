"""Strict source-denial baseline for durable raster memory packages.

The writer receives an image-only manifest and creates either a byte-for-byte
copy of the source container or a JPEG/WebP/AVIF package below a physical byte
cap.  The reader receives only a question manifest and the package manifest;
it never receives the original image path.  Running the reader under
``strace -e openat`` provides the final audit that no file under ``data/`` was
opened.

This module is a measurement baseline, not a new image codec.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_diagnosis.core import signals as S
from vlm_diagnosis.core.byte_codecs import (
    CodecUnavailableError,
    encode_image_to_budget,
)
from vlm_diagnosis.core.loader import load_vlm
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.metrics import anls, exact_match, score_sample
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.exps.m4_codec_frontier import (
    question_metric_inputs,
    question_prompt,
)
from vlm_diagnosis.exps.source_denial_kv import (
    MAX_PIXELS,
    ROOT,
    _done_ids,
    _jsonl,
    _questions,
    _safe_sample_id,
    _sha256,
    _shard_path,
    _sharded,
    assert_source_free,
)
from vlm_diagnosis.exps.source_denial_embedding import assert_question_free


SCHEMA_VERSION = "1.0"
REPRESENTATION = "RASTER_MEMORY"
SUPPORTED_CODECS = {"copy", "jpeg", "webp", "avif"}


def raster_package_path(
    package_dir: Path,
    sample_id: Any,
    codec: str,
    budget_bytes: int | None,
    source_suffix: str = ".img",
) -> Path:
    """Return a deterministic, path-safe destination for one raster arm."""

    codec = codec.lower()
    if codec not in SUPPORTED_CODECS:
        raise ValueError(f"unsupported codec {codec!r}")
    slug = _safe_sample_id(sample_id)
    if codec == "copy":
        suffix = source_suffix.lower() if source_suffix else ".img"
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("invalid source suffix")
        return package_dir / f"{slug}.copy{suffix}"
    if budget_bytes is None or budget_bytes < 1:
        raise ValueError("compressed raster package requires a positive byte cap")
    return package_dir / f"{slug}.{codec}.{budget_bytes}B.{codec}"


def condition_id(codec: str, budget_bytes: int | None) -> str:
    codec = codec.lower()
    if codec == "copy":
        return "SOURCE_CONTAINER_COPY"
    if budget_bytes is None:
        raise ValueError("compressed condition requires budget_bytes")
    return f"{codec.upper()}@{budget_bytes}B"


def _metadata(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    return {
        "record_type": "run_metadata",
        "schema_version": SCHEMA_VERSION,
        "stage": "D0_RASTER",
        "mode": mode,
        "representation": REPRESENTATION,
        "manifest": args.manifest,
        "manifest_sha256": _sha256((ROOT / args.manifest).resolve()),
        "device": getattr(args, "device", None),
        "source_path_available": mode == "write",
        "future_questions_visible": False if mode == "write" else None,
        "process_id": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def run_write(args: argparse.Namespace) -> None:
    manifest = (ROOT / args.manifest).resolve()
    rows = _sharded(_jsonl(manifest), args.shard, args.nshards, args.limit)
    for row in rows:
        assert_question_free(row)
    codecs = [item.strip().lower() for item in args.codecs.split(",") if item.strip()]
    unknown = set(codecs).difference(SUPPORTED_CODECS)
    if unknown:
        raise ValueError(f"unsupported codecs: {sorted(unknown)}")
    budgets = [int(item) for item in args.budgets_bytes.split(",") if item.strip()]
    if any(budget < 1 for budget in budgets):
        raise ValueError("all raster byte caps must be positive")

    package_dir = (ROOT / args.package_dir).resolve()
    package_dir.mkdir(parents=True, exist_ok=True)
    out_path = _shard_path((ROOT / args.out).resolve(), args.shard, args.nshards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    completed: set[tuple[str, str]] = set()
    if args.resume and out_path.exists():
        for record in _jsonl(out_path):
            if record.get("record_type") == "package":
                completed.add((str(record["sample_id"]), str(record["condition_id"])))

    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            meta = _metadata(args, "write")
            meta.update({"codecs": codecs, "budgets_bytes": budgets})
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for row_index, row in enumerate(rows, 1):
            source_path = (ROOT / row["image"]).resolve()
            source_hash = _sha256(source_path)
            if row.get("image_sha256") and row["image_sha256"] != source_hash:
                raise RuntimeError(f"source hash mismatch: {row['sample_id']}")
            with Image.open(source_path) as opened:
                source = opened.convert("RGB")

            arms: list[tuple[str, int | None]] = []
            if "copy" in codecs:
                arms.append(("copy", None))
            arms.extend(
                (codec, budget)
                for codec in codecs if codec != "copy"
                for budget in budgets
            )
            for codec, budget in arms:
                arm_id = condition_id(codec, budget)
                if (str(row["sample_id"]), arm_id) in completed:
                    continue
                destination = raster_package_path(
                    package_dir, row["sample_id"], codec, budget, source_path.suffix
                )
                started = time.perf_counter()
                record: dict[str, Any] = {
                    "record_type": "package",
                    "schema_version": SCHEMA_VERSION,
                    "representation": REPRESENTATION,
                    "sample_id": str(row["sample_id"]),
                    "condition_id": arm_id,
                    "codec": codec.upper(),
                    "target_bytes": budget,
                    "source_sha256": source_hash,
                    "future_question_used_to_encode": False,
                    "writer_pid": os.getpid(),
                }
                try:
                    if codec == "copy":
                        shutil.copyfile(source_path, destination)
                        record.update({
                            "quality": None,
                            "feasible": True,
                            "budget_utilization": None,
                        })
                    else:
                        encoded = encode_image_to_budget(source, codec, int(budget))
                        record.update({
                            "quality": encoded.quality,
                            "feasible": encoded.feasible,
                            "budget_utilization": encoded.budget_utilization,
                            "smallest_tested_bytes": encoded.smallest_tested_bytes,
                            "infeasible_reason": encoded.reason,
                        })
                        if encoded.feasible:
                            destination.write_bytes(encoded.payload)
                except CodecUnavailableError as error:
                    record.update({
                        "quality": None,
                        "feasible": False,
                        "budget_utilization": None,
                        "infeasible_reason": str(error),
                    })

                if record["feasible"]:
                    record.update({
                        "package": str(destination.relative_to(ROOT)),
                        "package_bytes": destination.stat().st_size,
                        "package_sha256": _sha256(destination),
                    })
                    if budget is not None and record["package_bytes"] > budget:
                        raise RuntimeError("raster package exceeded its physical byte cap")
                else:
                    record.update({
                        "package": None,
                        "package_bytes": None,
                        "package_sha256": None,
                    })
                record["write_seconds"] = time.perf_counter() - started
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[raster write {row_index}/{len(rows)}] {row['sample_id']} "
                    f"{arm_id} feasible={record['feasible']}",
                    flush=True,
                )


def load_package_index(path: Path, selected_condition: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in _jsonl(path):
        if record.get("record_type") != "package":
            continue
        assert_source_free(record)
        if record.get("condition_id") != selected_condition:
            continue
        sample_id = str(record["sample_id"])
        if sample_id in index:
            raise ValueError(f"duplicate raster package for {sample_id}/{selected_condition}")
        index[sample_id] = record
    return index


def resolve_memory_package(path_value: str) -> Path:
    """Resolve a package while refusing any path inside the source-data tree."""

    package_path = (ROOT / path_value).resolve()
    data_root = (ROOT / "data").resolve()
    if package_path == data_root or package_path.is_relative_to(data_root):
        raise ValueError(f"raster reader refuses source-data path: {path_value}")
    if not package_path.is_relative_to(ROOT):
        raise ValueError(f"raster package must stay inside repository: {path_value}")
    return package_path


@torch.inference_mode()
def run_read(args: argparse.Namespace) -> None:
    manifest = (ROOT / args.manifest).resolve()
    rows = _sharded(_jsonl(manifest), args.shard, args.nshards, args.limit)
    for row in rows:
        assert_source_free(row)
    package_manifest = (ROOT / args.package_manifest).resolve()
    packages = load_package_index(package_manifest, args.condition_id)

    out_path = _shard_path((ROOT / args.out).resolve(), args.shard, args.nshards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path) if args.resume else set()
    model, processor = load_vlm(args.model, device=args.device, max_pixels=MAX_PIXELS)
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            meta = _metadata(args, "read")
            meta.update({
                "package_manifest": args.package_manifest,
                "package_manifest_sha256": _sha256(package_manifest),
                "condition_id": args.condition_id,
                "model": args.model,
            })
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for row_index, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            package = packages.get(sample_id)
            if package is None:
                raise KeyError(f"no {args.condition_id} package for sample {sample_id}")
            if not package.get("feasible"):
                continue
            package_path = resolve_memory_package(package["package"])
            if package_path.stat().st_size != package["package_bytes"]:
                raise RuntimeError(f"package size changed: {package_path}")
            if _sha256(package_path) != package["package_sha256"]:
                raise RuntimeError(f"package hash changed: {package_path}")
            payload = package_path.read_bytes()
            with Image.open(BytesIO(payload)) as opened:
                image = opened.convert("RGB")
            for question_index, question in enumerate(
                _questions(row)[: args.questions_per_image]
            ):
                question_id = str(question.get("question_id", f"q{question_index}"))
                if (sample_id, question_id) in done:
                    continue
                started = time.perf_counter()
                inputs = S.vlm_inputs(
                    processor, image, question_prompt(question), args.device
                )
                prediction = greedy_generate_masked(
                    model, processor, inputs, max_new_tokens=args.max_new_tokens
                )
                answer_seconds = time.perf_counter() - started
                metric_task, answers, bbox = question_metric_inputs(question)
                score = score_sample(prediction, metric_task, answers, bbox)
                spans = token_spans(inputs["input_ids"], model.config)
                record = {
                    "record_type": "question_result",
                    "sample_id": sample_id,
                    "question_id": question_id,
                    "question": question["question"],
                    "gold": answers,
                    "task_type": metric_task,
                    "representation": REPRESENTATION,
                    "condition_id": args.condition_id,
                    "package_bytes": package["package_bytes"],
                    "package_sha256": package["package_sha256"],
                    "prediction": prediction,
                    "task_score": score,
                    "em": None if metric_task == "grounding" else exact_match(prediction, answers),
                    "anls": None if metric_task == "grounding" else anls(prediction, answers),
                    "n_visual_tokens": len(spans["visual"]),
                    "answer_seconds": answer_seconds,
                    "source_path_in_read_manifest": False,
                    "reader_pid": os.getpid(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[raster read {row_index}/{len(rows)}] {sample_id}/{question_id} "
                    f"score={score:.0f}",
                    flush=True,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["write", "read"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--package-dir", default="results/smoke/source_denial_raster_packages")
    parser.add_argument("--package-manifest")
    parser.add_argument("--out", default="results/smoke/source_denial_raster.jsonl")
    parser.add_argument("--codecs", default="copy,jpeg,webp,avif")
    parser.add_argument("--budgets-bytes", default="32768,65536,131072,262144")
    parser.add_argument("--condition-id")
    parser.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--questions-per-image", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "write":
        run_write(args)
    else:
        if not args.package_manifest or not args.condition_id:
            raise SystemExit("read requires --package-manifest and --condition-id")
        run_read(args)


if __name__ == "__main__":
    main()
