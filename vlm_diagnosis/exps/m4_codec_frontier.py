"""Controlled M4 smoke over byte-bounded raster memory packages.

This runner is intentionally narrower than the planned paper-grade M4.  It
tests whether a six-task, within-image mechanism set and physical byte-bounded
JPEG/WebP/optional AVIF packages can expose selective information loss.  It
does not claim that synthetic UI results generalize to real agent trajectories.

The model is always given the decoded package bytes for codec conditions.  The
source image is used only to build the package in this first single-process
smoke; D0's separate-process harness is the strict source-denial gate.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
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
    resize_image_long_side,
)
from vlm_diagnosis.core.loader import load_vlm
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.metrics import anls, exact_match, score_sample
from vlm_diagnosis.core.spans import token_spans


ROOT = Path(__file__).resolve().parents[2]
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."
SCHEMA_VERSION = "1.0"


def questions_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt controlled M4 or the audited real ScreenQA T4 pilot schema."""
    if "questions" in row:
        return row["questions"]
    questions: list[dict[str, Any]] = []
    for source in row.get("content_questions", []):
        question = dict(source)
        draft = str(question.get("type_draft", "unknown")).lower()
        primary = {
            "ocr/semantic": "ocr_semantic_ambiguous",
            "grounding": "state_grounding_text_qa",
            "count": "count",
        }.get(draft, draft)
        question.update({
            "primary_task_type": primary,
            "answer_type": "text",
            "acceptable_answers": question.get("answers", []),
        })
        questions.append(question)
    for source in row.get("location_questions", []):
        question = dict(source)
        answers = list(question.get("answers", []))
        if question.get("template") == "half":
            answers.extend(answer.removesuffix(" half") for answer in answers
                           if answer.endswith(" half"))
        question.update({
            "primary_task_type": "layout",
            "answer_type": "text",
            "acceptable_answers": list(dict.fromkeys(answers)),
            "evidence_bboxes": [question["source_bbox"]]
            if question.get("source_bbox") else [],
        })
        questions.append(question)
    if not questions:
        raise ValueError(f"row {row.get('sample_id')} has no supported questions")
    return questions


def question_metric_inputs(question: dict[str, Any]) -> tuple[str, list[str], Any]:
    answers = question.get("acceptable_answers", question.get("answers", []))
    bbox = question.get("target_bbox")
    answer_type = str(question.get("answer_type", "text")).lower()
    primary = str(question.get("primary_task_type", question.get("task_type", "qa"))).lower()
    metric_task = "grounding" if answer_type == "coordinate" or primary == "grounding" else primary
    return metric_task, answers, bbox


def question_prompt(question: dict[str, Any]) -> str:
    prompt = question["question"]
    answer_type = str(question.get("answer_type", "text")).lower()
    return prompt if answer_type == "coordinate" else prompt + BRIEF


def _shard_path(path: Path, shard: int, nshards: int) -> Path:
    if nshards == 1:
        return path
    return path.with_name(f"{path.stem}.shard{shard}{path.suffix}")


def _done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("record_type") == "question_result":
            done.add((str(row["sample_id"]), str(row["question_id"]), row["condition_id"]))
    return done


def _representations(
    source: Image.Image,
    source_bytes: int,
    codecs: list[str],
    budgets: list[int],
    max_long_sides: list[int | None] | None = None,
):
    max_long_sides = [None] if max_long_sides is None else max_long_sides
    yield {
        "condition_id": "SOURCE_IMAGE",
        "representation": "source_image",
        "codec": source.format,
        "target_bytes": None,
        "package_bytes": source_bytes,
        "budget_utilization": None,
        "quality": None,
        "feasible": True,
        "image": source.copy().convert("RGB"),
        "source_denial_enforced": False,
        "package_bytes_scope": "existing image container",
        "resize_max_long_side": None,
        "resampled": False,
        "source_width": source.width,
        "source_height": source.height,
        "spatial_scale": 1.0,
    }
    for codec in codecs:
        for max_long_side in max_long_sides:
            encoded_source = (
                source if max_long_side is None
                else resize_image_long_side(source, max_long_side)
            )
            resampled = encoded_source.size != source.size
            spatial_scale = (
                encoded_source.width * encoded_source.height
                / (source.width * source.height)
            )
            resize_label = (
                "source" if max_long_side is None else f"long{max_long_side}px"
            )
            for budget in budgets:
                base_id = f"{codec.upper()}@{budget // 1024}KiB"
                condition_id = (
                    base_id if max_long_sides == [None]
                    else f"{base_id}@{resize_label}"
                )
                started = time.perf_counter()
                common = {
                    "resize_max_long_side": max_long_side,
                    "resampled": resampled,
                    "source_width": source.width,
                    "source_height": source.height,
                    "spatial_scale": spatial_scale,
                }
                try:
                    result = encode_image_to_budget(encoded_source, codec, budget)
                except CodecUnavailableError as error:
                    yield {
                        "condition_id": condition_id,
                        "representation": "raster_codec",
                        "codec": codec.upper(),
                        "target_bytes": budget,
                        "package_bytes": None,
                        "budget_utilization": None,
                        "quality": None,
                        "feasible": False,
                        "infeasible_reason": str(error),
                        "encode_seconds": time.perf_counter() - started,
                        "image": None,
                        "source_denial_enforced": False,
                        "package_bytes_scope": "image container only",
                        **common,
                    }
                    continue
                record = asdict(result)
                payload = record.pop("payload")
                record.update({
                    "condition_id": condition_id,
                    "representation": "raster_codec",
                    "package_bytes": result.serialized_bytes,
                    "budget_utilization": result.budget_utilization,
                    "encode_seconds": time.perf_counter() - started,
                    "source_denial_enforced": False,
                    "package_bytes_scope": "image container only",
                    **common,
                })
                if result.feasible:
                    decode_started = time.perf_counter()
                    with Image.open(BytesIO(payload)) as decoded:
                        record["image"] = decoded.convert("RGB").copy()
                    record["decode_seconds"] = time.perf_counter() - decode_started
                else:
                    record["image"] = None
                    record["infeasible_reason"] = result.reason
                yield record


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    parser.add_argument("--codecs", default="jpeg,webp,avif")
    parser.add_argument("--budgets-kib", default="32,64,128,256")
    parser.add_argument(
        "--max-long-sides",
        default="source",
        help="comma-separated declared resolution arms, e.g. source,1024,768",
    )
    parser.add_argument("--task-types", default=None,
                        help="optional comma-separated primary task types")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--questions-per-image", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--out", default="results/smoke/m4_codec_frontier.jsonl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = (ROOT / args.manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = rows[args.shard::args.nshards]
    if args.limit is not None:
        rows = rows[:args.limit]
    codecs = [item.strip().lower() for item in args.codecs.split(",") if item.strip()]
    budgets = [int(item) * 1024 for item in args.budgets_kib.split(",") if item]
    max_long_sides: list[int | None] = []
    for item in args.max_long_sides.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "source":
            max_long_sides.append(None)
        else:
            value = int(item)
            if value < 1:
                raise ValueError("max-long-sides values must be positive")
            max_long_sides.append(value)
    if not max_long_sides or len(set(max_long_sides)) != len(max_long_sides):
        raise ValueError("max-long-sides must be non-empty and unique")
    task_types = ({item.strip().lower() for item in args.task_types.split(",") if item.strip()}
                  if args.task_types else None)
    out_path = _shard_path((ROOT / args.out).resolve(), args.shard, args.nshards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out_path) if args.resume else set()
    model, processor = load_vlm(args.model, device=args.device, max_pixels=MAX_PIXELS)
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            handle.write(json.dumps({
                "record_type": "run_metadata",
                "schema_version": SCHEMA_VERSION,
                "stage": "M4-S0",
                "run_kind": "controlled_mechanism_smoke",
                "manifest": args.manifest,
                "model": args.model,
                "device": args.device,
                "codecs": codecs,
                "budgets_bytes": budgets,
                "max_long_sides": max_long_sides,
                "future_question_used_to_encode": False,
                "strict_source_denial": False,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
        for image_index, row in enumerate(rows, 1):
            source_path = ROOT / row["image"]
            with Image.open(source_path) as opened:
                source = opened.copy()
                source.format = opened.format
            source_bytes = source_path.stat().st_size
            questions = questions_from_row(row)
            if task_types is not None:
                questions = [question for question in questions
                             if str(question.get("primary_task_type",
                                                 question.get("task_type", ""))).lower()
                             in task_types]
            if args.questions_per_image is not None:
                questions = questions[:args.questions_per_image]
            for representation in _representations(
                source, source_bytes, codecs, budgets, max_long_sides
            ):
                image = representation.pop("image")
                feasibility = {
                    key: value for key, value in representation.items()
                    if key not in {"encoder_settings"}
                }
                handle.write(json.dumps({
                    "record_type": "representation_feasibility",
                    "sample_id": str(row["sample_id"]),
                    "domain": row.get(
                        "domain", "mobile_ui" if row.get("dataset") == "ScreenQA" else None),
                    **feasibility,
                }, ensure_ascii=False) + "\n")
                handle.flush()
                if image is None:
                    continue
                for question in questions:
                    key = (str(row["sample_id"]), str(question["question_id"]),
                           representation["condition_id"])
                    if key in done:
                        continue
                    started = time.perf_counter()
                    inputs = S.vlm_inputs(
                        processor, image, question_prompt(question), args.device)
                    prediction = greedy_generate_masked(
                        model, processor, inputs,
                        max_new_tokens=args.max_new_tokens)
                    answer_seconds = time.perf_counter() - started
                    metric_task, answers, bbox = question_metric_inputs(question)
                    score = score_sample(prediction, metric_task, answers, bbox)
                    spans = token_spans(inputs["input_ids"], model.config)
                    record = {
                        "record_type": "question_result",
                        "sample_id": str(row["sample_id"]),
                        "domain": row.get(
                            "domain", "mobile_ui" if row.get("dataset") == "ScreenQA" else None),
                        "question_id": str(question["question_id"]),
                        "primary_task_type": question.get(
                            "primary_task_type", question.get("task_type")),
                        "answer_type": question.get("answer_type", "text"),
                        "question": question["question"],
                        "gold": answers,
                        "prediction": prediction,
                        "task_score": score,
                        "em": None if metric_task == "grounding" else exact_match(prediction, answers),
                        "anls": None if metric_task == "grounding" else anls(prediction, answers),
                        "n_visual_tokens": len(spans["visual"]),
                        "answer_seconds": answer_seconds,
                        **{key: value for key, value in representation.items()
                           if key not in {"encoder_settings", "reason"}},
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"[{image_index}/{len(rows)}] {row['sample_id']} "
                          f"{representation['condition_id']} {question['question_id']} "
                          f"score={score:.0f}", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
