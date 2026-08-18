"""D0 source-denial gate for durable visual memory.

The three modes are deliberately separate processes:

1. ``prepare`` splits an existing dataset row into an image-only write manifest
   and a question-only read manifest.
2. ``write`` sees only the write manifest and serializes a generic image-prefix
   KV package.  It cannot inspect future questions.
3. ``read`` sees only the read manifest and the package.  It reconstructs the
   prompt suffix from stored token/grid metadata and never opens an image.

This is a validity gate, not a compression method.  The first implementation
stores the full generic prefix+visual KV so that later sparse/quantized packages
can be checked against a known working source-free path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import DynamicCache

from vlm_diagnosis.core.loader import assert_finite_logits, load_vlm
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.metrics import anls, exact_match
from vlm_diagnosis.core.spans import token_spans


ROOT = Path(__file__).resolve().parents[2]
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."
SCHEMA_VERSION = "1.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _questions(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "questions" in row:
        return [dict(question) for question in row["questions"]]
    questions: list[dict[str, Any]] = []
    for question in row.get("content_questions", []):
        q = dict(question)
        q.setdefault("task_type", q.get("type_draft", "qa"))
        questions.append(q)
    for question in row.get("location_questions", []):
        q = dict(question)
        q.setdefault("task_type", "layout")
        questions.append(q)
    return questions


def split_manifest_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return strictly image-only and question-only views of one dataset row."""
    common = {
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "sample_id": str(row["sample_id"]),
    }
    write = {
        **common,
        "image": row["image"],
        "image_sha256": row.get("image_sha256"),
    }
    read = {**common, "questions": _questions(row)}
    assert "questions" not in write
    assert_source_free(read)
    return write, read


def assert_source_free(value: Any, trail: str = "root") -> None:
    """Reject image/source paths from anything passed to a read worker."""
    if isinstance(value, dict):
        for key, item in value.items():
            lower = str(key).lower()
            if lower in {"image", "image_path", "source_path", "pixel_values"}:
                raise ValueError(f"source-bearing key in read input: {trail}.{key}")
            assert_source_free(item, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_source_free(item, f"{trail}[{index}]")


def _safe_sample_id(sample_id: Any) -> str:
    raw = str(sample_id)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "sample"
    return f"{slug[:80]}-{hashlib.sha1(raw.encode()).hexdigest()[:10]}"


def package_path(package_dir: Path, sample_id: Any) -> Path:
    return package_dir / f"{_safe_sample_id(sample_id)}.full_kv.pt"


def _sharded(rows: list[dict[str, Any]], shard: int, nshards: int,
             limit: int | None) -> list[dict[str, Any]]:
    if nshards < 1 or not 0 <= shard < nshards:
        raise ValueError(f"invalid shard {shard}/{nshards}")
    selected = rows[shard::nshards]
    return selected[:limit] if limit is not None else selected


def _shard_path(path: Path, shard: int, nshards: int) -> Path:
    if nshards == 1:
        return path
    return path.with_name(f"{path.stem}.shard{shard}{path.suffix}")


def prepare_manifests(
    source: Path,
    write_path: Path,
    read_path: Path,
    limit: int | None = None,
    questions_per_image: int | None = None,
) -> None:
    """Materialize strict writer/reader manifests.

    When a read run intentionally evaluates only a prefix of each sample's
    questions, that selection is applied here rather than hidden inside the
    reader loop.  The resulting manifest is therefore the complete declared
    evaluation set and can be audited for full coverage.
    """
    if questions_per_image is not None and questions_per_image < 1:
        raise ValueError("questions_per_image must be positive")
    rows = _jsonl(source)
    if limit is not None:
        rows = rows[:limit]
    write_path.parent.mkdir(parents=True, exist_ok=True)
    read_path.parent.mkdir(parents=True, exist_ok=True)
    with write_path.open("w", encoding="utf-8") as write_handle, \
            read_path.open("w", encoding="utf-8") as read_handle:
        for row in rows:
            write, read = split_manifest_row(row)
            if questions_per_image is not None:
                source_question_count = len(read["questions"])
                read["questions"] = read["questions"][:questions_per_image]
                read["question_selection"] = {
                    "strategy": "manifest_order_prefix",
                    "requested_per_image": questions_per_image,
                    "source_question_count": source_question_count,
                    "selected_question_count": len(read["questions"]),
                }
                assert_source_free(read)
            write_handle.write(json.dumps(write, ensure_ascii=False) + "\n")
            read_handle.write(json.dumps(read, ensure_ascii=False) + "\n")


@torch.no_grad()
def _write_one(model, processor, row: dict[str, Any], device: str,
               destination: Path, model_name: str) -> dict[str, Any]:
    # PIL is imported only on the write path.  The read path has no image API.
    from PIL import Image

    image_path = ROOT / row["image"]
    source_sha256 = _sha256(image_path)
    if row.get("image_sha256"):
        if source_sha256 != row["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {row['sample_id']}")
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "x"}]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    spans = token_spans(inputs["input_ids"], model.config)
    prefix_len = spans["vis_end"] + 1
    prefix_ids = inputs["input_ids"][:, :prefix_len]
    attention = torch.ones_like(prefix_ids)
    position_ids = mrope_position_ids(
        model, prefix_ids, inputs["image_grid_thw"], attention)
    started = time.perf_counter()
    output = model(
        input_ids=prefix_ids,
        attention_mask=attention,
        position_ids=position_ids,
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        use_cache=True,
    )
    assert_finite_logits(output.logits, f"source_denial_write:{row['sample_id']}")
    write_seconds = time.perf_counter() - started
    legacy = output.past_key_values.to_legacy_cache()
    blob = {
        "schema_version": SCHEMA_VERSION,
        "representation": "FULL_GENERIC_PREFIX_KV",
        "sample_id": str(row["sample_id"]),
        "source_sha256": source_sha256,
        "model_family": model_name,
        "model_id": model.config._name_or_path,
        "dtype": "float16",
        "boundary": "system+vision_start+all_image_tokens; future question excluded",
        "prefix_ids": prefix_ids.cpu(),
        "prefix_position_ids": position_ids.cpu(),
        "image_grid_thw": inputs["image_grid_thw"].cpu(),
        "vis_start": int(spans["visual"].min()),
        "vis_end": int(spans["vis_end"]),
        "kv": [(key.cpu(), value.cpu()) for key, value in legacy],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, destination)
    return {
        "sample_id": str(row["sample_id"]),
        "package": str(destination.relative_to(ROOT)),
        "package_bytes": destination.stat().st_size,
        "package_sha256": _sha256(destination),
        "n_cached_tokens": prefix_len,
        "n_visual_tokens": len(spans["visual"]),
        "write_seconds": write_seconds,
        "writer_pid": os.getpid(),
    }


def _load_package(path: Path, device: str) -> tuple[dict[str, Any], DynamicCache]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema_version", "representation", "sample_id", "source_sha256",
                "model_family", "model_id", "prefix_ids", "image_grid_thw", "kv"}
    missing = required.difference(blob)
    if missing:
        raise RuntimeError(f"incomplete package {path}: {sorted(missing)}")
    cache = DynamicCache.from_legacy_cache(tuple(
        (key.to(device), value.to(device)) for key, value in blob["kv"]))
    if cache.get_seq_length() != blob["prefix_ids"].shape[1]:
        raise RuntimeError("cache length and stored prefix disagree")
    return blob, cache


def _question_suffix(processor, model, blob: dict[str, Any], question: str,
                     device: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    raw_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    raw_visual = (raw_ids[0] == model.config.image_token_id).nonzero(as_tuple=True)[0]
    if raw_visual.numel() != 1:
        raise RuntimeError("raw chat template must contain exactly one image placeholder")
    suffix = raw_ids[:, int(raw_visual[0]) + 1:]
    prefix = blob["prefix_ids"].to(device)
    full_ids = torch.cat((prefix, suffix), dim=1)
    grid = blob["image_grid_thw"].to(device)
    attention = torch.ones_like(full_ids)
    position_ids = mrope_position_ids(model, full_ids, grid, attention)
    return suffix, position_ids[:, :, prefix.shape[1]:], full_ids.shape[1]


@torch.no_grad()
def _read_one(model, processor, blob: dict[str, Any], cache: DynamicCache,
              question: str, device: str, max_new_tokens: int) -> tuple[str, float]:
    suffix, suffix_positions, full_len = _question_suffix(
        processor, model, blob, question, device)
    attention = torch.ones(1, full_len, dtype=torch.long, device=device)
    started = time.perf_counter()
    output = model(
        input_ids=suffix,
        attention_mask=attention,
        position_ids=suffix_positions,
        past_key_values=cache,
        use_cache=True,
    )
    assert_finite_logits(output.logits, f"source_denial_read:{blob['sample_id']}")
    first_token_seconds = time.perf_counter() - started
    next_id = output.logits[0, -1].argmax()
    generated = [int(next_id)]
    past = output.past_key_values
    next_position = int(suffix_positions.max()) + 1
    eos_value = model.config.eos_token_id
    eos = {eos_value} if isinstance(eos_value, int) else set(eos_value or [])
    for _ in range(max_new_tokens - 1):
        if int(next_id) in eos:
            break
        step_attention = torch.ones(
            1, past.get_seq_length() + 1, dtype=torch.long, device=device)
        step_position = torch.full(
            (3, 1, 1), next_position, dtype=suffix_positions.dtype, device=device)
        output = model(
            input_ids=next_id.view(1, 1),
            attention_mask=step_attention,
            position_ids=step_position,
            past_key_values=past,
            use_cache=True,
        )
        assert_finite_logits(output.logits, f"source_denial_decode:{blob['sample_id']}")
        past = output.past_key_values
        next_id = output.logits[0, -1].argmax()
        next_position += 1
        generated.append(int(next_id))
    prediction = processor.tokenizer.decode(
        generated, skip_special_tokens=True).strip()
    return prediction, first_token_seconds


def _done_ids(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for row in _jsonl(path):
        if row.get("record_type") != "run_metadata" and row.get("question_id"):
            done.add((str(row["sample_id"]), str(row["question_id"])))
    return done


def run_write(args: argparse.Namespace) -> None:
    manifest = (ROOT / args.manifest).resolve()
    rows = _sharded(_jsonl(manifest), args.shard, args.nshards, args.limit)
    out_path = _shard_path((ROOT / args.out).resolve(), args.shard, args.nshards)
    package_dir = (ROOT / args.package_dir).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, processor = load_vlm(args.model, device=args.device, max_pixels=MAX_PIXELS)
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            handle.write(json.dumps({
                "record_type": "run_metadata", "schema_version": SCHEMA_VERSION,
                "stage": "D0", "mode": "write", "manifest": args.manifest,
                "manifest_sha256": _sha256(manifest), "model": args.model,
                "device": args.device, "future_questions_visible": False,
                "process_id": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
        for index, row in enumerate(rows, 1):
            destination = package_path(package_dir, row["sample_id"])
            if args.resume and destination.exists():
                print(f"[write skip] {row['sample_id']}", flush=True)
                continue
            record = _write_one(
                model, processor, row, args.device, destination, args.model)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[write {index}/{len(rows)}] {row['sample_id']} "
                  f"{record['package_bytes'] / 2**20:.1f} MiB", flush=True)


def run_read(args: argparse.Namespace) -> None:
    manifest = (ROOT / args.manifest).resolve()
    rows = _sharded(_jsonl(manifest), args.shard, args.nshards, args.limit)
    for row in rows:
        assert_source_free(row)
    out_path = _shard_path((ROOT / args.out).resolve(), args.shard, args.nshards)
    package_dir = (ROOT / args.package_dir).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path) if args.resume else set()
    model, processor = load_vlm(args.model, device=args.device, max_pixels=MAX_PIXELS)
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            handle.write(json.dumps({
                "record_type": "run_metadata", "schema_version": SCHEMA_VERSION,
                "stage": "D0", "mode": "read", "manifest": args.manifest,
                "manifest_sha256": _sha256(manifest), "model": args.model,
                "device": args.device, "source_path_available": False,
                "process_id": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
        for index, row in enumerate(rows, 1):
            path = package_path(package_dir, row["sample_id"])
            package_sha256 = _sha256(path)
            fresh_cache = None
            for question in row["questions"][:args.questions_per_image]:
                question_id = str(question["question_id"])
                key = (str(row["sample_id"]), question_id)
                if key in done:
                    continue
                # Each question gets a fresh cache because decoding mutates DynamicCache.
                load_started = time.perf_counter()
                blob, fresh_cache = _load_package(path, args.device)
                load_seconds = time.perf_counter() - load_started
                if blob["model_family"] != args.model:
                    raise RuntimeError(
                        f"package model {blob['model_family']} != reader {args.model}")
                prediction, ttft = _read_one(
                    model, processor, blob, fresh_cache,
                    question["question"] + BRIEF, args.device,
                    args.max_new_tokens)
                answers = question.get("answers", question.get("acceptable_answers", []))
                record = {
                    "sample_id": str(row["sample_id"]),
                    "question_id": question_id,
                    "question": question["question"],
                    "gold": answers,
                    "task_type": question.get("task_type"),
                    "representation": blob["representation"],
                    "package_bytes": path.stat().st_size,
                    "package_sha256": package_sha256,
                    "package_load_seconds": load_seconds,
                    "first_token_seconds": ttft,
                    "prediction": prediction,
                    "em": exact_match(prediction, answers),
                    "anls": anls(prediction, answers),
                    "source_path_in_read_manifest": False,
                    "reader_pid": os.getpid(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[read {index}/{len(rows)}] {row['sample_id']}/{question_id} "
                      f"EM={record['em']:.0f} {ttft:.3f}s", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["prepare", "write", "read"])
    parser.add_argument("--manifest", help="source manifest (prepare) or split manifest")
    parser.add_argument("--write-manifest")
    parser.add_argument("--read-manifest")
    parser.add_argument("--package-dir", default="results/smoke/source_denial_packages")
    parser.add_argument("--out", default="results/smoke/source_denial.jsonl")
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
    if args.mode == "prepare":
        if not (args.manifest and args.write_manifest and args.read_manifest):
            raise SystemExit("prepare requires --manifest, --write-manifest, --read-manifest")
        prepare_manifests(
            (ROOT / args.manifest).resolve(),
            (ROOT / args.write_manifest).resolve(),
            (ROOT / args.read_manifest).resolve(),
            args.limit,
            args.questions_per_image,
        )
        return
    if not args.manifest:
        raise SystemExit(f"{args.mode} requires --manifest")
    if args.mode == "write":
        run_write(args)
    else:
        run_read(args)


if __name__ == "__main__":
    main()
