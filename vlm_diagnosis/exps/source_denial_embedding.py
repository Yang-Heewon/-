"""D1 source-denial smoke with full Qwen2.5-VL projected visual tokens.

This is a validity/baseline path, not a learned compression method.  The write
worker sees an image-only D0 manifest and stores the output of Qwen2.5-VL's
visual merger (the vectors that normally replace ``image_pad`` token
embeddings).  The reader supports the original FP16 package and diagnostic
per-token symmetric INT8/packed-INT4 variants.  It sees only the question
manifest and that package, rebuilding language-model input embeddings without
an image, PIL, ``pixel_values``, or a source path.

The representation is deliberately named *full projected visual tokens*.  No
pooling or claim of a generic/model-independent embedding is made here.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from vlm_diagnosis.core.loader import assert_finite_logits, load_vlm
from vlm_diagnosis.core.metrics import anls, exact_match
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.tensor_quantization import (
    QUANTIZATION_SCHEMES,
    dequantize_tensor,
    quantize_tensor,
    validate_quantized_tensor,
)
from vlm_diagnosis.exps.source_denial_kv import (
    BRIEF,
    MAX_PIXELS,
    ROOT,
    _done_ids,
    _jsonl,
    _safe_sample_id,
    _sha256,
    _shard_path,
    _sharded,
    assert_source_free,
)


SCHEMA_VERSION = "1.0"
REPRESENTATION = "FULL_PROJECTED_VISUAL_TOKENS"
MODEL_FAMILY = "qwen25vl"
_QUESTION_KEYS = {
    "question",
    "questions",
    "content_questions",
    "location_questions",
    "answers",
    "acceptable_answers",
}


def projected_package_path(
    package_dir: Path, sample_id: Any, quantization: str = "fp16"
) -> Path:
    """Use the legacy FP16 name and collision-free integer package names."""
    quantization = str(quantization).lower()
    if quantization not in QUANTIZATION_SCHEMES:
        raise ValueError(f"unsupported quantization scheme={quantization!r}")
    stem = f"{_safe_sample_id(sample_id)}.full_projected_visual"
    suffix = ".pt" if quantization == "fp16" else f".{quantization}.pt"
    return package_dir / f"{stem}{suffix}"


def _sync_if_cuda(device: str) -> None:
    parsed = torch.device(device)
    if parsed.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(parsed)


def _model_type(model: Any) -> str:
    return str(getattr(model.config, "model_type", ""))


def _require_qwen25vl(model: Any) -> None:
    if _model_type(model) != "qwen2_5_vl":
        raise RuntimeError(
            "projected visual packages are implemented only for Qwen2.5-VL; "
            f"got model_type={_model_type(model)!r}"
        )


def assert_question_free(value: Any, trail: str = "root") -> None:
    """Fail closed if future-query material reaches a write worker."""
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _QUESTION_KEYS:
                raise ValueError(f"question-bearing key in write input: {trail}.{key}")
            assert_question_free(item, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_question_free(item, f"{trail}[{index}]")


def merge_prefix_and_raw_prompt(
    prefix_ids: torch.Tensor,
    raw_prompt_ids: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace one unexpanded image placeholder with the stored visual prefix.

    ``prefix_ids`` ends at the final expanded ``image_pad`` token.  The raw
    tokenizer-only prompt contains exactly one ``image_pad`` placeholder; its
    suffix therefore begins with ``vision_end`` and the future question.
    """
    if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
        raise ValueError("prefix_ids must have shape (1, prefix_length)")
    if raw_prompt_ids.ndim != 2 or raw_prompt_ids.shape[0] != 1:
        raise ValueError("raw_prompt_ids must have shape (1, prompt_length)")
    prefix_visual = (prefix_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    raw_visual = (raw_prompt_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if prefix_visual.numel() < 1:
        raise ValueError("stored prefix has no expanded image tokens")
    if int(prefix_visual[-1]) != prefix_ids.shape[1] - 1:
        raise ValueError("stored prefix must end at its final image token")
    if raw_visual.numel() != 1:
        raise ValueError("raw chat template must contain exactly one image placeholder")
    suffix = raw_prompt_ids[:, int(raw_visual[0]) + 1 :]
    return torch.cat((prefix_ids, suffix), dim=1)


def inject_projected_visual_tokens(
    full_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    projected_visual_tokens: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Return token embeddings with every ``image_pad`` vector replaced."""
    if full_ids.ndim != 2 or full_ids.shape[0] != 1:
        raise ValueError("full_ids must have shape (1, sequence_length)")
    if token_embeddings.ndim != 3 or token_embeddings.shape[:2] != full_ids.shape:
        raise ValueError("token_embeddings must have shape (1, sequence_length, hidden)")
    if projected_visual_tokens.ndim != 2:
        raise ValueError("projected_visual_tokens must have shape (visual_tokens, hidden)")
    visual = (full_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if visual.numel() != projected_visual_tokens.shape[0]:
        raise ValueError(
            "visual placeholder/embedding count mismatch: "
            f"{visual.numel()} != {projected_visual_tokens.shape[0]}"
        )
    if token_embeddings.shape[-1] != projected_visual_tokens.shape[-1]:
        raise ValueError(
            "language/visual hidden size mismatch: "
            f"{token_embeddings.shape[-1]} != {projected_visual_tokens.shape[-1]}"
        )
    result = token_embeddings.clone()
    result[0, visual] = projected_visual_tokens.to(
        device=result.device, dtype=result.dtype
    )
    return result


def projected_quantization_scheme(blob: dict[str, Any]) -> str:
    """Return ``fp16`` for both legacy and metadata-bearing FP16 packages."""
    metadata = blob.get("quantization")
    if metadata is None:
        return "fp16"
    if not isinstance(metadata, dict):
        raise RuntimeError("quantization metadata must be a dictionary")
    scheme = metadata.get("scheme")
    if scheme not in QUANTIZATION_SCHEMES:
        raise RuntimeError(f"unsupported projected quantization scheme={scheme!r}")
    return str(scheme)


def _quantized_payload_from_blob(blob: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "metadata": blob["quantization"],
        "data": blob["projected_visual_tokens"],
    }
    if "projected_visual_scales" in blob:
        payload["scales"] = blob["projected_visual_scales"]
    return payload


def decode_projected_visual_tokens(blob: dict[str, Any]) -> torch.Tensor:
    """Reconstruct FP16 visual vectors using package content only.

    Packages written before quantization metadata was introduced remain valid:
    their FP16 tensor is returned unchanged apart from CPU contiguity.
    """
    validate_projected_package(blob)
    if "quantization" not in blob:
        return blob["projected_visual_tokens"].to(
            device="cpu", dtype=torch.float16
        ).contiguous()
    return dequantize_tensor(_quantized_payload_from_blob(blob), dtype=torch.float16)


def validate_projected_package(blob: dict[str, Any]) -> None:
    """Validate the portable tensor contract before allocating it on a GPU."""
    required = {
        "schema_version",
        "representation",
        "sample_id",
        "source_sha256",
        "model_family",
        "model_id",
        "dtype",
        "boundary",
        "prefix_ids",
        "prefix_position_ids",
        "prefix_rope_delta",
        "image_grid_thw",
        "vis_start",
        "vis_end",
        "projected_visual_tokens",
    }
    missing = required.difference(blob)
    if missing:
        raise RuntimeError(f"incomplete projected visual package: {sorted(missing)}")
    assert_source_free(blob)
    if blob["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported schema_version={blob['schema_version']!r}")
    if blob["representation"] != REPRESENTATION:
        raise RuntimeError(f"wrong representation={blob['representation']!r}")
    if blob["model_family"] != MODEL_FAMILY:
        raise RuntimeError(f"wrong model_family={blob['model_family']!r}")
    source_hash = blob["source_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise RuntimeError("source_sha256 must be a 64-character hex digest")
    try:
        int(source_hash, 16)
    except ValueError as exc:
        raise RuntimeError("source_sha256 is not hexadecimal") from exc

    prefix_ids = blob["prefix_ids"]
    positions = blob["prefix_position_ids"]
    rope_delta = blob["prefix_rope_delta"]
    grid = blob["image_grid_thw"]
    visual_tokens = blob["projected_visual_tokens"]
    if not torch.is_tensor(prefix_ids) or prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
        raise RuntimeError("prefix_ids must have tensor shape (1, prefix_length)")
    if not torch.is_tensor(positions) or positions.shape != (3, 1, prefix_ids.shape[1]):
        raise RuntimeError("prefix_position_ids must have shape (3, 1, prefix_length)")
    if not torch.is_tensor(rope_delta) or rope_delta.shape != (1, 1):
        raise RuntimeError("prefix_rope_delta must have shape (1, 1)")
    if not torch.is_tensor(grid) or grid.shape != (1, 3):
        raise RuntimeError("image_grid_thw must have shape (1, 3)")
    if not torch.is_tensor(visual_tokens):
        raise RuntimeError("projected_visual_tokens must be a tensor")
    scheme = projected_quantization_scheme(blob)
    expected_dtype_name = {
        "fp16": "float16",
        "int8": "int8",
        "int4": "int4",
    }[scheme]
    if blob["dtype"] != expected_dtype_name:
        raise RuntimeError(
            f"package dtype {blob['dtype']!r} disagrees with {scheme!r} encoding"
        )
    if "quantization" not in blob:
        if visual_tokens.ndim != 2:
            raise RuntimeError(
                "legacy projected_visual_tokens must have shape (visual_tokens, hidden)"
            )
        if visual_tokens.dtype != torch.float16:
            raise RuntimeError("legacy projected_visual_tokens tensor must be float16")
        if "projected_visual_scales" in blob:
            raise RuntimeError("legacy FP16 package cannot contain quantization scales")
        encoded_shape = tuple(visual_tokens.shape)
    else:
        try:
            payload = _quantized_payload_from_blob(blob)
            validate_quantized_tensor(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid projected visual quantization: {exc}") from exc
        encoded_shape = tuple(blob["quantization"]["shape"])
    vis_start = int(blob["vis_start"])
    vis_end = int(blob["vis_end"])
    if not (0 <= vis_start <= vis_end < prefix_ids.shape[1]):
        raise RuntimeError("stored visual span lies outside prefix_ids")
    if vis_end != prefix_ids.shape[1] - 1:
        raise RuntimeError("prefix boundary must end at vis_end")
    span_length = vis_end - vis_start + 1
    visual_token_id = prefix_ids[0, vis_start]
    if not torch.all(prefix_ids[0, vis_start : vis_end + 1] == visual_token_id):
        raise RuntimeError("visual span is not a single repeated placeholder run")
    if vis_start > 0 and prefix_ids[0, vis_start - 1] == visual_token_id:
        raise RuntimeError("vis_start does not mark the first visual placeholder")
    if encoded_shape[0] != span_length:
        raise RuntimeError(
            "visual span/embedding count mismatch: "
            f"{span_length} != {encoded_shape[0]}"
        )


def _prefix_rope_metadata(model: Any, prefix_ids: torch.Tensor,
                          grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    core = model.model if hasattr(model.model, "get_rope_index") else model
    attention = torch.ones_like(prefix_ids)
    return core.get_rope_index(
        prefix_ids, image_grid_thw=grid, attention_mask=attention
    )


@torch.no_grad()
def _write_one(model: Any, processor: Any, row: dict[str, Any], device: str,
               destination: Path, model_name: str,
               quantization: str = "fp16") -> dict[str, Any]:
    # PIL is deliberately scoped to the image-only writer.
    from PIL import Image

    quantization = str(quantization).lower()
    if quantization not in QUANTIZATION_SCHEMES:
        raise ValueError(f"unsupported quantization scheme={quantization!r}")
    _require_qwen25vl(model)
    image_path = ROOT / row["image"]
    source_sha256 = _sha256(image_path)
    if row.get("image_sha256") and source_sha256 != row["image_sha256"]:
        raise RuntimeError(f"image hash mismatch: {row['sample_id']}")

    preprocessing_started = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "x"}
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    spans = token_spans(inputs["input_ids"], model.config)
    prefix_len = spans["vis_end"] + 1
    prefix_ids = inputs["input_ids"][:, :prefix_len]
    prefix_positions, prefix_rope_delta = _prefix_rope_metadata(
        model, prefix_ids, inputs["image_grid_thw"]
    )

    _sync_if_cuda(device)
    encoding_started = time.perf_counter()
    image_features = model.get_image_features(
        inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"]
    )
    projected = torch.cat(image_features, dim=0).to(torch.float16)
    _sync_if_cuda(device)
    encoding_seconds = time.perf_counter() - encoding_started
    if projected.shape[0] != len(spans["visual"]):
        raise RuntimeError(
            "Qwen visual merger output does not match image placeholders: "
            f"{projected.shape[0]} != {len(spans['visual'])}"
        )

    quantization_started = time.perf_counter()
    encoded = quantize_tensor(projected, quantization)
    quantization_seconds = time.perf_counter() - quantization_started
    quantization_metadata = encoded["metadata"]

    blob = {
        "schema_version": SCHEMA_VERSION,
        "representation": REPRESENTATION,
        "sample_id": str(row["sample_id"]),
        "source_sha256": source_sha256,
        "model_family": model_name,
        "model_id": model.config._name_or_path,
        "dtype": {
            "fp16": "float16",
            "int8": "int8",
            "int4": "int4",
        }[quantization],
        "boundary": "system+vision_start+all_image_tokens; future question excluded",
        "prefix_ids": prefix_ids.cpu(),
        "prefix_position_ids": prefix_positions.cpu(),
        "prefix_rope_delta": prefix_rope_delta.cpu(),
        "image_grid_thw": inputs["image_grid_thw"].cpu(),
        "vis_start": int(spans["visual"].min()),
        "vis_end": int(spans["vis_end"]),
        "projected_visual_tokens": encoded["data"],
        "quantization": quantization_metadata,
    }
    if "scales" in encoded:
        blob["projected_visual_scales"] = encoded["scales"]
    validate_projected_package(blob)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialization_started = time.perf_counter()
    torch.save(blob, destination)
    serialization_seconds = time.perf_counter() - serialization_started
    return {
        "sample_id": str(row["sample_id"]),
        "package": str(destination.relative_to(ROOT)),
        "package_bytes": destination.stat().st_size,
        "package_sha256": _sha256(destination),
        "source_sha256": source_sha256,
        "n_prefix_tokens": prefix_len,
        "n_visual_tokens": projected.shape[0],
        "visual_hidden_size": projected.shape[1],
        "quantization": quantization,
        "quantization_metadata": quantization_metadata,
        "fp16_payload_bytes": quantization_metadata["reference_fp16_bytes"],
        "stored_payload_tensor_bytes": quantization_metadata[
            "payload_tensor_bytes"
        ],
        "payload_compression_ratio_vs_fp16": quantization_metadata[
            "payload_compression_ratio_vs_fp16"
        ],
        "quantization_error": quantization_metadata["error_stats"],
        "preprocessing_seconds": preprocessing_seconds,
        "visual_encoding_seconds": encoding_seconds,
        "quantization_seconds": quantization_seconds,
        "serialization_seconds": serialization_seconds,
        "writer_pid": os.getpid(),
    }


def _load_package(
    path: Path, expected_model: str, expected_quantization: str | None = None
) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    validate_projected_package(blob)
    if blob["model_family"] != expected_model:
        raise RuntimeError(
            f"package model {blob['model_family']} != reader {expected_model}"
        )
    if (
        expected_quantization is not None
        and projected_quantization_scheme(blob) != expected_quantization
    ):
        raise RuntimeError(
            "package quantization "
            f"{projected_quantization_scheme(blob)!r} != reader "
            f"{expected_quantization!r}"
        )
    return blob


def _raw_question_ids(processor: Any, question: str, device: str) -> torch.Tensor:
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def _full_position_ids(model: Any, full_ids: torch.Tensor,
                       grid: torch.Tensor) -> torch.Tensor:
    core = model.model if hasattr(model.model, "get_rope_index") else model
    positions, _ = core.get_rope_index(
        full_ids,
        image_grid_thw=grid,
        attention_mask=torch.ones_like(full_ids),
    )
    return positions


@torch.no_grad()
def _read_one(model: Any, processor: Any, blob: dict[str, Any], question: str,
              device: str, max_new_tokens: int) -> tuple[str, dict[str, float]]:
    _require_qwen25vl(model)
    if blob["model_id"] != model.config._name_or_path:
        raise RuntimeError(
            f"package checkpoint {blob['model_id']} != reader {model.config._name_or_path}"
        )
    reconstruction_started = time.perf_counter()
    prefix_ids = blob["prefix_ids"].to(device)
    raw_ids = _raw_question_ids(processor, question, device)
    full_ids = merge_prefix_and_raw_prompt(
        prefix_ids, raw_ids, model.config.image_token_id
    )
    grid = blob["image_grid_thw"].to(device)
    positions = _full_position_ids(model, full_ids, grid)
    stored_prefix_positions = blob["prefix_position_ids"].to(device)
    if not torch.equal(positions[:, :, : prefix_ids.shape[1]], stored_prefix_positions):
        raise RuntimeError("reader mRoPE prefix disagrees with stored writer metadata")
    token_embeddings = model.model.get_input_embeddings()(full_ids)
    projected_visual_tokens = decode_projected_visual_tokens(blob)
    inputs_embeds = inject_projected_visual_tokens(
        full_ids,
        token_embeddings,
        projected_visual_tokens,
        model.config.image_token_id,
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_started

    attention = torch.ones_like(full_ids)
    _sync_if_cuda(device)
    prefill_started = time.perf_counter()
    output = model(
        input_ids=full_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention,
        position_ids=positions,
        use_cache=True,
    )
    _sync_if_cuda(device)
    prefill_seconds = time.perf_counter() - prefill_started
    assert_finite_logits(output.logits, f"source_denial_projected:{blob['sample_id']}")

    next_id = output.logits[0, -1].argmax()
    generated = [int(next_id)]
    past = output.past_key_values
    next_position = int(positions.max()) + 1
    eos_value = model.config.eos_token_id
    eos = {eos_value} if isinstance(eos_value, int) else set(eos_value or [])
    _sync_if_cuda(device)
    decode_started = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        if int(next_id) in eos:
            break
        step_attention = torch.ones(
            1, past.get_seq_length() + 1, dtype=torch.long, device=device
        )
        step_position = torch.full(
            (3, 1, 1), next_position, dtype=positions.dtype, device=device
        )
        output = model(
            input_ids=next_id.view(1, 1),
            attention_mask=step_attention,
            position_ids=step_position,
            past_key_values=past,
            use_cache=True,
        )
        assert_finite_logits(
            output.logits, f"source_denial_projected_decode:{blob['sample_id']}"
        )
        past = output.past_key_values
        next_id = output.logits[0, -1].argmax()
        next_position += 1
        generated.append(int(next_id))
    _sync_if_cuda(device)
    decode_seconds = time.perf_counter() - decode_started
    prediction = processor.tokenizer.decode(
        generated, skip_special_tokens=True
    ).strip()
    return prediction, {
        "reconstruction_seconds": reconstruction_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
    }


def run_write(args: argparse.Namespace) -> None:
    manifest = (ROOT / args.manifest).resolve()
    rows = _sharded(_jsonl(manifest), args.shard, args.nshards, args.limit)
    for row in rows:
        assert_question_free(row)
    out_path = _shard_path((ROOT / args.out).resolve(), args.shard, args.nshards)
    package_dir = (ROOT / args.package_dir).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, processor = load_vlm(args.model, device=args.device, max_pixels=MAX_PIXELS)
    _require_qwen25vl(model)
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            handle.write(json.dumps({
                "record_type": "run_metadata",
                "schema_version": SCHEMA_VERSION,
                "stage": "D1",
                "mode": "write",
                "representation": REPRESENTATION,
                "quantization": args.quantization,
                "quantization_granularity": (
                    "none" if args.quantization == "fp16" else "per_token"
                ),
                "manifest": args.manifest,
                "manifest_sha256": _sha256(manifest),
                "model": args.model,
                "device": args.device,
                "future_questions_visible": False,
                "process_id": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
        for index, row in enumerate(rows, 1):
            destination = projected_package_path(
                package_dir, row["sample_id"], args.quantization
            )
            if args.resume and destination.exists():
                print(f"[write skip] {row['sample_id']}", flush=True)
                continue
            record = _write_one(
                model,
                processor,
                row,
                args.device,
                destination,
                args.model,
                args.quantization,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[write {index}/{len(rows)}] {row['sample_id']} "
                f"{record['package_bytes'] / 2**20:.1f} MiB",
                flush=True,
            )


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
    _require_qwen25vl(model)
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            handle.write(json.dumps({
                "record_type": "run_metadata",
                "schema_version": SCHEMA_VERSION,
                "stage": "D1",
                "mode": "read",
                "representation": REPRESENTATION,
                "quantization": args.quantization,
                "manifest": args.manifest,
                "manifest_sha256": _sha256(manifest),
                "model": args.model,
                "device": args.device,
                "source_path_available": False,
                "pixel_values_available": False,
                "process_id": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
        for index, row in enumerate(rows, 1):
            path = projected_package_path(
                package_dir, row["sample_id"], args.quantization
            )
            package_sha256 = _sha256(path)
            package_bytes = path.stat().st_size
            for question in row["questions"][: args.questions_per_image]:
                question_id = str(question["question_id"])
                key = (str(row["sample_id"]), question_id)
                if key in done:
                    continue
                load_started = time.perf_counter()
                blob = _load_package(path, args.model, args.quantization)
                package_load_seconds = time.perf_counter() - load_started
                prediction, timings = _read_one(
                    model,
                    processor,
                    blob,
                    question["question"] + BRIEF,
                    args.device,
                    args.max_new_tokens,
                )
                answers = question.get(
                    "answers", question.get("acceptable_answers", [])
                )
                record = {
                    "sample_id": str(row["sample_id"]),
                    "question_id": question_id,
                    "question": question["question"],
                    "gold": answers,
                    "task_type": question.get("task_type"),
                    "representation": blob["representation"],
                    "quantization": projected_quantization_scheme(blob),
                    "quantization_error": blob.get("quantization", {}).get(
                        "error_stats"
                    ),
                    "stored_payload_tensor_bytes": blob.get(
                        "quantization", {}
                    ).get("payload_tensor_bytes"),
                    "source_sha256": blob["source_sha256"],
                    "package_bytes": package_bytes,
                    "package_sha256": package_sha256,
                    "package_load_seconds": package_load_seconds,
                    **timings,
                    "prediction": prediction,
                    "em": exact_match(prediction, answers),
                    "anls": anls(prediction, answers),
                    "source_path_in_read_manifest": False,
                    "pixel_values_used": False,
                    "reader_pid": os.getpid(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[read {index}/{len(rows)}] "
                    f"{row['sample_id']}/{question_id} EM={record['em']:.0f} "
                    f"prefill={record['prefill_seconds']:.3f}s",
                    flush=True,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["write", "read"])
    parser.add_argument(
        "--manifest",
        required=True,
        help="D0 image-only write manifest or question-only read manifest",
    )
    parser.add_argument(
        "--package-dir",
        default="results/smoke/source_denial_projected_visual_packages",
    )
    parser.add_argument(
        "--out",
        help=(
            "JSONL output path. Defaults to the legacy FP16 filename or a "
            "quantization-specific INT8/INT4 filename."
        ),
    )
    parser.add_argument(
        "--quantization",
        default="fp16",
        choices=QUANTIZATION_SCHEMES,
        help=(
            "Stored projected-token precision. Integer modes use symmetric "
            "per-token scales; INT4 is packed two values per byte."
        ),
    )
    parser.add_argument("--model", default=MODEL_FAMILY, choices=[MODEL_FAMILY])
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
    if args.out is None:
        suffix = "" if args.quantization == "fp16" else f"_{args.quantization}"
        args.out = f"results/smoke/source_denial_projected_visual{suffix}.jsonl"
    if args.mode == "write":
        run_write(args)
    else:
        run_read(args)


if __name__ == "__main__":
    main()
