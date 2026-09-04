"""Controlled D3/D4 evaluator with a strict raster source-denial boundary.

The evaluator deliberately runs as three independent modes:

``prepare``
    Split the controlled episode manifest into a question-free, image-bearing
    writer manifest and a source-free trial/question reader manifest.
``write``
    Serialize each memory independently as a copied source container or as a
    JPEG/WebP/AVIF package under the requested physical byte caps.  No future
    question is visible to this process.
``read``
    Load only the reader manifest and raster package manifest.  Run the D3
    oracle-selected and inference-interference arms and the four D4 temporal
    conditions with Qwen2.5-VL multi-image prompts.  Original files under
    ``data/`` are rejected and never opened.

This is a controlled mechanism probe.  It does not implement a learned
retriever: D3 retrieval rows are evaluated through their explicitly labelled
``oracle_task_memory_ids`` so retrieval recall and inference interference are
not conflated.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image

from vlm_diagnosis.core.byte_codecs import (
    CodecUnavailableError,
    encode_image_to_budget,
)
from vlm_diagnosis.core.loader import load_vlm
from vlm_diagnosis.core.metrics import anls, exact_match
from vlm_diagnosis.exps.source_denial_embedding import assert_question_free
from vlm_diagnosis.exps.source_denial_kv import (
    MAX_PIXELS,
    ROOT,
    _jsonl,
    _safe_sample_id,
    _sha256,
    _shard_path,
    _sharded,
    assert_source_free,
)
from vlm_diagnosis.exps.source_denial_raster import resolve_memory_package
from vlm_diagnosis.scripts.gen_memory_dynamics import validate_manifest_row


SCHEMA_VERSION = "memory-dynamics-eval-v1"
REPRESENTATION = "BYTE_BOUNDED_RASTER_MEMORY"
SUPPORTED_CODECS = {"copy", "jpeg", "webp", "avif"}
DEFAULT_BUDGETS = (65536, 131072, 262144)
DEFAULT_ARMS = ("d3_oracle", "d3_interference", "d4")


def _resolve_repo_or_absolute(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_episode_row(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return question-free memory rows and one source-free episode row."""

    validate_manifest_row(row)
    common = {
        "dataset": row["dataset"],
        "dataset_revision": row["dataset_revision"],
        "split": row["split"],
        "episode_id": row["episode_id"],
    }
    write_rows = []
    memory_ids = []
    for memory in row["memories"]:
        memory_id = str(memory["memory_id"])
        memory_ids.append(memory_id)
        write_row = {
            **common,
            "memory_id": memory_id,
            "image": memory["image"],
            "image_sha256": memory.get("image_sha256"),
        }
        assert_question_free(write_row)
        write_rows.append(write_row)

    read_row = {
        **common,
        "synthetic": row["synthetic"],
        "probe_kind": row["probe_kind"],
        "claim_scope": row["claim_scope"],
        "factorial": copy.deepcopy(row["factorial"]),
        "available_memory_ids": memory_ids,
        "question": copy.deepcopy(row["question"]),
        "d3_trials": {
            "retrieval": copy.deepcopy(row["d3_trials"]["retrieval"]),
            "interference": copy.deepcopy(row["d3_trials"]["interference"]),
        },
        "d4_trials": copy.deepcopy(row["d4_trials"]),
    }
    assert_source_free(read_row)
    return write_rows, read_row


def prepare_manifests(
    source: Path,
    write_path: Path,
    read_path: Path,
    limit: int | None = None,
) -> tuple[int, int]:
    """Split episode rows while preserving the future-question boundary."""

    rows = _jsonl(source)
    if limit is not None:
        rows = rows[:limit]
    write_rows: list[dict[str, Any]] = []
    read_rows: list[dict[str, Any]] = []
    for row in rows:
        memory_rows, read_row = split_episode_row(row)
        write_rows.extend(memory_rows)
        read_rows.append(read_row)
    _write_jsonl(write_path, write_rows)
    _write_jsonl(read_path, read_rows)
    return len(write_rows), len(read_rows)


def package_condition(codec: str, budget_bytes: int | None) -> str:
    codec = codec.strip().lower()
    if codec not in SUPPORTED_CODECS:
        raise ValueError(f"unsupported codec: {codec!r}")
    if codec == "copy":
        if budget_bytes is not None:
            raise ValueError("copy packages do not have an encoder target")
        return "SOURCE_CONTAINER_COPY"
    if budget_bytes is None or budget_bytes < 1:
        raise ValueError("compressed packages require a positive byte cap")
    return f"{codec.upper()}@{budget_bytes}B"


def memory_package_path(
    package_dir: Path,
    memory_id: str,
    codec: str,
    budget_bytes: int | None,
    source_suffix: str = ".img",
) -> Path:
    """Build a deterministic path without exposing ``memory_id`` traversal."""

    codec = codec.strip().lower()
    condition = package_condition(codec, budget_bytes)
    slug = _safe_sample_id(memory_id)
    if codec == "copy":
        suffix = source_suffix.lower()
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("invalid source suffix")
    else:
        suffix = f".{codec}"
    condition_slug = condition.lower().replace("@", ".").replace("_", "-")
    return package_dir / f"{slug}.{condition_slug}{suffix}"


def _metadata(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    manifest = _resolve_repo_or_absolute(args.manifest)
    return {
        "record_type": "run_metadata",
        "schema_version": SCHEMA_VERSION,
        "stage": "D3_D4_CONTROLLED",
        "mode": mode,
        "representation": REPRESENTATION,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(manifest),
        "source_path_available": mode == "write",
        "future_questions_visible": False if mode == "write" else None,
        "process_id": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def run_write(args: argparse.Namespace) -> None:
    """Create one package per memory and representation/budget condition."""

    manifest = _resolve_repo_or_absolute(args.manifest)
    rows = _sharded(_jsonl(manifest), args.shard, args.nshards, args.limit)
    for row in rows:
        assert_question_free(row)
        if "image" not in row or "memory_id" not in row:
            raise ValueError("writer rows require image and memory_id")

    codecs = [value.strip().lower() for value in args.codecs.split(",") if value.strip()]
    unknown = set(codecs).difference(SUPPORTED_CODECS)
    if unknown:
        raise ValueError(f"unsupported codecs: {sorted(unknown)}")
    budgets = sorted({
        int(value) for value in args.budgets_bytes.split(",") if value.strip()
    })
    if any(value < 1 for value in budgets):
        raise ValueError("all byte caps must be positive")

    package_dir = _resolve_repo_or_absolute(args.package_dir)
    repo_root = ROOT.resolve()
    data_root = (ROOT / "data").resolve()
    if not package_dir.is_relative_to(repo_root):
        raise ValueError("package_dir must stay inside the repository")
    if package_dir == data_root or package_dir.is_relative_to(data_root):
        raise ValueError("package_dir must not be inside the source data tree")
    package_dir.mkdir(parents=True, exist_ok=True)
    out_path = _shard_path(_resolve_repo_or_absolute(args.out), args.shard, args.nshards)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[tuple[str, str]] = set()
    if args.resume and out_path.exists():
        for record in _jsonl(out_path):
            if record.get("record_type") == "package":
                completed.add((str(record["memory_id"]), str(record["condition_id"])))
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            meta = _metadata(args, "write")
            meta.update({"codecs": codecs, "budgets_bytes": budgets})
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for row_index, row in enumerate(rows, 1):
            source_path = _resolve_repo_or_absolute(row["image"])
            source_hash = _sha256(source_path)
            if row.get("image_sha256") and source_hash != row["image_sha256"]:
                raise RuntimeError(f"source hash mismatch: {row['memory_id']}")
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
                condition_id = package_condition(codec, budget)
                done_key = (str(row["memory_id"]), condition_id)
                if done_key in completed:
                    continue
                destination = memory_package_path(
                    package_dir,
                    str(row["memory_id"]),
                    codec,
                    budget,
                    source_path.suffix,
                )
                started = time.perf_counter()
                record: dict[str, Any] = {
                    "record_type": "package",
                    "schema_version": SCHEMA_VERSION,
                    "representation": REPRESENTATION,
                    "episode_id": str(row["episode_id"]),
                    "memory_id": str(row["memory_id"]),
                    "condition_id": condition_id,
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
                        raise RuntimeError("package exceeded its physical byte cap")
                else:
                    record.update({
                        "package": None,
                        "package_bytes": None,
                        "package_sha256": None,
                    })
                record["write_seconds"] = time.perf_counter() - started
                assert_source_free(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[memory write {row_index}/{len(rows)}] {row['memory_id']} "
                    f"{condition_id} feasible={record['feasible']}",
                    flush=True,
                )


def load_package_index(
    path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load a source-free package manifest indexed by memory and condition."""

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in _jsonl(path):
        if record.get("record_type") != "package":
            continue
        assert_source_free(record)
        key = (str(record["memory_id"]), str(record["condition_id"]))
        if key in result:
            raise ValueError(f"duplicate package for {key[0]}/{key[1]}")
        result[key] = record
    return result


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    arm: str
    diagnostic_component: str
    storage_memory_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    relevant_memory_id: str
    byte_contract: dict[str, Any]
    retrieval_bypassed: bool
    target_position: int | None
    source_trial: dict[str, Any]


def iter_trial_specs(
    row: dict[str, Any], arms: Sequence[str] = DEFAULT_ARMS
) -> list[TrialSpec]:
    """Normalize D3/D4 rows into the exact image lists passed to inference."""

    selected = set(arms)
    unknown = selected.difference(DEFAULT_ARMS)
    if unknown:
        raise ValueError(f"unsupported trial arms: {sorted(unknown)}")
    available = set(map(str, row["available_memory_ids"]))
    question = row["question"]
    current_id = str(question["current_evidence"]["memory_id"])
    specs: list[TrialSpec] = []

    if "d3_oracle" in selected:
        for trial in row["d3_trials"]["retrieval"]:
            memory_ids = tuple(map(str, trial["oracle_task_memory_ids"]))
            storage_memory_ids = tuple(map(str, trial["candidate_memory_ids"]))
            # The candidate set is a retrieval diagnostic, but the answer arm
            # intentionally uses its labelled oracle-selected relevant block.
            specs.append(TrialSpec(
                trial_id=str(trial["trial_id"]),
                arm="d3_oracle",
                diagnostic_component="retrieval_oracle_task",
                storage_memory_ids=storage_memory_ids,
                memory_ids=memory_ids,
                relevant_memory_id=current_id,
                byte_contract=copy.deepcopy(trial["byte_contract"]),
                # The candidate set remains part of byte accounting, but this
                # task-score arm injects the labelled relevant memory.  It is
                # therefore an oracle upper bound, not a retrieval result.
                retrieval_bypassed=True,
                target_position=(
                    storage_memory_ids.index(current_id)
                    if current_id in storage_memory_ids else None
                ),
                source_trial=copy.deepcopy(trial),
            ))

    if "d3_interference" in selected:
        for trial in row["d3_trials"]["interference"]:
            memory_ids = tuple(map(str, trial["preselected_memory_ids"]))
            position = memory_ids.index(current_id) if current_id in memory_ids else None
            if position != trial["target_position"]:
                raise ValueError(f"target_position mismatch in {trial['trial_id']}")
            specs.append(TrialSpec(
                trial_id=str(trial["trial_id"]),
                arm="d3_interference",
                diagnostic_component="inference_interference",
                storage_memory_ids=memory_ids,
                memory_ids=memory_ids,
                relevant_memory_id=current_id,
                byte_contract=copy.deepcopy(trial["byte_contract"]),
                retrieval_bypassed=bool(trial["retrieval_bypassed"]),
                target_position=position,
                source_trial=copy.deepcopy(trial),
            ))

    if "d4" in selected:
        for trial in row["d4_trials"]:
            memory_ids = tuple(map(str, trial["memory_ids"]))
            specs.append(TrialSpec(
                trial_id=str(trial["trial_id"]),
                arm="d4",
                diagnostic_component="temporal_state",
                storage_memory_ids=memory_ids,
                memory_ids=memory_ids,
                relevant_memory_id=current_id,
                byte_contract=copy.deepcopy(trial["byte_contract"]),
                retrieval_bypassed=True,
                target_position=(memory_ids.index(current_id) if current_id in memory_ids else None),
                source_trial=copy.deepcopy(trial),
            ))

    for spec in specs:
        if not set(spec.storage_memory_ids) <= available:
            raise ValueError(f"trial references unavailable memory: {spec.trial_id}")
    return specs


def build_messages(n_images: int, question: str) -> list[dict[str, Any]]:
    """Construct a Qwen2.5-VL prompt without injecting memory metadata."""

    if n_images < 0:
        raise ValueError("n_images cannot be negative")
    content: list[dict[str, str]] = []
    for index in range(n_images):
        content.append({"type": "text", "text": f"Memory image {index + 1}:\n"})
        content.append({"type": "image"})
    instruction = (
        "These are stored visual-memory snapshots. Read the project code, "
        "revision/timestamp, and requested field from the pixels. Use the newest "
        "applicable evidence when snapshots conflict. "
        if n_images
        else "No visual-memory snapshot was retrieved. "
    )
    content.append({
        "type": "text",
        "text": instruction + question + " Return only the requested value.",
    })
    return [{"role": "user", "content": content}]


@torch.inference_mode()
def generate_answer(
    model: Any,
    processor: Any,
    images: Sequence[Image.Image],
    question: str,
    device: str,
    max_new_tokens: int,
) -> str:
    """Generate from zero or more images using the standard Qwen interface."""

    messages = build_messages(len(images), question)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    processor_kwargs: dict[str, Any] = {
        "text": [prompt],
        "return_tensors": "pt",
    }
    if images:
        processor_kwargs["images"] = list(images)
    inputs = processor(**processor_kwargs).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    return processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def prediction_scores(
    prediction: str, question: dict[str, Any]
) -> dict[str, float]:
    acceptable = list(map(str, question["acceptable_answers"]))
    stale = list(map(str, question.get("stale_answers", [])))
    return {
        "current_em": exact_match(prediction, acceptable),
        "current_anls": anls(prediction, acceptable),
        "stale_capture": exact_match(prediction, stale) if stale else 0.0,
    }


def _condition_for_trial(codec: str, contract: dict[str, Any]) -> str:
    budget = contract.get("per_memory_budget_bytes")
    return package_condition(codec, None if codec == "copy" else int(budget))


def _load_trial_images(
    spec: TrialSpec,
    packages: dict[tuple[str, str], dict[str, Any]],
    codec: str,
) -> tuple[list[Image.Image], list[dict[str, Any]], str | None]:
    """Verify physical packages and load them without any source path."""

    if not spec.storage_memory_ids:
        return [], [], None
    condition_id = _condition_for_trial(codec, spec.byte_contract)
    loaded: list[Image.Image] = []
    records: list[dict[str, Any]] = []
    per_memory_cap = spec.byte_contract.get("per_memory_budget_bytes")
    total_cap = int(spec.byte_contract["total_payload_budget_bytes"])
    images_by_memory: dict[str, Image.Image] = {}
    for memory_id in spec.storage_memory_ids:
        record = packages.get((memory_id, condition_id))
        if record is None:
            return [], records, f"missing package {memory_id}/{condition_id}"
        if not record.get("feasible"):
            return [], records, f"infeasible package {memory_id}/{condition_id}"
        package_path = resolve_memory_package(str(record["package"]))
        actual_bytes = package_path.stat().st_size
        if actual_bytes != int(record["package_bytes"]):
            raise RuntimeError(f"package size changed: {package_path}")
        if _sha256(package_path) != record["package_sha256"]:
            raise RuntimeError(f"package hash changed: {package_path}")
        if per_memory_cap is not None and actual_bytes > int(per_memory_cap):
            return [], records, f"package exceeds per-memory cap: {memory_id}"
        payload = package_path.read_bytes()
        with Image.open(BytesIO(payload)) as opened:
            images_by_memory[memory_id] = opened.convert("RGB")
        records.append(record)
    if sum(int(record["package_bytes"]) for record in records) > total_cap:
        return [], records, "packages exceed total payload cap"
    loaded = [images_by_memory[memory_id] for memory_id in spec.memory_ids]
    return loaded, records, None


def _done_trial_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(record["trial_id"])
        for record in _jsonl(path)
        if record.get("record_type") == "trial_result"
    }


@torch.inference_mode()
def run_read(args: argparse.Namespace) -> None:
    """Evaluate source-free D3 oracle/interference and D4 trial arms."""

    manifest = _resolve_repo_or_absolute(args.manifest)
    rows = _sharded(_jsonl(manifest), args.shard, args.nshards, args.limit)
    for row in rows:
        assert_source_free(row)
    package_manifest = _resolve_repo_or_absolute(args.package_manifest)
    packages = load_package_index(package_manifest)
    codec = args.codec.strip().lower()
    if codec not in SUPPORTED_CODECS:
        raise ValueError(f"unsupported codec: {codec}")
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())

    out_path = _shard_path(_resolve_repo_or_absolute(args.out), args.shard, args.nshards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_trial_ids(out_path) if args.resume else set()
    model, processor = load_vlm("qwen25vl", device=args.device, max_pixels=MAX_PIXELS)
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type != "qwen2_5_vl":
        raise RuntimeError(f"D3/D4 reader requires Qwen2.5-VL, got {model_type!r}")

    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            meta = _metadata(args, "read")
            meta.update({
                "package_manifest": str(args.package_manifest),
                "package_manifest_sha256": _sha256(package_manifest),
                "codec": codec.upper(),
                "arms": list(arms),
                "model": "qwen25vl",
                "device": args.device,
                "package_manifest_bytes_excluded_from_payload": True,
            })
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")

        total_trials = sum(len(iter_trial_specs(row, arms)) for row in rows)
        completed_index = 0
        for row in rows:
            question = row["question"]
            for spec in iter_trial_specs(row, arms):
                completed_index += 1
                if spec.trial_id in done:
                    continue
                images, package_records, infeasible_reason = _load_trial_images(
                    spec, packages, codec
                )
                actual_bytes = sum(
                    int(record["package_bytes"]) for record in package_records
                )
                base_record: dict[str, Any] = {
                    "record_type": "trial_result",
                    "schema_version": SCHEMA_VERSION,
                    "representation": REPRESENTATION,
                    "episode_id": str(row["episode_id"]),
                    "trial_id": spec.trial_id,
                    "arm": spec.arm,
                    "diagnostic_component": spec.diagnostic_component,
                    "factorial_cell": row["factorial"]["cell_id"],
                    "time_gap": row["factorial"]["time_gap"],
                    "state_change": row["factorial"]["state_change"],
                    "memory_condition": spec.source_trial.get("memory_condition"),
                    "candidate_count": spec.source_trial.get("candidate_count"),
                    "distractor_count": spec.source_trial.get("distractor_count"),
                    "storage_memory_ids": list(spec.storage_memory_ids),
                    "memory_ids": list(spec.memory_ids),
                    "n_memory_packages": len(spec.storage_memory_ids),
                    "n_inference_images": len(spec.memory_ids),
                    "relevant_memory_id": spec.relevant_memory_id,
                    "target_position": spec.target_position,
                    "target_position_reference": (
                        "candidate_storage_order"
                        if spec.arm == "d3_oracle" else "inference_image_order"
                    ),
                    "inference_target_position": (
                        spec.memory_ids.index(spec.relevant_memory_id)
                        if spec.relevant_memory_id in spec.memory_ids else None
                    ),
                    "target_present": spec.target_position is not None,
                    "retrieval_bypassed": spec.retrieval_bypassed,
                    "source_trial_retrieval_bypassed": spec.source_trial.get(
                        "retrieval_bypassed"
                    ),
                    "retrieval_metrics_computed": False,
                    "oracle_selection_used": spec.arm == "d3_oracle",
                    "codec": codec.upper(),
                    "condition_id": (
                        None if not spec.storage_memory_ids
                        else _condition_for_trial(codec, spec.byte_contract)
                    ),
                    "total_package_bytes": actual_bytes,
                    "total_payload_budget_bytes": spec.byte_contract[
                        "total_payload_budget_bytes"
                    ],
                    "per_memory_budget_bytes": spec.byte_contract.get(
                        "per_memory_budget_bytes"
                    ),
                    "package_bytes_by_memory": {
                        str(record["memory_id"]): int(record["package_bytes"])
                        for record in package_records
                    },
                    "package_sha256_by_memory": {
                        str(record["memory_id"]): str(record["package_sha256"])
                        for record in package_records
                    },
                    "source_path_in_read_manifest": False,
                    "reader_pid": os.getpid(),
                }
                if infeasible_reason is not None:
                    base_record.update({
                        "feasible": False,
                        "infeasible_reason": infeasible_reason,
                        "prediction": None,
                        "current_em": None,
                        "current_anls": None,
                        "stale_capture": None,
                        "answer_seconds": None,
                    })
                else:
                    started = time.perf_counter()
                    prediction = generate_answer(
                        model,
                        processor,
                        images,
                        str(question["question"]),
                        args.device,
                        args.max_new_tokens,
                    )
                    base_record.update({
                        "feasible": True,
                        "infeasible_reason": None,
                        "question_id": str(question["question_id"]),
                        "prediction": prediction,
                        **prediction_scores(prediction, question),
                        "answer_seconds": time.perf_counter() - started,
                    })
                handle.write(json.dumps(base_record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[memory read {completed_index}/{total_trials}] {spec.trial_id} "
                    f"feasible={base_record['feasible']} "
                    f"em={base_record.get('current_em')}",
                    flush=True,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["prepare", "write", "read"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--write-manifest")
    parser.add_argument("--read-manifest")
    parser.add_argument(
        "--package-dir", default="results/smoke/memory_dynamics_packages"
    )
    parser.add_argument("--package-manifest")
    parser.add_argument("--out", default="results/smoke/memory_dynamics_eval.jsonl")
    parser.add_argument("--codecs", default="copy,jpeg,webp,avif")
    parser.add_argument(
        "--budgets-bytes", default=",".join(map(str, DEFAULT_BUDGETS))
    )
    parser.add_argument("--codec", default="jpeg", choices=sorted(SUPPORTED_CODECS))
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "prepare":
        if not args.write_manifest or not args.read_manifest:
            raise SystemExit("prepare requires --write-manifest and --read-manifest")
        counts = prepare_manifests(
            _resolve_repo_or_absolute(args.manifest),
            _resolve_repo_or_absolute(args.write_manifest),
            _resolve_repo_or_absolute(args.read_manifest),
            args.limit,
        )
        print(f"prepared {counts[0]} memory rows and {counts[1]} read episodes")
        return
    if args.mode == "write":
        run_write(args)
        return
    if not args.package_manifest:
        raise SystemExit("read requires --package-manifest")
    run_read(args)


if __name__ == "__main__":
    main()
