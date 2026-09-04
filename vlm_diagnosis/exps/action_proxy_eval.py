"""Strict source-denial evaluator for the controlled D5 action proxy.

The evaluator has three process boundaries:

``prepare``
    Split a :mod:`gen_action_proxy` episode into two question-free image rows
    for the writer and one source-free history/gold row for the reader.
``write``
    Encode each observation independently as Qwen2.5-VL full projected visual
    tokens.  Only this process can import PIL or open an original PNG.
``read``
    Evaluate ordered-history, current-only, old-only, and no-memory arms from
    stored tensor packages.  The reader never imports PIL, receives a source
    path, constructs ``pixel_values``, or opens any file under ``data/``.

This is a synthetic offline mechanism probe.  It does not execute actions in
an environment and cannot support real agent-trajectory, deployment,
generality, or novelty claims by itself.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from vlm_diagnosis.core.loader import assert_finite_logits, load_vlm
from vlm_diagnosis.core.tensor_quantization import QUANTIZATION_SCHEMES
from vlm_diagnosis.exps.source_denial_embedding import (
    MODEL_FAMILY,
    REPRESENTATION,
    _load_package,
    _require_qwen25vl,
    _sync_if_cuda,
    _write_one,
    assert_question_free,
    decode_projected_visual_tokens,
    projected_package_path,
    projected_quantization_scheme,
)
from vlm_diagnosis.exps.source_denial_kv import (
    MAX_PIXELS,
    ROOT,
    _jsonl,
    _sha256,
    _shard_path,
    _sharded,
    assert_source_free,
)
from vlm_diagnosis.scripts.gen_action_proxy import (
    CLAIM_SCOPE,
    PROBE_KIND,
    validate_manifest_row,
)


SCHEMA_VERSION = "action-proxy-eval-v2"
STAGE = "D5_CONTROLLED_OFFLINE_ACTION"
DEFAULT_ARMS = ("ordered_history", "current_only", "old_only", "no_memory")
PROMPT_CONTRACT_VERSION = "d5-action-json-v2"
OPERATIONAL_ARGUMENT_POLICY = {
    "version": "d5-operational-arguments-v1",
    # These defaults do not change which UI operation is requested.  They only
    # materialize values that are implicit in this fixture's action API.
    "harmless_defaults": {
        "click": {"button": "left"},
        "type": {"replace_existing": False},
        "scroll": {"amount": "one_viewport"},
    },
    # This is the sole accepted key alias.  It was fixed before rescoring and
    # preserves the exact, case-sensitive token payload.
    "key_aliases": {"type": {"token": "text"}},
    "unknown_keys_rejected": True,
}


def _resolve_repo_or_absolute(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _target_element(row: dict[str, Any]) -> dict[str, Any]:
    next_action = row["next_action"]
    current = next(
        observation for observation in row["observations"]
        if observation["observation_id"] == next_action["source_observation_id"]
    )
    return next(
        element for element in current["elements"]
        if element["element_id"] == next_action["target_element_id"]
    )


def _target_role_is_unique(row: dict[str, Any], target: dict[str, Any]) -> bool:
    next_action = row["next_action"]
    current = next(
        observation for observation in row["observations"]
        if observation["observation_id"] == next_action["source_observation_id"]
    )
    return sum(
        element["role"] == target["role"] and element["interactive"]
        for element in current["elements"]
    ) == 1


def split_episode_row(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return image-only writer rows and a source-free D5 reader row."""

    validate_manifest_row(row)
    common = {
        "dataset": row["dataset"],
        "dataset_revision": row["dataset_revision"],
        "split": row["split"],
        "episode_id": row["episode_id"],
    }
    write_rows: list[dict[str, Any]] = []
    observation_metadata: list[dict[str, Any]] = []
    for observation in row["observations"]:
        observation_id = str(observation["observation_id"])
        write_row = {
            **common,
            # ``_write_one`` uses sample_id as the package identity.
            "sample_id": observation_id,
            "image": observation["image"],
            "image_sha256": observation.get("image_sha256"),
        }
        assert_question_free(write_row)
        write_rows.append(write_row)
        observation_metadata.append({
            "observation_id": observation_id,
            "sequence_index": int(observation["sequence_index"]),
            "role": str(observation["role"]),
            "revision_id": str(observation["revision_id"]),
            "observed_at": str(observation["observed_at"]),
        })

    target = _target_element(row)
    next_action = row["next_action"]
    read_row = {
        **common,
        "synthetic": True,
        "probe_kind": row["probe_kind"],
        "claim_scope": row["claim_scope"],
        "real_trajectory_claim_allowed": False,
        "task_goal": row["task_goal"],
        "episode_order": copy.deepcopy(row["episode_order"]),
        "observation_metadata": observation_metadata,
        "available_observation_ids": [
            item["observation_id"] for item in observation_metadata
        ],
        "event_history": copy.deepcopy(row["event_history"]),
        "action_history": copy.deepcopy(row["action_history"]),
        "state_transition": copy.deepcopy(row["state_transition"]),
        "invalidated_actions": copy.deepcopy(row["invalidated_actions"]),
        "gold_action": {
            "action_id": next_action["action_id"],
            "action_type": next_action["action_type"],
            "action_label": next_action["action_label"],
            "target_element_id": next_action["target_element_id"],
            "target_label": target.get("label"),
            "target_role": target["role"],
            "target_role_unique_among_interactive_elements": _target_role_is_unique(
                row, target
            ),
            "target_bbox": copy.deepcopy(next_action["target_bbox"]),
            "arguments": copy.deepcopy(next_action["arguments"]),
            "valid_in_revision_id": next_action["valid_in_revision_id"],
        },
        "scoring_contract": copy.deepcopy(row["scoring_contract"]),
    }
    assert_source_free(read_row)
    return write_rows, read_row


def prepare_manifests(
    source: Path,
    write_path: Path,
    read_path: Path,
    limit: int | None = None,
) -> tuple[int, int]:
    """Split full episodes while preserving both source-denial boundaries."""

    rows = _jsonl(source)
    if limit is not None:
        rows = rows[:limit]
    write_rows: list[dict[str, Any]] = []
    read_rows: list[dict[str, Any]] = []
    for row in rows:
        episode_writes, episode_read = split_episode_row(row)
        write_rows.extend(episode_writes)
        read_rows.append(episode_read)
    _write_jsonl(write_path, write_rows)
    _write_jsonl(read_path, read_rows)
    return len(write_rows), len(read_rows)


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    arm: str
    observation_ids: tuple[str, ...]
    include_event_history: bool


def iter_trial_specs(
    row: dict[str, Any], arms: Sequence[str] = DEFAULT_ARMS
) -> list[TrialSpec]:
    """Create comparable D5 arms with an identical current-action gold label."""

    selected = list(dict.fromkeys(map(str, arms)))
    unknown = set(selected).difference(DEFAULT_ARMS)
    if unknown:
        raise ValueError(f"unsupported D5 arms: {sorted(unknown)}")
    by_role = {
        str(item["role"]): str(item["observation_id"])
        for item in row["observation_metadata"]
    }
    if set(by_role) != {"old_state", "current_state"}:
        raise ValueError("reader row must identify old_state and current_state")
    old_id = by_role["old_state"]
    current_id = by_role["current_state"]
    mapping = {
        "ordered_history": ((old_id, current_id), True),
        "current_only": ((current_id,), False),
        "old_only": ((old_id,), False),
        "no_memory": ((), False),
    }
    available = set(map(str, row["available_observation_ids"]))
    specs = [
        TrialSpec(
            trial_id=f"{row['episode_id']}:d5:{arm}",
            arm=arm,
            observation_ids=mapping[arm][0],
            include_event_history=mapping[arm][1],
        )
        for arm in selected
    ]
    for spec in specs:
        if not set(spec.observation_ids) <= available:
            raise ValueError(f"trial references unavailable observation: {spec.trial_id}")
    return specs


def _observation_metadata(row: dict[str, Any], observation_id: str) -> dict[str, Any]:
    return next(
        item for item in row["observation_metadata"]
        if item["observation_id"] == observation_id
    )


def build_messages(row: dict[str, Any], spec: TrialSpec) -> list[dict[str, Any]]:
    """Build the model-visible prompt without injecting any gold action field."""

    content: list[dict[str, str]] = []
    if spec.arm == "ordered_history":
        old_id, current_id = spec.observation_ids
        old = _observation_metadata(row, old_id)
        current = _observation_metadata(row, current_id)
        previous = row["action_history"][0]
        content.extend([
            {
                "type": "text",
                "text": (
                    f"Event 0 — old observation ({old['revision_id']}, "
                    f"{old['observed_at']}):\n"
                ),
            },
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "\nEvent 1 — completed action: "
                    f"{str(previous['action_type']).upper()} the OPEN control. "
                    f"Outcome: {previous['outcome']}. This completed action is "
                    "invalid in the next revision and must not be replayed.\n"
                ),
            },
            {
                "type": "text",
                "text": (
                    f"Event 2 — current observation ({current['revision_id']}, "
                    f"{current['observed_at']}):\n"
                ),
            },
            {"type": "image"},
        ])
    elif spec.arm in {"current_only", "old_only"}:
        item = _observation_metadata(row, spec.observation_ids[0])
        label = "current" if spec.arm == "current_only" else "old"
        content.extend([
            {
                "type": "text",
                "text": f"Available {label} observation ({item['revision_id']}):\n",
            },
            {"type": "image"},
        ])
    elif spec.arm == "no_memory":
        content.append({
            "type": "text",
            "text": "No stored visual observation or action history is available.\n",
        })
    else:  # pragma: no cover - guarded by iter_trial_specs
        raise ValueError(f"unsupported arm: {spec.arm}")

    content.append({
        "type": "text",
        "text": (
            "\nTask goal: " + str(row["task_goal"]) + "\n"
            "Predict the single next valid UI action. Use screenshot pixels and "
            "chronological history only; do not repeat a completed action. Coordinates "
            "must use the original 768x896 screenshot. target_bbox must always contain "
            "exactly four numbers [x1,y1,x2,y2], not a center point. Return one JSON "
            "object only with "
            'this schema: {"action_type":"click|type|scroll",'
            '"target_label":"visible label or UI role",'
            '"target_bbox":[x1,y1,x2,y2],"arguments":{...}}. '
            "Use exactly one of these action-specific arguments objects, with every "
            "shown key present: "
            'click -> {"button":"left"}; '
            'type -> {"text":"<exact saved token>","replace_existing":false}; '
            'scroll -> {"direction":"up|down","amount":"one_viewport"}. '
            "Do not rename text to token. For type actions preserve token case exactly."
        ),
    })
    return [{"role": "user", "content": content}]


def expand_multi_image_placeholders(
    raw_ids: torch.Tensor,
    image_token_id: int,
    visual_counts: Sequence[int],
) -> torch.Tensor:
    """Expand each raw image placeholder to its stored projected-token count."""

    if raw_ids.ndim != 2 or raw_ids.shape[0] != 1:
        raise ValueError("raw_ids must have shape (1, sequence_length)")
    counts = [int(value) for value in visual_counts]
    if any(value < 1 for value in counts):
        raise ValueError("every visual token count must be positive")
    placeholder_count = int((raw_ids[0] == image_token_id).sum())
    if placeholder_count != len(counts):
        raise ValueError(
            "raw prompt/package image count mismatch: "
            f"{placeholder_count} != {len(counts)}"
        )
    pieces: list[torch.Tensor] = []
    count_index = 0
    for token in raw_ids[0]:
        if int(token) == image_token_id:
            pieces.append(token.repeat(counts[count_index]))
            count_index += 1
        else:
            pieces.append(token.view(1))
    if not pieces:
        return raw_ids.clone()
    return torch.cat(pieces).view(1, -1)


def inject_multi_projected_visual_tokens(
    full_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    projected_tokens: Sequence[torch.Tensor],
    image_token_id: int,
) -> torch.Tensor:
    """Inject ordered observation packages without pixels or a vision encoder."""

    visual_positions = (full_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if not projected_tokens:
        if visual_positions.numel() != 0:
            raise ValueError("prompt has image placeholders but no projected packages")
        return token_embeddings.clone()
    combined = torch.cat(list(projected_tokens), dim=0)
    if visual_positions.numel() != combined.shape[0]:
        raise ValueError(
            "visual placeholder/embedding count mismatch: "
            f"{visual_positions.numel()} != {combined.shape[0]}"
        )
    if token_embeddings.shape[-1] != combined.shape[-1]:
        raise ValueError("language/visual hidden size mismatch")
    result = token_embeddings.clone()
    result[0, visual_positions] = combined.to(
        device=result.device, dtype=result.dtype
    )
    return result


def _raw_prompt_ids(
    processor: Any,
    messages: list[dict[str, Any]],
    device: str,
) -> torch.Tensor:
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)


@torch.no_grad()
def generate_prediction(
    model: Any,
    processor: Any,
    blobs: Sequence[dict[str, Any]],
    row: dict[str, Any],
    spec: TrialSpec,
    device: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, float]]:
    """Decode a D5 action using only projected tensors and source-free text."""

    _require_qwen25vl(model)
    for blob in blobs:
        if blob["model_id"] != model.config._name_or_path:
            raise RuntimeError(
                f"package checkpoint {blob['model_id']} != reader "
                f"{model.config._name_or_path}"
            )
    reconstruction_started = time.perf_counter()
    messages = build_messages(row, spec)
    raw_ids = _raw_prompt_ids(processor, messages, device)
    decoded = [decode_projected_visual_tokens(blob) for blob in blobs]
    counts = [int(tensor.shape[0]) for tensor in decoded]
    full_ids = expand_multi_image_placeholders(
        raw_ids, model.config.image_token_id, counts
    )
    grid = (
        torch.cat([blob["image_grid_thw"] for blob in blobs], dim=0).to(device)
        if blobs else None
    )
    core = model.model if hasattr(model.model, "get_rope_index") else model
    positions, _ = core.get_rope_index(
        full_ids,
        image_grid_thw=grid,
        attention_mask=torch.ones_like(full_ids),
    )
    token_embeddings = model.model.get_input_embeddings()(full_ids)
    inputs_embeds = inject_multi_projected_visual_tokens(
        full_ids,
        token_embeddings,
        decoded,
        model.config.image_token_id,
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_started

    _sync_if_cuda(device)
    prefill_started = time.perf_counter()
    output = model(
        input_ids=full_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones_like(full_ids),
        position_ids=positions,
        use_cache=True,
    )
    _sync_if_cuda(device)
    prefill_seconds = time.perf_counter() - prefill_started
    assert_finite_logits(output.logits, f"action_proxy:{row['episode_id']}:{spec.arm}")

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
            output.logits, f"action_proxy_decode:{row['episode_id']}:{spec.arm}"
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


def parse_action_prediction(prediction: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response, failing closed."""

    if not isinstance(prediction, str):
        return None
    start = prediction.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(prediction[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _normalized_text(value: Any) -> str | None:
    return value.strip().casefold() if isinstance(value, str) else None


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        return None
    result = [float(item) for item in value]
    if not (result[0] < result[2] and result[1] < result[3]):
        return None
    return result


def _point(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        return None
    return [float(value[0]), float(value[1])]


def bbox_iou(first: Any, second: Any) -> float | None:
    """Return pixel-space IoU, or ``None`` for an invalid prediction box."""

    a = _bbox(first)
    b = _bbox(second)
    if a is None or b is None:
        return None
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def point_in_bbox(point: Any, bbox: Any) -> bool:
    """Use an inclusive boundary for pixel-space point-hit diagnostics."""

    parsed_point = _point(point)
    parsed_bbox = _bbox(bbox)
    if parsed_point is None or parsed_bbox is None:
        return False
    return (
        parsed_bbox[0] <= parsed_point[0] <= parsed_bbox[2]
        and parsed_bbox[1] <= parsed_point[1] <= parsed_bbox[3]
    )


def spatial_target_diagnostics(
    parsed: dict[str, Any] | None,
    gold_bbox: Any,
) -> dict[str, Any]:
    """Report box IoU, predicted-box center hit, and explicit/legacy point hit.

    A two-number ``target_bbox`` is not silently treated as a valid box.  It is
    retained only as a separately named legacy point diagnostic so the strict
    IoU metric remains unchanged.
    """

    raw_bbox = parsed.get("target_bbox") if parsed else None
    predicted_bbox = _bbox(raw_bbox)
    predicted_center = (
        [
            (predicted_bbox[0] + predicted_bbox[2]) / 2,
            (predicted_bbox[1] + predicted_bbox[3]) / 2,
        ]
        if predicted_bbox is not None else None
    )
    center_hit = float(point_in_bbox(predicted_center, gold_bbox))

    explicit_point = _point(parsed.get("target_point")) if parsed else None
    point_source: str | None = "target_point" if explicit_point is not None else None
    if explicit_point is None and parsed:
        explicit_point = _point(parsed.get("target_center"))
        if explicit_point is not None:
            point_source = "target_center"
    if explicit_point is None:
        explicit_point = _point(raw_bbox)
        if explicit_point is not None:
            point_source = "target_bbox_legacy_point"
    point_hit = float(point_in_bbox(explicit_point, gold_bbox))
    target_iou = bbox_iou(raw_bbox, gold_bbox)
    iou_hit = float(target_iou is not None and target_iou >= 0.5)
    return {
        "target_bbox_iou": target_iou,
        "target_bbox_iou_at_0_5": iou_hit,
        "predicted_bbox_center": predicted_center,
        "target_bbox_center_inside_gold": center_hit,
        "predicted_target_point": explicit_point,
        "predicted_target_point_source": point_source,
        "target_point_hit": point_hit,
        "spatial_target_success": float(bool(iou_hit or center_hit or point_hit)),
    }


def _canonical_argument(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical_argument(item) for key, item in sorted(value.items())}
    if isinstance(value, str):
        return value.strip()
    return value


def normalize_operational_arguments(
    action_type: Any,
    arguments: Any,
) -> dict[str, Any]:
    """Apply only the predeclared alias/default policy for a secondary metric."""

    normalized_type = _normalized_text(action_type)
    result = copy.deepcopy(arguments) if isinstance(arguments, dict) else None
    trace: dict[str, Any] = {
        "policy_version": OPERATIONAL_ARGUMENT_POLICY["version"],
        "valid": False,
        "normalized_arguments": result,
        "applied_defaults": [],
        "applied_aliases": [],
        "rejection_reason": None,
    }
    allowed_keys = {
        "click": {"button"},
        "type": {"text", "replace_existing"},
        "scroll": {"direction", "amount"},
    }
    if normalized_type not in allowed_keys:
        trace["rejection_reason"] = "unsupported_action_type"
        return trace
    if result is None:
        trace["rejection_reason"] = "arguments_not_object"
        return trace

    aliases = OPERATIONAL_ARGUMENT_POLICY["key_aliases"].get(normalized_type, {})
    for alias, canonical in aliases.items():
        if alias not in result:
            continue
        if canonical in result and result[canonical] != result[alias]:
            trace["rejection_reason"] = f"conflicting_alias:{alias}->{canonical}"
            trace["normalized_arguments"] = result
            return trace
        result[canonical] = result.pop(alias)
        trace["applied_aliases"].append(f"{alias}->{canonical}")

    unknown = sorted(set(map(str, result)).difference(allowed_keys[normalized_type]))
    if unknown:
        trace["rejection_reason"] = "unknown_keys:" + ",".join(unknown)
        trace["normalized_arguments"] = result
        return trace
    defaults = OPERATIONAL_ARGUMENT_POLICY["harmless_defaults"][normalized_type]
    for key, value in defaults.items():
        if key not in result:
            result[key] = copy.deepcopy(value)
            trace["applied_defaults"].append(key)

    missing = sorted(allowed_keys[normalized_type].difference(result))
    if missing:
        trace["rejection_reason"] = "missing_required_keys:" + ",".join(missing)
        trace["normalized_arguments"] = result
        return trace
    trace["valid"] = True
    trace["normalized_arguments"] = result
    return trace


def score_action_prediction(
    prediction: str,
    gold: dict[str, Any],
    invalidated_actions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Score strict action semantics and target-grounding components."""

    parsed = parse_action_prediction(prediction)
    predicted_type = _normalized_text(parsed.get("action_type")) if parsed else None
    gold_type = _normalized_text(gold["action_type"])
    type_em = float(predicted_type == gold_type)
    predicted_arguments = parsed.get("arguments") if parsed else None
    gold_arguments = gold["arguments"]
    arguments_em = float(
        _canonical_argument(predicted_arguments) == _canonical_argument(gold_arguments)
    )
    argument_component_em = {
        str(key): float(
            isinstance(predicted_arguments, dict)
            and _canonical_argument(predicted_arguments.get(key))
            == _canonical_argument(value)
        )
        for key, value in gold_arguments.items()
    }
    operational = normalize_operational_arguments(gold_type, predicted_arguments)
    normalized_arguments = operational["normalized_arguments"]
    operational_arguments_em = float(
        operational["valid"]
        and _canonical_argument(normalized_arguments)
        == _canonical_argument(gold_arguments)
    )
    operational_argument_component_em = {
        str(key): float(
            operational["valid"]
            and isinstance(normalized_arguments, dict)
            and _canonical_argument(normalized_arguments.get(key))
            == _canonical_argument(value)
        )
        for key, value in gold_arguments.items()
    }

    predicted_label = parsed.get("target_label") if parsed else None
    target_label_em = float(
        _normalized_text(predicted_label) == _normalized_text(gold.get("target_label"))
    )
    target_role_em = float(
        bool(gold.get("target_role_unique_among_interactive_elements"))
        and _normalized_text(predicted_label) == _normalized_text(gold.get("target_role"))
    )
    predicted_id = parsed.get("target_element_id") if parsed else None
    target_element_id_em = float(predicted_id == gold["target_element_id"])
    spatial = spatial_target_diagnostics(parsed, gold["target_bbox"])
    target_iou = spatial["target_bbox_iou"]
    target_bbox_success = spatial["target_bbox_iou_at_0_5"]
    target_success = float(
        bool(
            target_label_em
            or target_role_em
            or target_element_id_em
            or target_bbox_success
        )
    )
    type_arguments_em = float(bool(type_em and arguments_em))
    full_action_em = float(bool(type_arguments_em and target_success))
    operational_type_arguments_em = float(bool(type_em and operational_arguments_em))
    operational_full_action_em = float(
        bool(operational_type_arguments_em and target_success)
    )

    invalidated_ids = {str(action["target_element_id"]) for action in invalidated_actions}
    stale_action_replay = float(
        bool(
            predicted_type == "click"
            and (
                (isinstance(predicted_id, str) and predicted_id in invalidated_ids)
                or _normalized_text(predicted_label) == "open"
            )
        )
    )
    components = {
        "action_type_em": type_em,
        "arguments_em": arguments_em,
        "argument_component_em": argument_component_em,
        "action_type_arguments_em": type_arguments_em,
        "operational_argument_policy_version": OPERATIONAL_ARGUMENT_POLICY["version"],
        "operational_argument_normalization": operational,
        "operational_arguments_em": operational_arguments_em,
        "operational_argument_component_em": operational_argument_component_em,
        "operational_action_type_arguments_em": operational_type_arguments_em,
        "target_label_em": target_label_em,
        "target_role_em": target_role_em,
        "target_element_id_em": target_element_id_em,
        "target_bbox_iou": target_iou,
        "target_bbox_iou_at_0_5": target_bbox_success,
        "predicted_bbox_center": spatial["predicted_bbox_center"],
        "target_bbox_center_inside_gold": spatial[
            "target_bbox_center_inside_gold"
        ],
        "predicted_target_point": spatial["predicted_target_point"],
        "predicted_target_point_source": spatial[
            "predicted_target_point_source"
        ],
        "target_point_hit": spatial["target_point_hit"],
        "spatial_target_success": spatial["spatial_target_success"],
        "target_success": target_success,
        "full_action_em": full_action_em,
        "operational_full_action_em": operational_full_action_em,
        "stale_action_replay": stale_action_replay,
    }
    return {
        "prediction_json_valid": parsed is not None,
        "parsed_prediction": parsed,
        **components,
        "component_metrics": components,
    }


def load_package_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Index source-free observation packages by ID and quantization."""

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in _jsonl(path):
        if record.get("record_type") != "package":
            continue
        assert_source_free(record)
        observation_id = str(record["observation_id"])
        quantization = str(record["quantization"])
        key = (observation_id, quantization)
        if key in result:
            raise ValueError(
                f"duplicate action package for {observation_id}/{quantization}"
            )
        if record.get("representation") != REPRESENTATION:
            raise ValueError(f"wrong package representation for {observation_id}")
        result[key] = record
    return result


def resolve_projected_package(path_value: str) -> Path:
    """Resolve a tensor package while rejecting source data and path escape."""

    package_path = (ROOT / path_value).resolve()
    repo_root = ROOT.resolve()
    data_root = (ROOT / "data").resolve()
    if package_path == data_root or package_path.is_relative_to(data_root):
        raise ValueError(f"D5 reader refuses source-data path: {path_value}")
    if not package_path.is_relative_to(repo_root):
        raise ValueError(f"D5 package must stay inside repository: {path_value}")
    return package_path


def _load_trial_packages(
    spec: TrialSpec,
    packages: dict[tuple[str, str], dict[str, Any]],
    quantization: str,
    model_family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load verified tensor packages without importing an image library."""

    blobs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for observation_id in spec.observation_ids:
        record = packages.get((observation_id, quantization))
        if record is None:
            raise KeyError(f"missing package {observation_id}/{quantization}")
        path = resolve_projected_package(str(record["package"]))
        actual_bytes = path.stat().st_size
        if actual_bytes != int(record["package_bytes"]):
            raise RuntimeError(f"package size changed: {path}")
        if _sha256(path) != record["package_sha256"]:
            raise RuntimeError(f"package hash changed: {path}")
        blob = _load_package(path, model_family, quantization)
        if str(blob["sample_id"]) != observation_id:
            raise RuntimeError("package observation identity mismatch")
        blobs.append(blob)
        records.append(record)
    return blobs, records


def _metadata(args: argparse.Namespace, mode: str, manifest: Path) -> dict[str, Any]:
    return {
        "record_type": "run_metadata",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "mode": mode,
        "representation": REPRESENTATION,
        "quantization": args.quantization,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "strict_metrics_preserved": True,
        "operational_argument_policy": OPERATIONAL_ARGUMENT_POLICY,
        "manifest_sha256": _sha256(manifest),
        # Do not put the image-bearing writer-manifest path into the package
        # index that will later be handed to a source-denied reader.
        "manifest_path_stored": mode == "read",
        "manifest": str(args.manifest) if mode == "read" else None,
        "source_path_available": mode == "write",
        "pil_available": mode == "write",
        "pixel_values_available": mode == "write",
        "future_action_visible": False if mode == "write" else None,
        "synthetic": True,
        "probe_kind": PROBE_KIND,
        "claim_scope": CLAIM_SCOPE,
        "real_trajectory_claim_allowed": False,
        "interactive_environment_execution": False,
        "process_id": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def run_write(args: argparse.Namespace) -> None:
    """Create one question-free projected-token package per observation."""

    manifest = _resolve_repo_or_absolute(args.manifest)
    rows = _sharded(_jsonl(manifest), args.shard, args.nshards, args.limit)
    for row in rows:
        assert_question_free(row)
        if "image" not in row or "sample_id" not in row:
            raise ValueError("writer rows require image and sample_id")
    package_dir = _resolve_repo_or_absolute(args.package_dir)
    data_root = (ROOT / "data").resolve()
    if not package_dir.is_relative_to(ROOT.resolve()):
        raise ValueError("package_dir must stay inside repository")
    if package_dir == data_root or package_dir.is_relative_to(data_root):
        raise ValueError("package_dir must not be inside source data")
    package_dir.mkdir(parents=True, exist_ok=True)
    out_path = _shard_path(_resolve_repo_or_absolute(args.out), args.shard, args.nshards)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_vlm(args.model, device=args.device, max_pixels=MAX_PIXELS)
    _require_qwen25vl(model)
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            handle.write(json.dumps(_metadata(args, "write", manifest)) + "\n")
        for index, row in enumerate(rows, 1):
            destination = projected_package_path(
                package_dir, row["sample_id"], args.quantization
            )
            if args.resume and destination.exists():
                print(f"[action write skip] {row['sample_id']}", flush=True)
                continue
            encoded = _write_one(
                model,
                processor,
                row,
                args.device,
                destination,
                args.model,
                args.quantization,
            )
            record = {
                "record_type": "package",
                "schema_version": SCHEMA_VERSION,
                "representation": REPRESENTATION,
                "episode_id": str(row["episode_id"]),
                "observation_id": str(row["sample_id"]),
                **encoded,
            }
            assert_source_free(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[action write {index}/{len(rows)}] {row['sample_id']} "
                f"{record['package_bytes'] / 2**20:.1f} MiB",
                flush=True,
            )


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
    """Evaluate D5 arms without source paths, PIL, or pixel values."""

    manifest = _resolve_repo_or_absolute(args.manifest)
    rows = _sharded(_jsonl(manifest), args.shard, args.nshards, args.limit)
    for row in rows:
        assert_source_free(row)
        if row.get("real_trajectory_claim_allowed") is not False:
            raise ValueError("D5 controlled reader row must forbid trajectory claims")
    package_manifest = _resolve_repo_or_absolute(args.package_manifest)
    packages = load_package_index(package_manifest)
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())

    out_path = _shard_path(_resolve_repo_or_absolute(args.out), args.shard, args.nshards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_trial_ids(out_path) if args.resume else set()
    model, processor = load_vlm(args.model, device=args.device, max_pixels=MAX_PIXELS)
    _require_qwen25vl(model)

    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        if mode == "w" or out_path.stat().st_size == 0:
            meta = _metadata(args, "read", manifest)
            meta.update({
                "package_manifest": str(args.package_manifest),
                "package_manifest_sha256": _sha256(package_manifest),
                "arms": list(arms),
                "model": args.model,
                "device": args.device,
                "source_path_available": False,
                "pil_available": False,
                "pixel_values_available": False,
            })
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")

        total_trials = sum(len(iter_trial_specs(row, arms)) for row in rows)
        trial_index = 0
        for row in rows:
            for spec in iter_trial_specs(row, arms):
                trial_index += 1
                if spec.trial_id in done:
                    continue
                load_started = time.perf_counter()
                blobs, package_records = _load_trial_packages(
                    spec, packages, args.quantization, args.model
                )
                load_seconds = time.perf_counter() - load_started
                prediction, timings = generate_prediction(
                    model,
                    processor,
                    blobs,
                    row,
                    spec,
                    args.device,
                    args.max_new_tokens,
                )
                metrics = score_action_prediction(
                    prediction, row["gold_action"], row["invalidated_actions"]
                )
                record = {
                    "record_type": "trial_result",
                    "schema_version": SCHEMA_VERSION,
                    "stage": STAGE,
                    "representation": REPRESENTATION,
                    "quantization": args.quantization,
                    "episode_id": str(row["episode_id"]),
                    "trial_id": spec.trial_id,
                    "arm": spec.arm,
                    "observation_ids": list(spec.observation_ids),
                    "n_memory_packages": len(package_records),
                    "event_history_in_prompt": spec.include_event_history,
                    "total_package_bytes": sum(
                        int(item["package_bytes"]) for item in package_records
                    ),
                    "package_bytes_by_observation": {
                        str(item["observation_id"]): int(item["package_bytes"])
                        for item in package_records
                    },
                    "package_sha256_by_observation": {
                        str(item["observation_id"]): str(item["package_sha256"])
                        for item in package_records
                    },
                    "package_load_seconds": load_seconds,
                    **timings,
                    "prediction": prediction,
                    "gold_action": row["gold_action"],
                    **metrics,
                    "source_path_in_read_manifest": False,
                    "pil_used": False,
                    "pixel_values_used": False,
                    "offline_only": True,
                    "environment_action_executed": False,
                    "synthetic": True,
                    "probe_kind": row["probe_kind"],
                    "claim_scope": row["claim_scope"],
                    "real_trajectory_claim_allowed": False,
                    "reader_pid": os.getpid(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[action read {trial_index}/{total_trials}] {spec.trial_id} "
                    f"type+args={record['action_type_arguments_em']:.0f} "
                    f"full={record['full_action_em']:.0f}",
                    flush=True,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["prepare", "write", "read"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--write-manifest")
    parser.add_argument("--read-manifest")
    parser.add_argument(
        "--package-dir", default="results/smoke/action_proxy_projected_packages"
    )
    parser.add_argument("--package-manifest")
    parser.add_argument(
        "--out",
        help=(
            "output JSONL; defaults to a package manifest in write mode and "
            "an evaluation result in read mode"
        ),
    )
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--model", default=MODEL_FAMILY, choices=[MODEL_FAMILY])
    parser.add_argument(
        "--quantization", default="fp16", choices=QUANTIZATION_SCHEMES
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=160)
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
        print(f"prepared {counts[0]} observation rows and {counts[1]} D5 episodes")
        return
    if args.mode == "write":
        if args.out is None:
            args.out = "results/smoke/action_proxy_projected_packages.jsonl"
        run_write(args)
        return
    if not args.package_manifest:
        raise SystemExit("read requires --package-manifest")
    if args.out is None:
        args.out = "results/smoke/action_proxy_eval.jsonl"
    run_read(args)


if __name__ == "__main__":
    main()
