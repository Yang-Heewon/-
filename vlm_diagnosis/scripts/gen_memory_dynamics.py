"""Generate a deterministic controlled D3/D4 visual-memory fixture.

The fixture contains two revisions of the same synthetic UI entity, three
visually similar entity distractors, and read-time questions whose answer is
always defined by the current revision.  It is designed to separate:

* retrieval misses from inference-time distractor interference (D3), and
* near/far reads from unchanged/changed state (D4).

This is a controlled mechanism probe, not a real trajectory benchmark.  The
generator creates only images and labels; it never loads or runs a model.

Examples::

    python -m vlm_diagnosis.scripts.gen_memory_dynamics
    python -m vlm_diagnosis.scripts.gen_memory_dynamics --n-episodes 4 \
        --image-dir /tmp/memory-dynamics/images \
        --manifest /tmp/memory-dynamics/manifest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = ROOT / "data" / "memory_dynamics_controlled"
DEFAULT_MANIFEST = (
    ROOT / "experiments" / "manifests" / "memory_dynamics_controlled.jsonl"
)
DEFAULT_N_EPISODES = 32
DEFAULT_SEED = 42

IMAGE_WIDTH = 768
IMAGE_HEIGHT = 896
GENERATOR_VERSION = "memory-dynamics-generator-v1"
SCHEMA_VERSION = "memory-dynamics-schema-v1"
DATASET_NAME = "synthetic_memory_dynamics_controlled"
DOMAIN = "synthetic_project_dashboard"
PROBE_KIND = "controlled_d3_d4_mechanism_probe"
CLAIM_SCOPE = (
    "synthetic retrieval/interference/staleness mechanism probe only; "
    "no real-world trajectory, deployment, or novelty claims"
)

FACTOR_CELLS = (
    ("near", "unchanged"),
    ("near", "changed"),
    ("far", "unchanged"),
    ("far", "changed"),
)
TIME_GAP_SECONDS = {
    "near": 5 * 60,
    "far": 30 * 24 * 60 * 60,
}
RETRIEVAL_CANDIDATE_COUNTS = (1, 2, 4)
INTERFERENCE_DISTRACTOR_COUNTS = (0, 1, 3)
TEMPORAL_MEMORY_CONDITIONS = ("old_only", "current_only", "old+current", "no_memory")
TOTAL_PAYLOAD_BUDGET_BYTES = 256 * 1024

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

FIELD_VALUES = {
    "status": ("PLANNED", "IN REVIEW", "READY", "BLOCKED", "PAUSED", "SHIPPED"),
    "priority": ("LOW", "MEDIUM", "HIGH", "CRITICAL", "BACKLOG", "EXPEDITE"),
    "owner": ("ALICE", "BRUNO", "CARMEN", "DINESH", "ELENA", "FARAH"),
    "region": ("SEOUL", "TOKYO", "FRANKFURT", "TORONTO", "SYDNEY", "DUBLIN"),
}
FIELD_LABELS = {
    "status": "STATUS",
    "priority": "PRIORITY",
    "owner": "OWNER",
    "region": "REGION",
}
ACCENTS = (
    (51, 104, 214),
    (35, 145, 151),
    (133, 77, 190),
    (42, 153, 90),
)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _rng_for(seed: int, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:memory-dynamics:{index}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _memory_id(episode_id: str, role: str, revision: str) -> str:
    return f"memory:{episode_id}:{role}:{revision}"


def _revision_id(entity_id: str, revision: str) -> str:
    return f"{entity_id}@{revision}"


def _project_code(rng: random.Random, used: set[str]) -> str:
    while True:
        value = f"PRJ-{rng.randint(1000, 9999)}"
        if value not in used:
            used.add(value)
            return value


def _base_facts(rng: random.Random) -> dict[str, str]:
    return {field: rng.choice(values) for field, values in FIELD_VALUES.items()}


def _draw_screen(
    *,
    project_code: str,
    facts: dict[str, str],
    revision: str,
    observed_at: str,
    accent: tuple[int, int, int],
) -> tuple[Image.Image, dict[str, list[int]]]:
    """Draw one project-dashboard revision and return fact evidence boxes."""

    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (235, 240, 247))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [24, 18, IMAGE_WIDTH - 24, IMAGE_HEIGHT - 18],
        radius=36,
        fill=(255, 255, 255),
        outline=(204, 212, 224),
        width=3,
    )
    draw.rounded_rectangle([48, 44, 720, 166], radius=26, fill=(245, 248, 252))
    draw.text((76, 78), "MEMORYBOARD", font=_font(26, bold=True), fill=(37, 45, 60))
    draw.rounded_rectangle([504, 66, 690, 128], radius=16, fill=accent)
    draw.text((597, 97), project_code, font=_font(23, bold=True), fill="white", anchor="mm")

    captured = _parse_iso(observed_at).strftime("%Y-%m-%d %H:%M UTC")
    draw.text((76, 198), f"SNAPSHOT {revision.upper()}", font=_font(19, bold=True),
              fill=(75, 84, 101))
    draw.text((692, 198), captured, font=_font(16), fill=(103, 112, 129), anchor="ra")

    evidence: dict[str, list[int]] = {}
    row_top = 244
    row_height = 126
    for row_index, field in enumerate(("status", "priority", "owner", "region")):
        y1 = row_top + row_index * row_height
        y2 = y1 + 96
        draw.rounded_rectangle(
            [72, y1, 696, y2], radius=22, fill=(248, 250, 253),
            outline=(224, 229, 237), width=2,
        )
        draw.text((104, y1 + 29), FIELD_LABELS[field], font=_font(17, bold=True),
                  fill=(106, 115, 132))
        value_bbox = [382, y1 + 18, 660, y2 - 18]
        draw.rounded_rectangle(value_bbox, radius=16, fill=(237, 242, 249))
        draw.text((521, (value_bbox[1] + value_bbox[3]) // 2), facts[field],
                  font=_font(22, bold=True), fill=(38, 47, 63), anchor="mm")
        evidence[field] = value_bbox

    draw.line([76, 776, 692, 776], fill=(220, 226, 235), width=2)
    draw.text((76, 818), "CONTROLLED VISUAL MEMORY", font=_font(16, bold=True),
              fill=(107, 116, 132))
    draw.ellipse([646, 807, 670, 831], fill=accent)
    return image, evidence


def _ordered_memory_ids(
    memory_ids: list[str], *, seed: int, trial_id: str
) -> list[str]:
    digest = hashlib.sha256(f"{seed}:{trial_id}".encode("utf-8")).digest()
    trial_rng = random.Random(int.from_bytes(digest[:16], "big"))
    result = list(memory_ids)
    trial_rng.shuffle(result)
    return result


def _byte_contract(n_memories: int) -> dict[str, Any]:
    if n_memories <= 0:
        return {
            "total_payload_budget_bytes": TOTAL_PAYLOAD_BUDGET_BYTES,
            "per_memory_budget_bytes": None,
            "actual_package_bytes_required": True,
            "padding_forbidden": True,
        }
    return {
        "total_payload_budget_bytes": TOTAL_PAYLOAD_BUDGET_BYTES,
        "per_memory_budget_bytes": TOTAL_PAYLOAD_BUDGET_BYTES // n_memories,
        "actual_package_bytes_required": True,
        "padding_forbidden": True,
    }


def _build_d3_trials(
    episode_id: str,
    current_memory_id: str,
    distractor_ids: list[str],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    retrieval: list[dict[str, Any]] = []
    for candidate_count in RETRIEVAL_CANDIDATE_COUNTS:
        trial_id = f"{episode_id}:d3:retrieval:n{candidate_count}"
        candidates = [current_memory_id, *distractor_ids[: candidate_count - 1]]
        candidates = _ordered_memory_ids(candidates, seed=seed, trial_id=trial_id)
        retrieval.append({
            "trial_id": trial_id,
            "diagnostic_component": "retrieval",
            "candidate_count": candidate_count,
            "candidate_memory_ids": candidates,
            "relevant_memory_ids": [current_memory_id],
            "recall_at_k": [1, min(2, candidate_count)],
            "oracle_task_memory_ids": [current_memory_id],
            "task_score_selection": "oracle_relevant_only",
            "retrieval_bypassed": False,
            "inference_distractors_in_oracle_score": 0,
            "failure_taxonomy": "stored_not_retrieved",
            "byte_contract": _byte_contract(candidate_count),
        })

    interference: list[dict[str, Any]] = []
    for distractor_count in INTERFERENCE_DISTRACTOR_COUNTS:
        trial_id = f"{episode_id}:d3:interference:d{distractor_count}"
        included = [current_memory_id, *distractor_ids[:distractor_count]]
        included = _ordered_memory_ids(included, seed=seed, trial_id=trial_id)
        interference.append({
            "trial_id": trial_id,
            "diagnostic_component": "inference_interference",
            "distractor_count": distractor_count,
            "preselected_memory_ids": included,
            "relevant_memory_ids": [current_memory_id],
            "relevant_always_included": True,
            "retrieval_bypassed": True,
            "target_position": included.index(current_memory_id),
            "ordering_policy": "deterministic_seeded_permutation",
            "stale_target_revision_included": False,
            "failure_taxonomy": "retrieved_not_used",
            "byte_contract": _byte_contract(len(included)),
        })

    block_set_id = f"{episode_id}:d3:composition:blockset4"
    block_ids = _ordered_memory_ids(
        [current_memory_id, *distractor_ids], seed=seed, trial_id=block_set_id
    )
    composition = []
    for mode in ("independent_serialize_concat", "joint_prefill"):
        composition.append({
            "trial_id": f"{episode_id}:d3:composition:{mode}",
            "diagnostic_component": "composition",
            "block_set_id": block_set_id,
            "composition_mode": mode,
            "memory_ids": block_ids,
            "relevant_memory_ids": [current_memory_id],
            "retrieval_bypassed": True,
            "position_metadata_required": True,
            "byte_contract": _byte_contract(len(block_ids)),
        })

    return {
        "retrieval": retrieval,
        "interference": interference,
        "composition": composition,
    }


def _build_d4_trials(
    episode_id: str,
    old_memory_id: str,
    current_memory_id: str,
    *,
    time_gap: str,
    state_change: str,
    stale_answer: str | None,
) -> list[dict[str, Any]]:
    trial_memories = {
        "old_only": [old_memory_id],
        "current_only": [current_memory_id],
        "old+current": [old_memory_id, current_memory_id],
        "no_memory": [],
    }
    changed = state_change == "changed"
    trials = []
    for condition in TEMPORAL_MEMORY_CONDITIONS:
        memory_ids = trial_memories[condition]
        old_present = old_memory_id in memory_ids
        current_present = current_memory_id in memory_ids
        trials.append({
            "trial_id": f"{episode_id}:d4:{condition}",
            "diagnostic_component": "temporal_state",
            "factorial_cell": f"{time_gap}__{state_change}",
            "time_gap": time_gap,
            "time_gap_seconds": TIME_GAP_SECONDS[time_gap],
            "state_change": state_change,
            "memory_condition": condition,
            "memory_ids": memory_ids,
            "old_revision_present": old_present,
            "current_revision_present": current_present,
            "stale_evidence_present": changed and old_present,
            "cross_revision_conflict_present": changed and old_present and current_present,
            "stale_capture_eligible": changed and old_present,
            "stale_answer": stale_answer if changed else None,
            "expected_answer_source": current_memory_id,
            "byte_contract": _byte_contract(len(memory_ids)),
        })
    return trials


def build_episode(
    index: int,
    seed: int,
    image_prefix: str,
) -> tuple[list[tuple[str, Image.Image]], dict[str, Any]]:
    """Build one balanced-factor episode and its five memory images."""

    if index < 0:
        raise ValueError("index must be non-negative")
    rng = _rng_for(seed, index)
    episode_id = f"memory_dynamics_{index:04d}"
    entity_id = f"entity:{episode_id}:target"
    time_gap, state_change = FACTOR_CELLS[index % len(FACTOR_CELLS)]
    state_changed = state_change == "changed"
    # Rotate the queried field across factorial cells instead of tying a fact
    # type permanently to one time/state condition.  Every consecutive 16
    # episodes form a complete 2x2 x 4-field Latin-style block.
    cell_index = index % len(FACTOR_CELLS)
    replicate_index = index // len(FACTOR_CELLS)
    nuisance_index = (cell_index + replicate_index) % len(FIELD_VALUES)
    mutable_field = tuple(FIELD_VALUES)[nuisance_index]
    values = list(FIELD_VALUES[mutable_field])
    rng.shuffle(values)
    current_answer = values[0]
    old_answer = values[1] if state_changed else current_answer
    distractor_answers = [value for value in values if value not in {current_answer, old_answer}][:3]
    if len(distractor_answers) != 3:  # pragma: no cover - guarded by constants
        raise RuntimeError("each mutable field needs at least five distinct values")

    base_at = datetime(2025, 1, 1, 9, tzinfo=timezone.utc) + timedelta(days=index * 2)
    old_observed_at = base_at
    current_observed_at = base_at + timedelta(hours=1)
    query_at = current_observed_at + timedelta(seconds=TIME_GAP_SECONDS[time_gap])

    used_codes: set[str] = set()
    target_code = _project_code(rng, used_codes)
    target_base_facts = _base_facts(rng)
    old_facts = dict(target_base_facts)
    current_facts = dict(target_base_facts)
    old_facts[mutable_field] = old_answer
    current_facts[mutable_field] = current_answer
    # Keep the visual theme fixed so color itself cannot identify a factor cell.
    accent = ACCENTS[0]

    image_records: list[tuple[str, Image.Image]] = []
    memories: list[dict[str, Any]] = []

    def add_memory(
        *,
        role: str,
        entity: str,
        project_code: str,
        revision: str,
        observed_at: datetime,
        facts: dict[str, str],
        answer_label: str,
    ) -> dict[str, Any]:
        filename = f"{episode_id}_{role}.png"
        image_path = f"{image_prefix}/{filename}" if image_prefix else filename
        image, evidence_bboxes = _draw_screen(
            project_code=project_code,
            facts=facts,
            revision=revision,
            observed_at=_iso(observed_at),
            accent=accent,
        )
        memory = {
            "memory_id": _memory_id(episode_id, role, revision),
            "entity_id": entity,
            "revision_id": _revision_id(entity, revision),
            "revision": revision,
            "role": role,
            "observed_at": _iso(observed_at),
            "image": image_path,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "project_code": project_code,
            "facts": facts,
            "queried_fact": {
                "field": mutable_field,
                "value": facts[mutable_field],
                "evidence_bbox": evidence_bboxes[mutable_field],
                "evidence_label": answer_label,
            },
        }
        image_records.append((filename, image))
        memories.append(memory)
        return memory

    old_memory = add_memory(
        role="target_old",
        entity=entity_id,
        project_code=target_code,
        revision="r1",
        observed_at=old_observed_at,
        facts=old_facts,
        answer_label="stale_conflict" if state_changed else "historical_consistent",
    )
    current_memory = add_memory(
        role="target_current",
        entity=entity_id,
        project_code=target_code,
        revision="r2",
        observed_at=current_observed_at,
        facts=current_facts,
        answer_label="current_truth",
    )

    distractor_memories = []
    for distractor_index, distractor_answer in enumerate(distractor_answers):
        distractor_entity = f"entity:{episode_id}:distractor:{distractor_index}"
        distractor_facts = _base_facts(rng)
        distractor_facts[mutable_field] = distractor_answer
        distractor_memories.append(add_memory(
            role=f"distractor_{distractor_index}",
            entity=distractor_entity,
            project_code=_project_code(rng, used_codes),
            revision="r1",
            observed_at=current_observed_at - timedelta(minutes=distractor_index + 1),
            facts=distractor_facts,
            answer_label="entity_distractor",
        ))

    distractor_ids = [memory["memory_id"] for memory in distractor_memories]
    stale_answers = [old_answer] if state_changed else []
    question = {
        "question_id": f"{episode_id}:question:current_{mutable_field}",
        "question": (
            f"As of {_iso(query_at)}, what is the current {mutable_field} of "
            f"project {target_code}? Answer with the value only."
        ),
        "answer_type": "categorical",
        "acceptable_answers": [current_answer],
        "current_answer": current_answer,
        "stale_answers": stale_answers,
        "queried_entity_id": entity_id,
        "queried_project_code": target_code,
        "queried_field": mutable_field,
        "query_at": _iso(query_at),
        "current_evidence": {
            "memory_id": current_memory["memory_id"],
            "revision_id": current_memory["revision_id"],
            "evidence_bbox": current_memory["queried_fact"]["evidence_bbox"],
            "evidence_label": "current_truth",
        },
        "conflicting_evidence": ([{
            "memory_id": old_memory["memory_id"],
            "revision_id": old_memory["revision_id"],
            "answer": old_answer,
            "evidence_bbox": old_memory["queried_fact"]["evidence_bbox"],
            "evidence_label": "stale_conflict",
        }] if state_changed else []),
        "distractor_evidence": [{
            "memory_id": memory["memory_id"],
            "answer": memory["queried_fact"]["value"],
            "evidence_bbox": memory["queried_fact"]["evidence_bbox"],
            "evidence_label": "entity_distractor",
        } for memory in distractor_memories],
    }

    row = {
        "dataset": DATASET_NAME,
        "dataset_revision": GENERATOR_VERSION,
        "split": "mechanism_probe",
        "episode_id": episode_id,
        "generation_index": index,
        "domain": DOMAIN,
        "synthetic": True,
        "probe_kind": PROBE_KIND,
        "claim_scope": CLAIM_SCOPE,
        "selection_seed": seed,
        "factorial": {
            "cell_id": f"{time_gap}__{state_change}",
            "time_gap": time_gap,
            "time_gap_seconds": TIME_GAP_SECONDS[time_gap],
            "state_change": state_change,
            "state_changed": state_changed,
            "axes_are_orthogonal_by_construction": True,
            "old_observed_at": _iso(old_observed_at),
            "current_observed_at": _iso(current_observed_at),
            "query_at": _iso(query_at),
        },
        "memory_write_contract": {
            "future_question_hidden": True,
            "one_package_per_memory": True,
            "actual_package_bytes_required": True,
            "source_denial_at_read_required": True,
        },
        "memories": memories,
        "question": question,
        "d3_trials": _build_d3_trials(
            episode_id,
            current_memory["memory_id"],
            distractor_ids,
            seed=seed,
        ),
        "d4_trials": _build_d4_trials(
            episode_id,
            old_memory["memory_id"],
            current_memory["memory_id"],
            time_gap=time_gap,
            state_change=state_change,
            stale_answer=old_answer if state_changed else None,
        ),
        "error_taxonomy": [
            "stored_not_retrieved",
            "retrieved_not_used",
            "stale_conflict",
            "uncertain",
        ],
    }
    validate_manifest_row(row)
    return image_records, row


def _validate_bbox(bbox: Any, field_name: str) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{field_name} must be a four-number list")
    if any(not isinstance(value, int) for value in bbox):
        raise ValueError(f"{field_name} coordinates must be integers")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= IMAGE_WIDTH and 0 <= y1 < y2 <= IMAGE_HEIGHT):
        raise ValueError(f"{field_name} is outside the image: {bbox}")


def validate_manifest_row(row: dict[str, Any]) -> None:
    """Validate D3/D4 separation and temporal ground-truth invariants."""

    required = {
        "dataset", "dataset_revision", "episode_id", "domain", "synthetic",
        "probe_kind", "claim_scope", "factorial", "memories", "question",
        "d3_trials", "d4_trials", "memory_write_contract",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(f"manifest row is missing fields: {sorted(missing)}")
    if row["dataset"] != DATASET_NAME or row["probe_kind"] != PROBE_KIND:
        raise ValueError("row does not identify this controlled D3/D4 fixture")
    if row["synthetic"] is not True or row["domain"] != DOMAIN:
        raise ValueError("row must be labeled as the controlled synthetic domain")

    memories = row["memories"]
    if not isinstance(memories, list) or len(memories) != 5:
        raise ValueError("each episode must contain old/current targets and three distractors")
    memory_ids = [memory.get("memory_id") for memory in memories]
    if len(set(memory_ids)) != len(memory_ids):
        raise ValueError("memory_id values must be unique within an episode")
    by_role = {memory.get("role"): memory for memory in memories}
    required_roles = {"target_old", "target_current", "distractor_0", "distractor_1", "distractor_2"}
    if set(by_role) != required_roles:
        raise ValueError(f"unexpected memory roles: {sorted(by_role)}")
    old = by_role["target_old"]
    current = by_role["target_current"]
    if old["entity_id"] != current["entity_id"]:
        raise ValueError("old and current revisions must share a stable entity_id")
    if old["revision_id"] == current["revision_id"]:
        raise ValueError("old and current revision_id values must differ")
    if _parse_iso(old["observed_at"]) >= _parse_iso(current["observed_at"]):
        raise ValueError("old revision must precede current revision")

    for memory in memories:
        required_memory = {
            "memory_id", "entity_id", "revision_id", "revision", "role", "observed_at",
            "image", "image_width", "image_height", "project_code", "facts", "queried_fact",
        }
        if required_memory - memory.keys():
            raise ValueError(f"memory {memory.get('memory_id')} lacks required metadata")
        if memory["image_width"] != IMAGE_WIDTH or memory["image_height"] != IMAGE_HEIGHT:
            raise ValueError("memory image dimensions do not match the fixture")
        _parse_iso(memory["observed_at"])
        _validate_bbox(memory["queried_fact"]["evidence_bbox"], "queried_fact.evidence_bbox")

    question = row["question"]
    required_question = {
        "question_id", "question", "acceptable_answers", "current_answer", "stale_answers",
        "queried_entity_id", "queried_project_code", "queried_field", "query_at",
        "current_evidence", "conflicting_evidence", "distractor_evidence",
    }
    if required_question - question.keys():
        raise ValueError("question lacks evidence or answer labels")
    if question["queried_entity_id"] != current["entity_id"]:
        raise ValueError("question entity must point to the target revisions")
    if question["current_evidence"]["memory_id"] != current["memory_id"]:
        raise ValueError("current evidence must point to target_current")
    if question["acceptable_answers"] != [question["current_answer"]]:
        raise ValueError("the expected answer must always be the current answer")
    _parse_iso(question["query_at"])
    _validate_bbox(question["current_evidence"]["evidence_bbox"], "current_evidence")

    factors = row["factorial"]
    time_gap = factors.get("time_gap")
    state_change = factors.get("state_change")
    if (time_gap, state_change) not in FACTOR_CELLS:
        raise ValueError("factorial cell must be one of the balanced 2x2 cells")
    expected_gap = TIME_GAP_SECONDS[time_gap]
    observed_gap = int((_parse_iso(factors["query_at"]) -
                        _parse_iso(factors["current_observed_at"])).total_seconds())
    if factors["time_gap_seconds"] != expected_gap or observed_gap != expected_gap:
        raise ValueError("time gap timestamps and factor label disagree")
    old_value = old["queried_fact"]["value"]
    current_value = current["queried_fact"]["value"]
    changed = old_value != current_value
    if changed != factors["state_changed"] or changed != (state_change == "changed"):
        raise ValueError("state-change label disagrees with old/current facts")
    expected_stale = [old_value] if changed else []
    if question["stale_answers"] != expected_stale:
        raise ValueError("stale-answer labels disagree with the target revisions")
    if len(question["conflicting_evidence"]) != int(changed):
        raise ValueError("conflicting evidence must exist exactly when state changed")
    distractor_values = [item["answer"] for item in question["distractor_evidence"]]
    if len(distractor_values) != 3 or len(set(distractor_values)) != 3:
        raise ValueError("three distinct distractor answers are required")
    if current_value in distractor_values or old_value in distractor_values:
        raise ValueError("distractor answers must not alias target answers")

    valid_ids = set(memory_ids)
    d3 = row["d3_trials"]
    retrieval = d3.get("retrieval", [])
    if [trial["candidate_count"] for trial in retrieval] != list(RETRIEVAL_CANDIDATE_COUNTS):
        raise ValueError("retrieval trials must use N=1/2/4")
    for trial in retrieval:
        candidates = trial["candidate_memory_ids"]
        if trial["diagnostic_component"] != "retrieval" or trial["retrieval_bypassed"]:
            raise ValueError("retrieval trials must isolate the retrieval component")
        if len(candidates) != trial["candidate_count"] or not set(candidates) <= valid_ids:
            raise ValueError("invalid retrieval candidate set")
        if trial["relevant_memory_ids"] != [current["memory_id"]]:
            raise ValueError("retrieval relevance must point only to current target memory")
        if trial["oracle_task_memory_ids"] != [current["memory_id"]]:
            raise ValueError("oracle task score must exclude inference distractors")

    interference = d3.get("interference", [])
    if [trial["distractor_count"] for trial in interference] != list(INTERFERENCE_DISTRACTOR_COUNTS):
        raise ValueError("interference trials must use 0/1/3 distractors")
    for trial in interference:
        included = trial["preselected_memory_ids"]
        if not trial["retrieval_bypassed"] or not trial["relevant_always_included"]:
            raise ValueError("interference trials must bypass retrieval and include relevance")
        if current["memory_id"] not in included or old["memory_id"] in included:
            raise ValueError("interference context must include current but not stale target")
        if len(included) != trial["distractor_count"] + 1 or not set(included) <= valid_ids:
            raise ValueError("invalid preselected interference context")

    composition = d3.get("composition", [])
    if {trial.get("composition_mode") for trial in composition} != {
        "independent_serialize_concat", "joint_prefill"
    }:
        raise ValueError("composition requires independent and joint arms")
    if len({tuple(trial["memory_ids"]) for trial in composition}) != 1:
        raise ValueError("composition arms must use identical ordered blocks")

    d4 = row["d4_trials"]
    if [trial["memory_condition"] for trial in d4] != list(TEMPORAL_MEMORY_CONDITIONS):
        raise ValueError("D4 must contain old/current/both/none conditions")
    for trial in d4:
        if trial["time_gap"] != time_gap or trial["state_change"] != state_change:
            raise ValueError("D4 trial factors disagree with the episode cell")
        if not set(trial["memory_ids"]) <= {old["memory_id"], current["memory_id"]}:
            raise ValueError("D4 conditions may reference only target revisions")
        if trial["expected_answer_source"] != current["memory_id"]:
            raise ValueError("D4 ground truth must always come from current state")


def _manifest_owned_by_generator(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            first = next(line for line in handle if line.strip())
        row = json.loads(first)
    except (OSError, StopIteration, json.JSONDecodeError):
        return False
    return (row.get("dataset") == DATASET_NAME
            and row.get("dataset_revision") == GENERATOR_VERSION
            and row.get("probe_kind") == PROBE_KIND)


def _meta_owned_by_generator(path: Path) -> bool:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (meta.get("generator_version") == GENERATOR_VERSION
            and meta.get("probe_kind") == PROBE_KIND)


def _check_output_safety(
    image_paths: Iterable[Path], manifest_path: Path, meta_path: Path, *, overwrite: bool
) -> None:
    images = list(image_paths)
    existing_images = [path for path in images if path.exists()]
    existing_controls = [path for path in (manifest_path, meta_path) if path.exists()]
    if not overwrite:
        existing = [*existing_controls, *existing_images]
        if existing:
            shown = ", ".join(str(path) for path in existing[:3])
            raise FileExistsError(f"refusing to overwrite existing output(s): {shown}")
        return
    if manifest_path.exists() and not _manifest_owned_by_generator(manifest_path):
        raise FileExistsError(f"existing manifest is not owned by this generator: {manifest_path}")
    if meta_path.exists() and not _meta_owned_by_generator(meta_path):
        raise FileExistsError(f"existing meta file is not owned by this generator: {meta_path}")
    if existing_images and not (manifest_path.exists() or meta_path.exists()):
        raise FileExistsError(
            "existing image names cannot be proven to belong to this generator; "
            "choose another --image-dir"
        )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _default_image_prefix(image_dir: Path) -> str:
    try:
        return image_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return image_dir.resolve().as_posix()


def generate_dataset(
    image_dir: Path,
    manifest_path: Path,
    *,
    meta_path: Path | None = None,
    image_prefix: str | None = None,
    n_episodes: int = DEFAULT_N_EPISODES,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Generate controlled D3/D4 images, JSONL labels, and dataset metadata."""

    image_dir = Path(image_dir)
    manifest_path = Path(manifest_path)
    meta_path = Path(meta_path) if meta_path is not None else manifest_path.with_suffix(".meta.json")
    if n_episodes <= 0 or n_episodes % len(FACTOR_CELLS) != 0:
        raise ValueError("n_episodes must be a positive multiple of four for a balanced 2x2")
    if manifest_path == meta_path:
        raise ValueError("manifest_path and meta_path must differ")
    prefix = image_prefix.rstrip("/") if image_prefix else _default_image_prefix(image_dir)

    expected_filenames = [
        f"memory_dynamics_{index:04d}_{role}.png"
        for index in range(n_episodes)
        for role in ("target_old", "target_current", "distractor_0", "distractor_1", "distractor_2")
    ]
    image_paths = [image_dir / filename for filename in expected_filenames]
    _check_output_safety(image_paths, manifest_path, meta_path, overwrite=overwrite)

    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for index in range(n_episodes):
        image_records, row = build_episode(index, seed, prefix)
        for filename, image in image_records:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", compress_level=9)
            payload = buffer.getvalue()
            payloads[filename] = payload
        for memory in row["memories"]:
            filename = Path(memory["image"]).name
            memory["image_sha256"] = hashlib.sha256(payloads[filename]).hexdigest()
            memory["source_image_bytes"] = len(payloads[filename])
        validate_manifest_row(row)
        rows.append(row)

    if set(payloads) != set(expected_filenames):
        raise RuntimeError("generated image names do not match the planned output set")
    for filename in expected_filenames:
        _atomic_write(image_dir / filename, payloads[filename])

    manifest_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    cell_counts = {
        f"{time_gap}__{state_change}": n_episodes // len(FACTOR_CELLS)
        for time_gap, state_change in FACTOR_CELLS
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset": DATASET_NAME,
        "domain": DOMAIN,
        "synthetic": True,
        "probe_kind": PROBE_KIND,
        "claim_scope": CLAIM_SCOPE,
        "seed": seed,
        "n_episodes": n_episodes,
        "n_images": n_episodes * 5,
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "factorial_cells": cell_counts,
        "retrieval_candidate_counts": list(RETRIEVAL_CANDIDATE_COUNTS),
        "interference_distractor_counts": list(INTERFERENCE_DISTRACTOR_COUNTS),
        "temporal_memory_conditions": list(TEMPORAL_MEMORY_CONDITIONS),
    }
    _atomic_write(manifest_path, manifest_payload)
    _atomic_write(
        meta_path,
        (json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--image-prefix", default=None)
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = generate_dataset(
        args.image_dir,
        args.manifest,
        meta_path=args.meta,
        image_prefix=args.image_prefix,
        n_episodes=args.n_episodes,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    meta_path = args.meta or args.manifest.with_suffix(".meta.json")
    print(
        f"wrote {len(rows)} controlled D3/D4 episodes and {len(rows) * 5} images "
        f"-> {args.manifest} (meta: {meta_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
