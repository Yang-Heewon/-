"""Generate a deterministic controlled D5 offline-action proxy fixture.

Each episode contains an ordered two-screen synthetic mobile/project workflow:

1. an old dashboard observation carrying a task-relevant visual memory cue,
2. an executed ``OPEN`` click that changes the screen, and
3. a current observation whose correct next action is click, type, or scroll.

The previous ``OPEN`` action is explicitly marked invalid after navigation.
The next action target has a stable element ID and pixel bounding box.  This is
only a controlled mechanism probe: it is not an interactive environment or a
real agent-trajectory benchmark and cannot support real-world trajectory,
deployment, or novelty claims by itself.  The generator never loads a model.

Examples::

    python -m vlm_diagnosis.scripts.gen_action_proxy
    python -m vlm_diagnosis.scripts.gen_action_proxy --n-episodes 6 \
        --image-dir /tmp/action-proxy/images \
        --manifest /tmp/action-proxy/manifest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = ROOT / "data" / "action_proxy_controlled"
DEFAULT_MANIFEST = ROOT / "experiments" / "manifests" / "action_proxy_controlled.jsonl"
DEFAULT_N_EPISODES = 24
DEFAULT_SEED = 42

IMAGE_WIDTH = 768
IMAGE_HEIGHT = 896
GENERATOR_VERSION = "action-proxy-generator-v1"
SCHEMA_VERSION = "action-proxy-schema-v1"
DATASET_NAME = "synthetic_action_proxy_controlled"
DOMAIN = "synthetic_mobile_project_ui"
PROBE_KIND = "controlled_d5_offline_action_mechanism_probe"
CLAIM_SCOPE = (
    "synthetic offline action-selection mechanism probe only; no real agent-trajectory, "
    "interactive-environment, deployment, generality, or novelty claims"
)

ACTION_TYPES = ("click", "type", "scroll")
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

ACCENTS = (
    (51, 104, 214),
    (35, 145, 151),
    (133, 77, 190),
    (42, 153, 90),
)
CLICK_LABELS = ("APPROVE", "ARCHIVE", "SHARE", "PUBLISH", "REVIEW", "EXPORT")
SECTION_LABELS = ("OVERVIEW", "PLANNING", "DETAILS", "USAGE", "AUDIT", "BILLING")
TOKEN_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _rng_for(seed: int, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:action-proxy:{index}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _element(
    element_id: str,
    bbox: list[int],
    *,
    role: str,
    label: str | None,
    interactive: bool,
    action_types: list[str],
    state: str = "visible",
) -> dict[str, Any]:
    return {
        "element_id": element_id,
        "bbox": bbox,
        "role": role,
        "label": label,
        "interactive": interactive,
        "action_types": action_types,
        "state": state,
    }


def _draw_shell(draw: ImageDraw.ImageDraw, *, project_code: str, subtitle: str) -> None:
    draw.rounded_rectangle(
        [24, 18, IMAGE_WIDTH - 24, IMAGE_HEIGHT - 18],
        radius=36,
        fill=(255, 255, 255),
        outline=(204, 212, 224),
        width=3,
    )
    draw.rounded_rectangle([48, 44, 720, 158], radius=26, fill=(245, 248, 252))
    draw.text((76, 76), "PROJECT FLOW", font=_font(27, bold=True), fill=(37, 45, 60))
    draw.text((76, 119), subtitle, font=_font(17), fill=(102, 112, 129))
    draw.rounded_rectangle([512, 68, 688, 128], radius=15, fill=(228, 234, 243))
    draw.text((600, 98), project_code, font=_font(21, bold=True), fill=(49, 60, 78), anchor="mm")


def _draw_old_screen(
    *,
    episode_id: str,
    project_code: str,
    decoy_code: str,
    cue_kind: str,
    cue_value: str,
    accent: tuple[int, int, int],
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, list[int]]]:
    """Draw the old dashboard containing the remembered cue and OPEN target."""

    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (235, 240, 247))
    draw = ImageDraw.Draw(image)
    _draw_shell(draw, project_code=project_code, subtitle="WORKSPACE / SAVED TASK")

    draw.text((72, 202), "TARGET PROJECT", font=_font(17, bold=True), fill=(92, 102, 119))
    card_bbox = [64, 230, 704, 540]
    draw.rounded_rectangle(card_bbox, radius=26, fill=(248, 250, 253),
                           outline=accent, width=3)
    draw.rounded_rectangle([88, 258, 216, 310], radius=13, fill=accent)
    draw.text((152, 284), project_code, font=_font(17, bold=True), fill="white", anchor="mm")
    draw.text((92, 350), "SAVED VISUAL CUE", font=_font(17, bold=True), fill=(98, 108, 126))

    cue_bbox = [92, 384, 676, 450]
    draw.rounded_rectangle(cue_bbox, radius=17, fill=(236, 241, 248))
    cue_prefix = {"click": "ACTION", "type": "TOKEN", "scroll": "SECTION"}[cue_kind]
    draw.text((116, 417), f"{cue_prefix}: {cue_value}", font=_font(25, bold=True),
              fill=(39, 49, 66), anchor="lm")

    open_bbox = [442, 470, 676, 520]
    draw.rounded_rectangle(open_bbox, radius=15, fill=accent)
    draw.text((559, 495), "OPEN", font=_font(21, bold=True), fill="white", anchor="mm")

    # A visually similar decoy prevents the old action from being reducible to
    # a fixed coordinate independent of the task's project identity.
    draw.text((72, 588), "OTHER PROJECT", font=_font(17, bold=True), fill=(92, 102, 119))
    draw.rounded_rectangle([64, 616, 704, 788], radius=24, fill=(249, 250, 252),
                           outline=(220, 226, 235), width=2)
    draw.text((96, 660), decoy_code, font=_font(21, bold=True), fill=(67, 77, 95))
    decoy_open_bbox = [442, 712, 676, 762]
    draw.rounded_rectangle(decoy_open_bbox, radius=15, fill=(184, 193, 207))
    draw.text((559, 737), "OPEN", font=_font(21, bold=True), fill="white", anchor="mm")

    cue_id = f"element:{episode_id}:r1:memory_cue"
    open_id = f"element:{episode_id}:r1:open_target"
    elements = [
        _element(
            f"element:{episode_id}:r1:target_card", card_bbox,
            role="project_card", label=project_code, interactive=False, action_types=[],
        ),
        _element(
            cue_id, cue_bbox, role="memory_cue", label=f"{cue_prefix}: {cue_value}",
            interactive=False, action_types=[],
        ),
        _element(
            open_id, open_bbox, role="button", label="OPEN", interactive=True,
            action_types=["click"], state="enabled",
        ),
        _element(
            f"element:{episode_id}:r1:decoy_open", decoy_open_bbox,
            role="button", label="OPEN", interactive=True, action_types=["click"],
            state="enabled",
        ),
    ]
    return image, elements, {"cue": cue_bbox, "open": open_bbox}


def _draw_current_screen(
    *,
    episode_id: str,
    project_code: str,
    action_type: str,
    cue_value: str,
    click_options: list[str],
    scroll_direction: str | None,
    accent: tuple[int, int, int],
) -> tuple[Image.Image, list[dict[str, Any]], str, list[int]]:
    """Draw the post-navigation screen and return the next-action target."""

    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (235, 240, 247))
    draw = ImageDraw.Draw(image)
    _draw_shell(draw, project_code=project_code, subtitle="CURRENT PROJECT / REVISION R2")
    elements: list[dict[str, Any]] = []

    if action_type == "click":
        draw.text((72, 204), "EXECUTE SAVED ACTION", font=_font(18, bold=True),
                  fill=(88, 98, 116))
        draw.text((72, 240), "Choose the action saved on the previous workspace card.",
                  font=_font(17), fill=(101, 111, 128))
        target_id = ""
        target_bbox: list[int] = []
        for option_index, label in enumerate(click_options):
            y1 = 312 + option_index * 132
            bbox = [112, y1, 656, y1 + 86]
            element_id = f"element:{episode_id}:r2:action:{label.lower()}"
            draw.rounded_rectangle(bbox, radius=21, fill=(248, 250, 253),
                                   outline=accent, width=3)
            draw.text((384, y1 + 43), label, font=_font(25, bold=True),
                      fill=(45, 55, 72), anchor="mm")
            elements.append(_element(
                element_id, bbox, role="button", label=label, interactive=True,
                action_types=["click"], state="enabled",
            ))
            if label == cue_value:
                target_id, target_bbox = element_id, bbox
        if not target_id:  # pragma: no cover - guarded by construction and validation
            raise RuntimeError("saved click action is absent from current options")

    elif action_type == "type":
        draw.text((72, 204), "RESTORE WORKSPACE TOKEN", font=_font(18, bold=True),
                  fill=(88, 98, 116))
        draw.text((72, 240), "Enter the token saved on the previous workspace card.",
                  font=_font(17), fill=(101, 111, 128))
        draw.rounded_rectangle([72, 302, 696, 618], radius=26, fill=(248, 250, 253),
                               outline=(221, 227, 236), width=2)
        draw.text((108, 348), "WORKSPACE TOKEN", font=_font(17, bold=True),
                  fill=(95, 105, 122))
        target_bbox = [108, 386, 660, 470]
        target_id = f"element:{episode_id}:r2:token_field"
        draw.rounded_rectangle(target_bbox, radius=17, fill="white", outline=accent, width=3)
        draw.text((132, 428), "Type saved token", font=_font(21), fill=(153, 162, 176),
                  anchor="lm")
        draw.rounded_rectangle([436, 512, 660, 574], radius=16, fill=(190, 198, 211))
        draw.text((548, 543), "CONTINUE", font=_font(20, bold=True), fill="white", anchor="mm")
        elements.extend([
            _element(
                target_id, target_bbox, role="text_field", label="WORKSPACE TOKEN",
                interactive=True, action_types=["click", "type"], state="empty_focused",
            ),
            _element(
                f"element:{episode_id}:r2:continue", [436, 512, 660, 574],
                role="button", label="CONTINUE", interactive=False, action_types=[],
                state="disabled_until_text_entry",
            ),
        ])

    elif action_type == "scroll":
        if scroll_direction not in {"up", "down"}:
            raise ValueError("scroll_direction must be up or down for a scroll episode")
        draw.text((72, 204), "FIND SAVED SECTION", font=_font(18, bold=True),
                  fill=(88, 98, 116))
        draw.text((72, 240), "Navigate toward the section saved on the workspace card.",
                  font=_font(17), fill=(101, 111, 128))
        target_bbox = [72, 286, 696, 760]
        target_id = f"element:{episode_id}:r2:scroll_container"
        draw.rounded_rectangle(target_bbox, radius=24, fill=(248, 250, 253),
                               outline=(220, 226, 235), width=2)
        visible_sections = ("DETAILS", "USAGE")
        for section_index, section in enumerate(visible_sections):
            y1 = 330 + section_index * 174
            draw.rounded_rectangle([104, y1, 624, y1 + 128], radius=19, fill="white",
                                   outline=(226, 231, 238), width=2)
            draw.text((132, y1 + 39), section, font=_font(22, bold=True),
                      fill=(55, 65, 82))
            draw.line([132, y1 + 74, 570, y1 + 74], fill=(227, 232, 239), width=4)
            draw.line([132, y1 + 96, 510, y1 + 96], fill=(232, 236, 242), width=4)
        # The thumb is deliberately centered: direction comes from combining
        # the remembered waypoint with the current middle-of-document state.
        draw.rounded_rectangle([658, 314, 674, 732], radius=8, fill=(225, 230, 237))
        draw.rounded_rectangle([658, 470, 674, 574], radius=8, fill=accent)
        elements.append(_element(
            target_id, target_bbox, role="scroll_container", label="PROJECT SECTIONS",
            interactive=True, action_types=["scroll"], state="middle_of_document",
        ))
    else:  # pragma: no cover - guarded by ACTION_TYPES and validation
        raise ValueError(f"unsupported action type: {action_type}")

    draw.line([76, 806, 692, 806], fill=(220, 226, 235), width=2)
    draw.text((76, 838), "CONTROLLED OFFLINE ACTION PROXY", font=_font(15, bold=True),
              fill=(108, 117, 133))
    return image, elements, target_id, target_bbox


def build_episode(
    index: int,
    seed: int,
    image_prefix: str,
) -> tuple[list[tuple[str, Image.Image]], dict[str, Any]]:
    """Build one ordered old/action/current/next-action proxy episode."""

    if index < 0:
        raise ValueError("index must be non-negative")
    rng = _rng_for(seed, index)
    episode_id = f"action_proxy_{index:04d}"
    action_type = ACTION_TYPES[index % len(ACTION_TYPES)]
    project_code = f"PRJ-{rng.randint(1000, 9999)}"
    decoy_code = f"PRJ-{rng.randint(1000, 9999)}"
    while decoy_code == project_code:
        decoy_code = f"PRJ-{rng.randint(1000, 9999)}"
    accent = ACCENTS[(index // len(ACTION_TYPES)) % len(ACCENTS)]

    click_options = rng.sample(list(CLICK_LABELS), 3)
    scroll_direction: str | None = None
    if action_type == "click":
        cue_value = rng.choice(click_options)
        task_goal = f"Open project {project_code} and execute its saved action."
        memory_dependency = "old_visual_action_label"
    elif action_type == "type":
        cue_value = "".join(rng.choices(TOKEN_ALPHABET, k=7))
        task_goal = f"Open project {project_code} and enter its saved workspace token."
        memory_dependency = "old_visual_text_payload"
    else:
        scroll_direction = "up" if (index // len(ACTION_TYPES)) % 2 == 0 else "down"
        cue_value = rng.choice(("OVERVIEW", "PLANNING")) if scroll_direction == "up" else rng.choice(
            ("AUDIT", "BILLING")
        )
        task_goal = f"Open project {project_code} and navigate to its saved section."
        memory_dependency = "old_visual_waypoint_plus_current_scroll_state"

    old_filename = f"{episode_id}_old.png"
    current_filename = f"{episode_id}_current.png"
    old_image, old_elements, old_boxes = _draw_old_screen(
        episode_id=episode_id,
        project_code=project_code,
        decoy_code=decoy_code,
        cue_kind=action_type,
        cue_value=cue_value,
        accent=accent,
    )
    current_image, current_elements, target_id, target_bbox = _draw_current_screen(
        episode_id=episode_id,
        project_code=project_code,
        action_type=action_type,
        cue_value=cue_value,
        click_options=click_options,
        scroll_direction=scroll_direction,
        accent=accent,
    )

    prefix = image_prefix.rstrip("/")
    old_image_path = f"{prefix}/{old_filename}" if prefix else old_filename
    current_image_path = f"{prefix}/{current_filename}" if prefix else current_filename
    old_observation_id = f"observation:{episode_id}:r1"
    current_observation_id = f"observation:{episode_id}:r2"
    old_revision_id = f"state:{episode_id}:r1"
    current_revision_id = f"state:{episode_id}:r2"
    previous_action_id = f"action:{episode_id}:open"
    next_action_id = f"action:{episode_id}:next"
    old_open_id = f"element:{episode_id}:r1:open_target"

    base_at = datetime(2025, 3, 1, 9, tzinfo=timezone.utc) + timedelta(hours=index * 2)
    observations = [
        {
            "observation_id": old_observation_id,
            "sequence_index": 0,
            "role": "old_state",
            "revision_id": old_revision_id,
            "observed_at": _iso(base_at),
            "image": old_image_path,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "elements": old_elements,
            "state_facts": {
                "screen": "workspace_dashboard",
                "project_code": project_code,
                "saved_cue_kind": action_type,
                "saved_cue_value": cue_value,
            },
        },
        {
            "observation_id": current_observation_id,
            "sequence_index": 2,
            "role": "current_state",
            "revision_id": current_revision_id,
            "observed_at": _iso(base_at + timedelta(seconds=2)),
            "image": current_image_path,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "elements": current_elements,
            "state_facts": {
                "screen": f"project_{action_type}_task",
                "project_code": project_code,
                "old_open_target_present": False,
                "scroll_position": "middle" if action_type == "scroll" else None,
            },
        },
    ]

    previous_action = {
        "action_id": previous_action_id,
        "sequence_index": 1,
        "action_type": "click",
        "target_element_id": old_open_id,
        "target_bbox": old_boxes["open"],
        "arguments": {"button": "left"},
        "source_observation_id": old_observation_id,
        "valid_in_revision_id": old_revision_id,
        "executed": True,
        "outcome": "navigated_to_current_project_task",
        "result_observation_id": current_observation_id,
    }

    if action_type == "click":
        arguments: dict[str, Any] = {"button": "left"}
        action_label = f"click:{cue_value}"
    elif action_type == "type":
        arguments = {"text": cue_value, "replace_existing": False}
        action_label = f"type:{cue_value}"
    else:
        arguments = {"direction": scroll_direction, "amount": "one_viewport"}
        action_label = f"scroll:{scroll_direction}"

    next_action = {
        "action_id": next_action_id,
        "sequence_index": 3,
        "action_type": action_type,
        "action_label": action_label,
        "target_element_id": target_id,
        "target_bbox": target_bbox,
        "arguments": arguments,
        "source_observation_id": current_observation_id,
        "valid_in_revision_id": current_revision_id,
        "memory_dependency": memory_dependency,
        "supporting_old_observation_id": old_observation_id,
        "supporting_old_evidence_bbox": old_boxes["cue"],
    }

    history = [
        {
            "event_id": f"event:{episode_id}:0",
            "sequence_index": 0,
            "event_type": "observation",
            "observation_id": old_observation_id,
        },
        {
            "event_id": f"event:{episode_id}:1",
            "sequence_index": 1,
            "event_type": "action",
            "action_id": previous_action_id,
        },
        {
            "event_id": f"event:{episode_id}:2",
            "sequence_index": 2,
            "event_type": "observation",
            "observation_id": current_observation_id,
        },
    ]
    invalidated_actions = [{
        "action_id": previous_action_id,
        "action_type": "click",
        "target_element_id": old_open_id,
        "target_bbox": old_boxes["open"],
        "valid_in_revision_id": old_revision_id,
        "invalid_in_revision_id": current_revision_id,
        "invalidated_at_event_id": history[-1]["event_id"],
        "reason_code": "target_removed_after_navigation",
        "previously_executed": True,
        "must_not_be_replayed": True,
    }]

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
        "real_trajectory_claim_allowed": False,
        "selection_seed": seed,
        "task_goal": task_goal,
        "episode_order": {
            "episode_order_index": index,
            "chronological_event_ids": [event["event_id"] for event in history],
            "observation_ids_chronological": [old_observation_id, current_observation_id],
            "history_cutoff_sequence_index": 2,
            "next_action_sequence_index": 3,
            "strict_chronology_required": True,
        },
        "observations": observations,
        "action_history": [previous_action],
        "event_history": history,
        "state_transition": {
            "from_revision_id": old_revision_id,
            "to_revision_id": current_revision_id,
            "caused_by_action_id": previous_action_id,
            "state_changed": True,
            "old_action_target_removed": True,
        },
        "invalidated_actions": invalidated_actions,
        "next_action": next_action,
        "offline_action_trials": [
            {
                "trial_id": f"{episode_id}:ordered_history",
                "condition": "ordered_history",
                "input_event_ids": [event["event_id"] for event in history],
                "input_observation_ids": [old_observation_id, current_observation_id],
                "expected_next_action_id": next_action_id,
                "expected_action_label": action_label,
                "retrieval_bypassed": True,
            },
            {
                "trial_id": f"{episode_id}:current_only",
                "condition": "current_only_control",
                "input_event_ids": [history[-1]["event_id"]],
                "input_observation_ids": [current_observation_id],
                "expected_next_action_id": next_action_id,
                "expected_action_label": action_label,
                "old_visual_cue_withheld": True,
                "retrieval_bypassed": True,
            },
        ],
        "scoring_contract": {
            "action_type_exact_match": True,
            "target_element_id_exact_match": True,
            "arguments_exact_match": True,
            "target_bbox_iou_optional_diagnostic": True,
            "offline_only": True,
            "environment_execution_performed": False,
        },
    }
    validate_manifest_row(row)
    return [(old_filename, old_image), (current_filename, current_image)], row


def _validate_bbox(bbox: Any, field_name: str) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{field_name} must be a four-number list")
    if any(not isinstance(value, int) for value in bbox):
        raise ValueError(f"{field_name} coordinates must be integers")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= IMAGE_WIDTH and 0 <= y1 < y2 <= IMAGE_HEIGHT):
        raise ValueError(f"{field_name} is outside the image: {bbox}")


def validate_manifest_row(row: dict[str, Any]) -> None:
    """Validate chronological, action-target, and stale-action invariants."""

    required = {
        "dataset", "dataset_revision", "episode_id", "generation_index", "domain",
        "synthetic", "probe_kind", "claim_scope", "real_trajectory_claim_allowed",
        "task_goal", "episode_order", "observations", "action_history", "event_history",
        "state_transition", "invalidated_actions", "next_action", "offline_action_trials",
        "scoring_contract",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(f"manifest row is missing fields: {sorted(missing)}")
    if row["dataset"] != DATASET_NAME or row["domain"] != DOMAIN:
        raise ValueError("row does not identify this controlled D5 fixture")
    if row["synthetic"] is not True or row["probe_kind"] != PROBE_KIND:
        raise ValueError("row must be labeled as a controlled synthetic probe")
    if row["real_trajectory_claim_allowed"] is not False:
        raise ValueError("controlled proxy must explicitly forbid real-trajectory claims")
    if not isinstance(row["claim_scope"], str) or "no real agent-trajectory" not in row["claim_scope"]:
        raise ValueError("claim scope must state the real-trajectory limitation")

    observations = row["observations"]
    if len(observations) != 2:
        raise ValueError("each episode must contain exactly old and current observations")
    if [item["role"] for item in observations] != ["old_state", "current_state"]:
        raise ValueError("observations must be ordered old_state then current_state")
    if [item["sequence_index"] for item in observations] != [0, 2]:
        raise ValueError("old/action/current sequence indices must be 0/1/2")
    if observations[0]["revision_id"] == observations[1]["revision_id"]:
        raise ValueError("old and current observations need different revisions")

    elements_by_observation: dict[str, dict[str, dict[str, Any]]] = {}
    for observation in observations:
        if observation["image_width"] != IMAGE_WIDTH or observation["image_height"] != IMAGE_HEIGHT:
            raise ValueError("observation dimensions do not match the controlled canvas")
        element_map: dict[str, dict[str, Any]] = {}
        for element in observation["elements"]:
            _validate_bbox(element.get("bbox"), f"element {element.get('element_id')} bbox")
            element_id = element.get("element_id")
            if not isinstance(element_id, str) or element_id in element_map:
                raise ValueError("element IDs must be non-empty and unique within an observation")
            element_map[element_id] = element
        elements_by_observation[observation["observation_id"]] = element_map

    history = row["event_history"]
    if [event["sequence_index"] for event in history] != [0, 1, 2]:
        raise ValueError("event history must be strictly ordered observation/action/observation")
    if [event["event_type"] for event in history] != ["observation", "action", "observation"]:
        raise ValueError("event history types must be observation/action/observation")
    order = row["episode_order"]
    if order["chronological_event_ids"] != [event["event_id"] for event in history]:
        raise ValueError("episode_order disagrees with event_history")
    if order["next_action_sequence_index"] != 3 or not order["strict_chronology_required"]:
        raise ValueError("next action must follow the ordered history at index 3")

    if len(row["action_history"]) != 1:
        raise ValueError("fixture requires one executed transition action")
    previous = row["action_history"][0]
    old_obs, current_obs = observations
    old_elements = elements_by_observation[old_obs["observation_id"]]
    current_elements = elements_by_observation[current_obs["observation_id"]]
    if previous["sequence_index"] != 1 or previous["action_type"] != "click":
        raise ValueError("transition action must be the OPEN click at index 1")
    if previous["target_element_id"] not in old_elements:
        raise ValueError("previous action target must exist in the old observation")
    if previous["target_bbox"] != old_elements[previous["target_element_id"]]["bbox"]:
        raise ValueError("previous action bbox must match its old element")
    if previous["target_element_id"] in current_elements:
        raise ValueError("consumed old action target must be absent from current state")

    invalidated = row["invalidated_actions"]
    if len(invalidated) != 1 or invalidated[0]["action_id"] != previous["action_id"]:
        raise ValueError("the executed OPEN action must be the explicitly invalidated action")
    if not invalidated[0]["must_not_be_replayed"]:
        raise ValueError("invalidated action must forbid replay")
    if invalidated[0]["valid_in_revision_id"] != old_obs["revision_id"]:
        raise ValueError("invalidated action must point to the old valid revision")
    if invalidated[0]["invalid_in_revision_id"] != current_obs["revision_id"]:
        raise ValueError("invalidated action must point to the current invalid revision")

    next_action = row["next_action"]
    if next_action["sequence_index"] != 3 or next_action["action_type"] not in ACTION_TYPES:
        raise ValueError("next action needs a supported type at sequence index 3")
    if next_action["source_observation_id"] != current_obs["observation_id"]:
        raise ValueError("next action must be grounded in the current observation")
    if next_action["target_element_id"] not in current_elements:
        raise ValueError("next action target must exist in the current observation")
    target = current_elements[next_action["target_element_id"]]
    if next_action["target_bbox"] != target["bbox"]:
        raise ValueError("next action bbox must match its current element")
    if next_action["action_type"] not in target["action_types"]:
        raise ValueError("target element does not support the labeled next action")
    if next_action["supporting_old_observation_id"] != old_obs["observation_id"]:
        raise ValueError("next action must identify its supporting old visual observation")
    _validate_bbox(next_action["supporting_old_evidence_bbox"], "supporting old evidence bbox")

    cue = old_obs["state_facts"]["saved_cue_value"]
    if next_action["action_type"] == "click":
        if next_action["action_label"] != f"click:{cue}":
            raise ValueError("click label must reproduce the saved old action cue")
    elif next_action["action_type"] == "type":
        if next_action["arguments"] != {"text": cue, "replace_existing": False}:
            raise ValueError("type payload must reproduce the saved old token")
    else:
        expected_direction = "up" if cue in {"OVERVIEW", "PLANNING"} else "down"
        if next_action["arguments"].get("direction") != expected_direction:
            raise ValueError("scroll direction must agree with the saved old waypoint")

    trials = row["offline_action_trials"]
    if [trial["condition"] for trial in trials] != ["ordered_history", "current_only_control"]:
        raise ValueError("offline trials must include ordered-history and current-only arms")
    if any(trial["expected_next_action_id"] != next_action["action_id"] for trial in trials):
        raise ValueError("offline trials must share the same current next-action label")


def _manifest_owned_by_generator(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            first = next(line for line in handle if line.strip())
        row = json.loads(first)
    except (OSError, StopIteration, json.JSONDecodeError):
        return False
    return (
        row.get("dataset") == DATASET_NAME
        and row.get("dataset_revision") == GENERATOR_VERSION
        and row.get("probe_kind") == PROBE_KIND
    )


def _meta_owned_by_generator(path: Path) -> bool:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        meta.get("generator_version") == GENERATOR_VERSION
        and meta.get("probe_kind") == PROBE_KIND
    )


def _check_output_safety(
    image_paths: Iterable[Path],
    manifest_path: Path,
    meta_path: Path,
    *,
    overwrite: bool,
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
    """Generate controlled offline-action images, JSONL, and metadata."""

    image_dir = Path(image_dir)
    manifest_path = Path(manifest_path)
    meta_path = Path(meta_path) if meta_path is not None else manifest_path.with_suffix(".meta.json")
    if n_episodes <= 0 or n_episodes % len(ACTION_TYPES) != 0:
        raise ValueError("n_episodes must be a positive multiple of three for balanced action types")
    if manifest_path == meta_path:
        raise ValueError("manifest_path and meta_path must differ")
    prefix = image_prefix.rstrip("/") if image_prefix else _default_image_prefix(image_dir)

    expected_filenames = [
        f"action_proxy_{index:04d}_{role}.png"
        for index in range(n_episodes)
        for role in ("old", "current")
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
            payloads[filename] = buffer.getvalue()
        for observation in row["observations"]:
            filename = Path(observation["image"]).name
            observation["image_sha256"] = hashlib.sha256(payloads[filename]).hexdigest()
            observation["source_image_bytes"] = len(payloads[filename])
        validate_manifest_row(row)
        rows.append(row)

    if set(payloads) != set(expected_filenames):
        raise RuntimeError("generated image names do not match the planned output set")
    for filename in expected_filenames:
        _atomic_write(image_dir / filename, payloads[filename])

    manifest_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    action_type_counts = Counter(row["next_action"]["action_type"] for row in rows)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset": DATASET_NAME,
        "domain": DOMAIN,
        "synthetic": True,
        "probe_kind": PROBE_KIND,
        "claim_scope": CLAIM_SCOPE,
        "real_trajectory_claim_allowed": False,
        "interactive_environment": False,
        "seed": seed,
        "n_episodes": n_episodes,
        "n_images": n_episodes * 2,
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "action_type_counts": {
            action_type: action_type_counts[action_type] for action_type in ACTION_TYPES
        },
        "history_pattern": ["old_observation", "executed_click", "current_observation"],
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
        f"wrote {len(rows)} controlled D5 offline-action episodes and "
        f"{len(rows) * 2} images -> {args.manifest} (meta: {meta_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
