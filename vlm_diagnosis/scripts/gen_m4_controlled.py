"""Generate a deterministic synthetic M4 controlled-mechanism probe.

Each generated 896 x 896 mobile-UI composite carries the same six primary
information types: OCR, visual state, layout, coordinate grounding, non-text
icon/color, and true visual count.  This is deliberately a synthetic mechanism
probe; it is not a real-world benchmark and must not be used by itself to make
novelty, generality, or deployment claims.

The command refuses to replace outputs by default.  ``--overwrite`` is accepted
only when an existing manifest/meta file identifies this generator, so unrelated
files with coincidentally matching names are not overwritten.

Examples::

    python -m vlm_diagnosis.scripts.gen_m4_controlled
    python -m vlm_diagnosis.scripts.gen_m4_controlled --n-images 4 --seed 7 \
        --image-dir /tmp/m4/images --manifest /tmp/m4/manifest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = ROOT / "data" / "m4_controlled"
DEFAULT_MANIFEST = ROOT / "experiments" / "manifests" / "m4_controlled.jsonl"
DEFAULT_N_IMAGES = 24
DEFAULT_SEED = 42
IMAGE_SIZE = 896
QUESTIONS_PER_IMAGE = 6
GENERATOR_VERSION = "m4-controlled-generator-v1"
SCHEMA_VERSION = "m4-controlled-schema-v1"
DATASET_NAME = "synthetic_m4_controlled"
DOMAIN = "synthetic_mobile_ui"
PROBE_KIND = "controlled_mechanism_probe"
CLAIM_SCOPE = "synthetic mechanism probe only; no real-world or novelty claims"

TASK_TYPES = ("ocr", "semantic", "layout", "grounding", "icon", "count")
TASK_SUBTYPES = {
    "ocr": "exact_reference_code",
    "semantic": "visual_state",
    "layout": "relative_card_position",
    "grounding": "coordinate_click",
    "icon": "non_text_icon_color",
    "count": "true_visual_count",
}

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

COLORS = {
    "red": (210, 61, 70),
    "blue": (51, 104, 214),
    "green": (42, 153, 90),
    "orange": (224, 126, 45),
    "purple": (133, 77, 190),
    "teal": (35, 145, 151),
}
SHAPES = ("circle", "triangle", "diamond", "star")
CARD_LABELS = ("FILES", "MAIL", "NOTES", "TASKS", "EVENTS", "PHOTOS")
ACTION_LABELS = ("APPLY", "SAVE", "SHARE", "UPLOAD", "REVIEW", "ARCHIVE")
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
NUMBER_WORDS = {
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Load the bundled system font, with a portable Pillow fallback."""

    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _rng_for(seed: int, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:m4-controlled:{index}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    anchor: str = "la",
    pad: int = 4,
) -> list[int]:
    raw = draw.textbbox(xy, text, font=font, anchor=anchor)
    return [raw[0] - pad, raw[1] - pad, raw[2] + pad, raw[3] + pad]


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int],
) -> list[int]:
    cx, cy = center
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    if shape == "circle":
        draw.ellipse(bbox, fill=fill)
    elif shape == "triangle":
        draw.polygon(
            [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)],
            fill=fill,
        )
    elif shape == "diamond":
        draw.polygon(
            [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)],
            fill=fill,
        )
    elif shape == "star":
        points = []
        for point_index in range(10):
            angle = -math.pi / 2 + point_index * math.pi / 5
            point_radius = radius if point_index % 2 == 0 else radius * 0.43
            points.append(
                (cx + point_radius * math.cos(angle), cy + point_radius * math.sin(angle))
            )
        draw.polygon(points, fill=fill)
    else:  # pragma: no cover - guarded by constants and schema tests
        raise ValueError(f"unknown shape: {shape}")
    return bbox


def _flags(
    *,
    text: bool = False,
    spatial: bool = False,
    icon: bool = False,
    count: bool = False,
    state: bool = False,
    color: bool = False,
) -> dict[str, bool]:
    return {
        "requires_text": text,
        "requires_spatial": spatial,
        "requires_icon": icon,
        "requires_count": count,
        "requires_state": state,
        "requires_color": color,
    }


def _question(
    sample_id: str,
    task_type: str,
    prompt: str,
    answer_type: str,
    acceptable_answers: list[str],
    evidence_bboxes: list[list[int]],
    *,
    target_bbox: list[int] | None = None,
    flags: dict[str, bool],
) -> dict[str, Any]:
    return {
        "question_id": f"{sample_id}_{task_type}",
        "task_type": task_type,
        "task_subtype": TASK_SUBTYPES[task_type],
        "question": prompt,
        "answer_type": answer_type,
        "acceptable_answers": acceptable_answers,
        "target_bbox": target_bbox,
        "evidence_bboxes": evidence_bboxes,
        **flags,
    }


def _draw_section_card(draw: ImageDraw.ImageDraw, y: int, title: str) -> None:
    draw.rounded_rectangle([48, y, 848, y + 112], radius=24, fill=(247, 249, 252),
                           outline=(221, 226, 234), width=2)
    draw.text((72, y + 18), title, font=_font(17, bold=True), fill=(83, 92, 108),
              anchor="la")


def build_sample(index: int, seed: int, image_path: str) -> tuple[Image.Image, dict[str, Any]]:
    """Build one image and its six-question manifest row."""

    if index < 0:
        raise ValueError("index must be non-negative")
    rng = _rng_for(seed, index)
    sample_id = f"m4_controlled_{index:04d}"

    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (238, 242, 247))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([24, 18, 872, 878], radius=38, fill=(255, 255, 255),
                           outline=(204, 211, 222), width=3)

    # Header and OCR-only reference code.
    accent_name = rng.choice(tuple(COLORS))
    accent = COLORS[accent_name]
    draw.rounded_rectangle([48, 42, 848, 150], radius=26, fill=(245, 247, 251))
    draw.text((76, 77), "WORKSPACE", font=_font(29, bold=True), fill=(34, 42, 57), anchor="la")
    code = "".join(rng.choices(CODE_ALPHABET, k=7))
    code_x = rng.choice((568, 592, 616))
    draw.rounded_rectangle([code_x - 24, 66, 822, 129], radius=16, fill=accent)
    code_font = _font(31, bold=True)
    draw.text((code_x, 98), code, font=code_font, fill=(255, 255, 255), anchor="lm")
    code_bbox = _text_bbox(draw, (code_x, 98), code, code_font, anchor="lm", pad=5)

    section_names = ["state", "layout", "icons", "count"]
    rng.shuffle(section_names)
    section_y = dict(zip(section_names, (174, 310, 446, 582)))

    # Visual-state section: the answer is encoded only by switch appearance.
    state_y = section_y["state"]
    _draw_section_card(draw, state_y, "SYNC")
    switch_on = bool(rng.getrandbits(1))
    switch_left = rng.choice((604, 678))
    switch_bbox = [switch_left, state_y + 43, switch_left + 112, state_y + 91]
    switch_fill = COLORS["green"] if switch_on else (155, 164, 178)
    draw.rounded_rectangle(switch_bbox, radius=24, fill=switch_fill)
    knob_x = switch_bbox[2] - 24 if switch_on else switch_bbox[0] + 24
    draw.ellipse([knob_x - 18, switch_bbox[1] + 6, knob_x + 18, switch_bbox[3] - 6],
                 fill=(255, 255, 255), outline=(225, 228, 233), width=1)

    # Relative-layout section: two independently identifiable text cards swap sides.
    layout_y = section_y["layout"]
    _draw_section_card(draw, layout_y, "SHORTCUTS")
    first_label, second_label = rng.sample(CARD_LABELS, 2)
    left_label, right_label = ([first_label, second_label] if rng.random() < 0.5
                               else [second_label, first_label])
    card_boxes = {
        left_label: [150, layout_y + 40, 410, layout_y + 94],
        right_label: [486, layout_y + 40, 746, layout_y + 94],
    }
    for label, bbox in card_boxes.items():
        draw.rounded_rectangle(bbox, radius=14, fill=(255, 255, 255),
                               outline=(177, 187, 202), width=2)
        draw.text(((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2), label,
                  font=_font(22, bold=True), fill=(51, 61, 78), anchor="mm")
    first_side = "left" if left_label == first_label else "right"

    # Non-text icon/color section. Shapes and colors are unique within the panel.
    icon_y = section_y["icons"]
    _draw_section_card(draw, icon_y, "ICONS")
    shapes = list(SHAPES)
    rng.shuffle(shapes)
    color_names = rng.sample(list(COLORS), len(shapes))
    centers = [(230, icon_y + 72), (380, icon_y + 72), (530, icon_y + 72),
               (680, icon_y + 72)]
    icon_bboxes: dict[str, list[int]] = {}
    icon_colors: dict[str, str] = {}
    for shape, color_name, center in zip(shapes, color_names, centers):
        icon_bboxes[shape] = _draw_shape(draw, shape, center, 27, COLORS[color_name])
        icon_colors[shape] = color_name
    target_shape = rng.choice(shapes)

    # True visual count section. No numeral or number word is rendered in the image.
    count_y = section_y["count"]
    _draw_section_card(draw, count_y, "ACTIVITY")
    dot_count = rng.randint(3, 8)
    dot_color_name = rng.choice(tuple(COLORS))
    dot_color = COLORS[dot_color_name]
    spacing = 58
    first_dot_x = IMAGE_SIZE // 2 - (dot_count - 1) * spacing // 2
    dot_bboxes = []
    for dot_index in range(dot_count):
        cx = first_dot_x + dot_index * spacing
        cy = count_y + 73
        dot_bbox = [cx - 16, cy - 16, cx + 16, cy + 16]
        draw.ellipse(dot_bbox, fill=dot_color)
        dot_bboxes.append(dot_bbox)

    # Two-button footer supplies a nontrivial coordinate grounding target.
    action_labels = rng.sample(ACTION_LABELS, 2)
    action_order = action_labels if rng.random() < 0.5 else list(reversed(action_labels))
    action_boxes = {
        action_order[0]: [142, 742, 410, 816],
        action_order[1]: [486, 742, 754, 816],
    }
    target_action = rng.choice(action_labels)
    for label, bbox in action_boxes.items():
        draw.rounded_rectangle(
            bbox,
            radius=20,
            # Equal styling prevents the target from being inferred without
            # reading the requested label.
            fill=accent,
            outline=accent,
            width=3,
        )
        draw.text(((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2), label,
                  font=_font(24, bold=True), fill=(255, 255, 255),
                  anchor="mm")

    questions = [
        _question(
            sample_id,
            "ocr",
            "What exact reference code is displayed in the header? Answer with the code only.",
            "string",
            [code],
            [code_bbox],
            flags=_flags(text=True),
        ),
        _question(
            sample_id,
            "semantic",
            "What is the visual state of the SYNC switch: on or off?",
            "categorical",
            ["on" if switch_on else "off"],
            [switch_bbox],
            flags=_flags(state=True),
        ),
        _question(
            sample_id,
            "layout",
            f"Is the {first_label} card to the left or right of the {second_label} card?",
            "categorical",
            [first_side],
            [card_boxes[first_label], card_boxes[second_label]],
            flags=_flags(text=True, spatial=True),
        ),
        _question(
            sample_id,
            "grounding",
            f"Where is the center of the button labeled {target_action} in this "
            f"{IMAGE_SIZE}x{IMAGE_SIZE} image? Respond with only the coordinates "
            "in the form (x, y).",
            "coordinate",
            [],
            [action_boxes[target_action]],
            target_bbox=action_boxes[target_action],
            flags=_flags(text=True, spatial=True),
        ),
        _question(
            sample_id,
            "icon",
            f"What color is the {target_shape} icon in the ICONS panel? Answer with one word.",
            "categorical",
            [icon_colors[target_shape]],
            [icon_bboxes[target_shape]],
            flags=_flags(icon=True, color=True),
        ),
        _question(
            sample_id,
            "count",
            f"How many {dot_color_name} dots are shown in the ACTIVITY panel?",
            "integer",
            [str(dot_count), NUMBER_WORDS[dot_count]],
            dot_bboxes,
            flags=_flags(count=True, color=True),
        ),
    ]

    row = {
        "dataset": DATASET_NAME,
        "dataset_revision": GENERATOR_VERSION,
        "split": "mechanism_probe",
        "sample_id": sample_id,
        "image": image_path,
        "image_width": IMAGE_SIZE,
        "image_height": IMAGE_SIZE,
        "domain": DOMAIN,
        "synthetic": True,
        "probe_kind": PROBE_KIND,
        "claim_scope": CLAIM_SCOPE,
        "questions": questions,
        "selection_seed": seed,
        "generation_index": index,
        "gen_meta": {
            "section_order_top_to_bottom": section_names,
            "accent_color": accent_name,
            "dot_color": dot_color_name,
        },
    }
    validate_manifest_row(row)
    return image, row


def _validate_bbox(bbox: Any, field_name: str) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{field_name} must be a four-number list")
    if any(not isinstance(value, int) for value in bbox):
        raise ValueError(f"{field_name} coordinates must be integers")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= IMAGE_SIZE and 0 <= y1 < y2 <= IMAGE_SIZE):
        raise ValueError(f"{field_name} is outside the {IMAGE_SIZE}x{IMAGE_SIZE} image: {bbox}")


def validate_manifest_row(row: dict[str, Any]) -> None:
    """Validate the controlled-probe invariants for one image-level row."""

    required_row_keys = {
        "dataset", "sample_id", "image", "image_width", "image_height", "domain",
        "synthetic", "probe_kind", "claim_scope", "questions", "selection_seed",
    }
    missing = required_row_keys - row.keys()
    if missing:
        raise ValueError(f"manifest row is missing fields: {sorted(missing)}")
    if row["dataset"] != DATASET_NAME or row["domain"] != DOMAIN:
        raise ValueError("row does not identify the controlled synthetic M4 dataset/domain")
    if row["synthetic"] is not True or row["probe_kind"] != PROBE_KIND:
        raise ValueError("row must be explicitly labeled as a synthetic mechanism probe")
    if row["image_width"] != IMAGE_SIZE or row["image_height"] != IMAGE_SIZE:
        raise ValueError(f"controlled images must be {IMAGE_SIZE}x{IMAGE_SIZE}")

    questions = row["questions"]
    if not isinstance(questions, list) or len(questions) != QUESTIONS_PER_IMAGE:
        raise ValueError(f"each image must carry exactly {QUESTIONS_PER_IMAGE} questions")
    task_types = [question.get("task_type") for question in questions]
    if len(set(task_types)) != QUESTIONS_PER_IMAGE or set(task_types) != set(TASK_TYPES):
        raise ValueError(f"each image must carry exactly one of each task type: {TASK_TYPES}")

    required_question_keys = {
        "question_id", "task_type", "task_subtype", "question", "answer_type",
        "acceptable_answers", "target_bbox", "evidence_bboxes", "requires_text",
        "requires_spatial", "requires_icon", "requires_count", "requires_state",
        "requires_color",
    }
    for question in questions:
        missing_question = required_question_keys - question.keys()
        if missing_question:
            raise ValueError(
                f"question {question.get('question_id')} is missing fields: "
                f"{sorted(missing_question)}"
            )
        task_type = question["task_type"]
        if question["task_subtype"] != TASK_SUBTYPES[task_type]:
            raise ValueError(f"unexpected subtype for {task_type}")
        if not isinstance(question["question"], str) or not question["question"].strip():
            raise ValueError("question text must be non-empty")
        if not isinstance(question["acceptable_answers"], list):
            raise ValueError("acceptable_answers must be a list")
        evidence = question["evidence_bboxes"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("evidence_bboxes must be a non-empty list")
        for evidence_index, bbox in enumerate(evidence):
            _validate_bbox(bbox, f"evidence_bboxes[{evidence_index}]")
        for flag in (
            "requires_text", "requires_spatial", "requires_icon", "requires_count",
            "requires_state", "requires_color",
        ):
            if not isinstance(question[flag], bool):
                raise ValueError(f"{flag} must be boolean")
        if task_type == "grounding":
            _validate_bbox(question["target_bbox"], "target_bbox")
            if question["answer_type"] != "coordinate":
                raise ValueError("grounding question must use coordinate answer_type")
        elif question["target_bbox"] is not None:
            raise ValueError("only grounding questions may carry target_bbox")
        elif not question["acceptable_answers"]:
            raise ValueError("non-grounding questions need at least one acceptable answer")


def _manifest_owned_by_generator(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            first_nonempty = next(line for line in handle if line.strip())
        row = json.loads(first_nonempty)
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
    image_paths = list(image_paths)
    existing_images = [path for path in image_paths if path.exists()]
    existing_control = [path for path in (manifest_path, meta_path) if path.exists()]
    if not overwrite:
        existing = existing_control + existing_images
        if existing:
            shown = ", ".join(str(path) for path in existing[:3])
            suffix = " ..." if len(existing) > 3 else ""
            raise FileExistsError(f"refusing to overwrite existing output(s): {shown}{suffix}")
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
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _default_image_prefix(image_dir: Path) -> str:
    resolved = image_dir.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def generate_dataset(
    image_dir: Path,
    manifest_path: Path,
    *,
    meta_path: Path | None = None,
    image_prefix: str | None = None,
    n_images: int = DEFAULT_N_IMAGES,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Generate images, JSONL manifest, and deterministic metadata."""

    image_dir = Path(image_dir)
    manifest_path = Path(manifest_path)
    meta_path = Path(meta_path) if meta_path is not None else manifest_path.with_suffix(".meta.json")
    if n_images <= 0:
        raise ValueError("n_images must be positive")
    if manifest_path == meta_path:
        raise ValueError("manifest_path and meta_path must be different")
    prefix = image_prefix.rstrip("/") if image_prefix else _default_image_prefix(image_dir)

    filenames = [f"m4_controlled_{index:04d}.png" for index in range(n_images)]
    image_paths = [image_dir / filename for filename in filenames]
    _check_output_safety(image_paths, manifest_path, meta_path, overwrite=overwrite)

    rows: list[dict[str, Any]] = []
    encoded_images: list[bytes] = []
    for index, filename in enumerate(filenames):
        manifest_image_path = f"{prefix}/{filename}" if prefix else filename
        image, row = build_sample(index, seed, manifest_image_path)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=9)
        payload = buffer.getvalue()
        row["image_sha256"] = hashlib.sha256(payload).hexdigest()
        validate_manifest_row(row)
        rows.append(row)
        encoded_images.append(payload)

    for path, payload in zip(image_paths, encoded_images):
        _atomic_write(path, payload)

    manifest_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset": DATASET_NAME,
        "domain": DOMAIN,
        "synthetic": True,
        "probe_kind": PROBE_KIND,
        "claim_scope": CLAIM_SCOPE,
        "seed": seed,
        "n_images": n_images,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "questions_per_image": QUESTIONS_PER_IMAGE,
        "task_types": list(TASK_TYPES),
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
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="metadata path (default: MANIFEST with .meta.json suffix)",
    )
    parser.add_argument(
        "--image-prefix",
        default=None,
        help="path stored before each image filename in the manifest",
    )
    parser.add_argument("--n-images", type=int, default=DEFAULT_N_IMAGES)
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
        n_images=args.n_images,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    meta_path = args.meta or args.manifest.with_suffix(".meta.json")
    print(
        f"wrote {len(rows)} synthetic mechanism-probe images, "
        f"{len(rows) * QUESTIONS_PER_IMAGE} questions -> {args.manifest} "
        f"(meta: {meta_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
