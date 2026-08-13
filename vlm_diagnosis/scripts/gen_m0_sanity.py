"""M0-02 — 합성 비텍스트 sanity 세트 생성기.

896×896(=32×28, processor 무리사이즈) 흰 배경 이미지에 4개 task type을 그린다.

  ocr        : 무작위 6자 대문자 코드           → EM
  icon       : 색이 다른 도형 3개, 색 질문      → EM (색 단어)
  layout     : 두 단어의 좌/우 (또는 상/하) 배치 → EM (left/right/top/bottom)
  grounding  : 단색 원 하나의 중심 좌표          → click-in-bbox

각 type N개(기본 10)를 seed 고정으로 생성하고 manifest + .meta.json을 쓴다.

  python -m vlm_diagnosis.scripts.gen_m0_sanity --n-per-task 10 --seed 42
"""
import argparse
import json
import os
import random
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
IMG_DIR = os.path.join(ROOT, "data", "m0_sanity")
MANIFEST = os.path.join(ROOT, "experiments", "manifests", "m0_sanity.jsonl")
GENERATOR_VERSION = "m0-sanity-gen-v1"

SIZE = 896  # 32*28 — Qwen2.5-VL processor가 리사이즈하지 않는 크기
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLORS = {"red": (220, 30, 30), "green": (20, 150, 40), "blue": (30, 60, 220),
          "orange": (240, 140, 20), "purple": (140, 40, 180)}
SHAPES = ("circle", "square", "triangle")
WORDS = ("BANK", "MAIL", "SHOP", "PARK", "GATE", "DESK", "LAMP", "ROAD")


def _grid_cells(rng, n, margin=120):
    """겹치지 않는 배치 지점 n개 (3×3 그리드에서 무작위 선택)."""
    step = (SIZE - 2 * margin) // 2
    cells = [(margin + c * step, margin + r * step) for r in range(3) for c in range(3)]
    rng.shuffle(cells)
    return cells[:n]


def draw_shape(d, shape, color, cx, cy, r):
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)


def gen_ocr(rng, i):
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)
    code = "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))
    (cx, cy) = _grid_cells(rng, 1, margin=200)[0]
    d.text((cx, cy), code, fill="black", font=ImageFont.truetype(FONT_BOLD, 90),
           anchor="mm")
    q = "What is the code written in the image? Answer with the code only."
    return img, q, [code], None, {"code": code}


def gen_icon(rng, i):
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)
    shapes = rng.sample(SHAPES, 3)
    colors = rng.sample(list(COLORS), 3)
    cells = _grid_cells(rng, 3)
    for s, c, (cx, cy) in zip(shapes, colors, cells):
        draw_shape(d, s, COLORS[c], cx, cy, 70)
    target = rng.randrange(3)
    q = f"What color is the {shapes[target]}? Answer with one word."
    return img, q, [colors[target]], None, {"shapes": shapes, "colors": colors}


def gen_layout(rng, i):
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)
    w1, w2 = rng.sample(WORDS, 2)
    font = ImageFont.truetype(FONT_BOLD, 72)
    horizontal = rng.random() < 0.5
    if horizontal:
        d.text((200, SIZE // 2), w1, fill="black", font=font, anchor="mm")
        d.text((SIZE - 200, SIZE // 2), w2, fill="black", font=font, anchor="mm")
        sides = {w1: "left", w2: "right"}
        opts = "left or right"
    else:
        d.text((SIZE // 2, 200), w1, fill="black", font=font, anchor="mm")
        d.text((SIZE // 2, SIZE - 200), w2, fill="black", font=font, anchor="mm")
        sides = {w1: "top", w2: "bottom"}
        opts = "top or bottom"
    target = rng.choice([w1, w2])
    q = (f"Is the word {target} on the {opts} side of the image? "
         f"Answer with one word.")
    return img, q, [sides[target]], None, {"words": [w1, w2], "horizontal": horizontal}


def gen_grounding(rng, i):
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)
    color = rng.choice(list(COLORS))
    (cx, cy) = _grid_cells(rng, 1)[0]
    r = 55
    draw_shape(d, "circle", COLORS[color], cx, cy, r)
    q = (f"Where is the center of the {color} circle in this {SIZE}x{SIZE} "
         f"image? Respond with only the coordinates in the form (x, y).")
    bbox = [cx - 2 * r, cy - 2 * r, cx + 2 * r, cy + 2 * r]  # 관대한 히트박스
    return img, q, None, bbox, {"center": [cx, cy], "radius": r, "color": color}


GENERATORS = {"ocr": gen_ocr, "icon": gen_icon, "layout": gen_layout,
              "grounding": gen_grounding}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-task", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    rows = []
    for task, gen in GENERATORS.items():
        for i in range(a.n_per_task):
            rng = random.Random(f"{a.seed}:{task}:{i}")
            img, q, answers, bbox, meta = gen(rng, i)
            sid = f"{task}_{i:02d}"
            rel = os.path.join("data", "m0_sanity", f"{sid}.png")
            img.save(os.path.join(ROOT, rel))
            rows.append({
                "dataset": "synthetic_m0", "dataset_revision": GENERATOR_VERSION,
                "split": "smoke", "sample_id": sid, "image": rel,
                "task_type": task, "question": q,
                "acceptable_answers": answers, "target_bbox": bbox,
                "selection_seed": a.seed, "gen_meta": meta,
            })
    with open(MANIFEST, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(MANIFEST.replace(".jsonl", ".meta.json"), "w") as f:
        json.dump({"schema_version": "1.1", "generator": GENERATOR_VERSION,
                   "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "selection_seed": a.seed, "n_per_task": a.n_per_task,
                   "image_size": SIZE, "task_types": list(GENERATORS)}, f, indent=1)
    print(f"wrote {len(rows)} samples → {MANIFEST}")


if __name__ == "__main__":
    main()
