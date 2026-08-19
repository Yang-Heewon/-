"""Experiment A — partial invalidation for the forget gate.

Question: when a screen changes, can the memory system keep the OLD stored
snapshot and add only a small CROP of the changed region from the new
observation (a "patch"), instead of re-storing the whole screen — without
losing the corrected answer?

Conditions (changed episodes of md_scaled, n=32):
  old_pkg          stored old snapshot only (AVIF@65536B package)  -> stale
  old_pkg+patch    old snapshot + newer crop of the changed row    -> target 1.0
  patch_only       the crop alone (sanity: is the value readable?)

The crop is the full field row (label + value pill) around the ground-truth
changed bbox, taken from the CURRENT screen (fresh observation) and encoded
as JPEG q85 — its byte size is the cost of the patch. Savings = 1 - patch
bytes / full re-store bytes (the episode's current AVIF package).

Usage:
  python -m vlm_diagnosis.exps.gate_partial_invalidation \
      --device cuda:0 --shard 0 --nshards 2 \
      --out results/discovery/gates/partial_invalidation.shard0.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

import vlm_diagnosis.core.byte_codecs  # noqa: F401  (registers AVIF plugin with PIL)
from vlm_diagnosis.core.loader import load_vlm
from vlm_diagnosis.core.metrics import anls, exact_match

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "experiments/manifests/md_scaled_read.jsonl"
PKG_DIR = REPO / "results/discovery/md_pkg"
DATA_DIR = REPO / "data/md_scaled"

INSTRUCTION = (
    "These are stored visual-memory snapshots. Read the project code, "
    "revision/timestamp, and requested field from the pixels. Use the newest "
    "applicable evidence when snapshots conflict. "
)


def find_package(episode: str, which: str) -> Path:
    pattern = str(PKG_DIR / f"memory_{episode}_{which}-*.avif.65536b.avif")
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern} -> {hits}")
    return Path(hits[0])


def make_patch(episode: str, bbox: list[int]) -> tuple[Image.Image, int]:
    """Crop the changed field's full row from the CURRENT screen and encode
    as JPEG q85. Returns (decoded patch image, patch bytes)."""
    cur = Image.open(DATA_DIR / f"{episode}_target_current.png").convert("RGB")
    x1, y1, x2, y2 = bbox
    row = (72, max(0, y1 - 36), 696, min(cur.height, y2 + 36))
    crop = cur.crop(row)
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    return Image.open(BytesIO(data)).convert("RGB"), len(data)


def build_messages(parts: list[tuple[str, Any]], question: str):
    content: list[dict[str, str]] = []
    for label, _img in parts:
        content.append({"type": "text", "text": label + "\n"})
        content.append({"type": "image"})
    text = INSTRUCTION if parts else "No visual-memory snapshot was retrieved. "
    content.append({"type": "text", "text": text + question + " Return only the requested value."})
    return [{"role": "user", "content": content}]


@torch.inference_mode()
def answer(model, processor, parts, question, device, max_new_tokens=16) -> str:
    messages = build_messages(parts, question)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
    if parts:
        kwargs["images"] = [img for _lab, img in parts]
    inputs = processor(**kwargs).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episodes = []
    for line in open(MANIFEST):
        r = json.loads(line)
        if r["factorial"]["state_changed"]:
            episodes.append(r)
    episodes = [r for i, r in enumerate(episodes) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(episodes)} changed episodes")

    model, processor = load_vlm("qwen25vl", device=args.device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, r in enumerate(episodes, 1):
            ep = r["episode_id"]
            q = r["question"]
            bbox = q["current_evidence"]["evidence_bbox"]
            old_pkg = find_package(ep, "target_old_r1")
            cur_pkg = find_package(ep, "target_current_r2")
            old_img = Image.open(old_pkg).convert("RGB")
            patch_img, patch_bytes = make_patch(ep, bbox)

            conds = {
                "old_pkg": [("Memory image 1 (stored snapshot):", old_img)],
                "old_pkg+patch": [
                    ("Memory image 1 (stored snapshot):", old_img),
                    ("Memory image 2 (newer partial patch of the same screen,"
                     " captured later; it supersedes the overlapping region of"
                     " Memory image 1):", patch_img),
                ],
                "patch_only": [
                    ("Memory image 1 (partial patch of a screen, latest state"
                     " of one field):", patch_img)
                ],
            }
            for cond, parts in conds.items():
                pred = answer(model, processor, parts, q["question"], args.device)
                acceptable = list(map(str, q["acceptable_answers"]))
                stale = list(map(str, q.get("stale_answers", [])))
                rec = {
                    "record_type": "trial_result",
                    "episode_id": ep,
                    "condition": cond,
                    "prediction": pred,
                    "current_em": exact_match(pred, acceptable),
                    "current_anls": anls(pred, acceptable),
                    "stale_em": exact_match(pred, stale) if stale else 0.0,
                    "patch_bytes": patch_bytes,
                    "old_pkg_bytes": old_pkg.stat().st_size,
                    "full_restore_bytes": cur_pkg.stat().st_size,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{idx}/{len(episodes)}] {ep} {cond} em={rec['current_em']}"
                      f" stale={rec['stale_em']} patch={patch_bytes}B")


if __name__ == "__main__":
    main()
