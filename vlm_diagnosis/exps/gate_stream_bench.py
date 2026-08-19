"""E2E stream benchmark — GPU read stage (preregistered: docs/E2E-STREAM-PREREG.md).

Consumes the policy simulator's read manifest
(results/discovery/gates/stream/read_manifest.jsonl): for each
(question x policy x budget) row, injects the policy's stored memory items
(images and/or text notes, each with capture metadata) and asks
Qwen2.5-VL-7B the question. Scores exact match vs the latest-value gold and
vs the stale (superseded) values for the H2 stale-answer rate.

Usage:
  python -m vlm_diagnosis.exps.gate_stream_bench \
      --device cuda:0 --shard 0 --nshards 2 \
      --out results/discovery/gates/stream_bench.shard0.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

import vlm_diagnosis.core.byte_codecs  # noqa: F401  (registers AVIF with PIL)
from vlm_diagnosis.core.loader import load_vlm
from vlm_diagnosis.core.metrics import anls, exact_match

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "results/discovery/gates/stream/read_manifest.jsonl"

INSTRUCTION = (
    "These are stored visual-memory records of pages observed over time. "
    "Each record's caption states when it was captured. Use the newest "
    "applicable evidence; an UPDATE PATCH supersedes the overlapping region "
    "of the snapshot it applies to. "
)


def build_messages(row: dict[str, Any]):
    content: list[dict[str, str]] = []
    images: list[Image.Image] = []
    for i, item in enumerate(row["memory_items"], 1):
        meta = item.get("meta", "")
        if item["kind"] == "image":
            content.append({"type": "text", "text": f"Memory item {i}: {meta}\n"})
            content.append({"type": "image"})
            images.append(Image.open(item["path"]).convert("RGB"))
        else:
            content.append({
                "type": "text",
                "text": f"Memory note {i}: {meta}\n{item['text']}\n",
            })
    lead = INSTRUCTION if row["memory_items"] else "No stored memory was retrieved. "
    content.append({"type": "text", "text": lead + row["question_text"]})
    return [{"role": "user", "content": content}], images


@torch.inference_mode()
def answer(model, processor, row, device, max_new_tokens=16) -> str:
    messages, images = build_messages(row)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--max-pixels", type=int, default=1003520,
        help="processor max_pixels cap per image; V100 eager attention OOMs "
             "on 4x native-resolution (2.05MP) screenshots, so full shots are "
             "downscaled to ~1MP (thumbnails/patches are below the cap already)",
    )
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    if args.limit:
        rows = rows[: args.limit]

    done: set[tuple[str, str, int]] = set()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["question_id"], r["policy"], r["budget"]))
    print(f"shard {args.shard}/{args.nshards}: {len(rows)} rows, {len(done)} done")

    model, processor = load_vlm(
        "qwen25vl", device=args.device, max_pixels=args.max_pixels
    )

    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as fh:
        for idx, row in enumerate(rows, 1):
            key = (row["question_id"], row["policy"], row["budget"])
            if key in done:
                continue
            pred = answer(model, processor, row, args.device)
            gold = [str(row["gold"])]
            stale = [str(s) for s in row.get("stale_values", [])]
            rec = {
                "record_type": "trial_result",
                "episode_id": row["episode_id"],
                "question_id": row["question_id"],
                "policy": row["policy"],
                "budget": row["budget"],
                "page": row.get("page"),
                "revised": row.get("revised"),
                "n_memory_items": len(row["memory_items"]),
                "prediction": pred,
                "em": exact_match(pred, gold),
                "anls": anls(pred, gold),
                "stale_em": exact_match(pred, stale) if stale else 0.0,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{idx}/{len(rows)}] {row['policy']}@{row['budget']} "
                  f"{row['question_id']} em={rec['em']} stale={rec['stale_em']}")


if __name__ == "__main__":
    main()
