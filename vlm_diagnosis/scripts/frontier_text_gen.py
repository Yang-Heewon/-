"""Frontier 최소판 준비 1 — 각 이미지의 텍스트 표현(짙은 캡션 + 전사) 생성.

표현 비교 실험에서 '텍스트로 저장' 팔의 재료. 이미지당:
  caption : 상세 서술 (레이아웃·객체·관계 포함)
  ocr     : 보이는 모든 글자의 전사 (GUI에서 핵심)
byte 크기와 함께 기록 — 이후 byte-매칭 예산 비교에 사용.

  python -m vlm_diagnosis.scripts.frontier_text_gen \
    --manifest experiments/manifests/screenqa_discovery.jsonl --device cuda:0
"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_vlm
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
PROMPTS = {
    "caption": "Describe this image in detail: layout, every visible element, "
               "objects, their attributes and relations.",
    "ocr": "Transcribe ALL text visible in this image exactly, "
           "preserving structure. If none, say 'no text'.",
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--out", default="results/discovery/frontier_text.jsonl")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    rows = rows[a.shard::a.nshards]
    if a.nshards > 1:
        a.out = a.out.replace(".jsonl", f".shard{a.shard}.jsonl")
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if a.resume and os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(str(json.loads(line)["sample_id"]))
            except Exception:
                pass
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    tok = processor.tokenizer

    with open(out_path, "a" if a.resume else "w") as f:
        for di, row in enumerate(rows):
            if str(row["sample_id"]) in done:
                continue
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            rec = {"sample_id": row["sample_id"], "image": row["image"]}
            for name, prompt in PROMPTS.items():
                ins = S.vlm_inputs(processor, img, prompt, a.device)
                gen = model.generate(**ins, max_new_tokens=a.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                text = tok.decode(gen[0, ins["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
                rec[name] = text
                rec[f"{name}_bytes"] = len(text.encode("utf-8"))
                rec[f"{name}_tokens"] = int(gen.shape[1] - ins["input_ids"].shape[1])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} "
                  f"cap={rec['caption_bytes']}B ocr={rec['ocr_bytes']}B "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
