"""아이디어 3 완성형 검증 — 과거 답을 초안으로, 보관 KV 위 병렬 검증 (spec-dec식).

배치 시나리오(UNION 부분집합)에서, 평가 질문 qB에 대해 소스 질문 qA들의 기록된
답(FULL 이미지에서 생성된 것 = base_accuracy 예측)을 초안으로 넣고:
  [이미지(보관 조각만 보임) + qB + 초안토큰들] forward 1회
  → 각 초안 토큰 위치의 greedy 예측이 초안과 전부 일치하면 "수락"
  → 수락 시 반환되는 답 = 초안 (2단 생성 경로와 greedy 동일성 보증)
측정: 수락률과 수락 시 정답률(EM)을 쌍 라벨별로.

  python -m vlm_diagnosis.exps.draft_verify_probe \
    --manifest experiments/manifests/gqa_transfer.jsonl \
    --base results/smoke/gqa_base.shard*.jsonl --device cuda:2 --limit 3
"""
import argparse
import glob
import json
import os
import time
import zlib

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import (mrope_position_ids, causal_mask_4d,
                                            evict_columns)
from vlm_diagnosis.core.kv_baselines import KVShape, dense_storage, max_keep_for_budget
from vlm_diagnosis.core.metrics import exact_match
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."


@torch.no_grad()
def verify_draft(model, processor, img, question, draft_text, keep, n_vis, device):
    """보관 조각 위 [질문+초안] 병렬 검증. 반환: (수락 여부, 최소 margin, 일치 수)."""
    tok = processor.tokenizer
    ins = S.vlm_inputs(processor, img, question + BRIEF, device)
    d_ids = tok(draft_text, add_special_tokens=False,
                return_tensors="pt").input_ids.to(device)
    if d_ids.shape[1] == 0:
        return False, 0.0, 0, 0
    full = torch.cat([ins["input_ids"], d_ids], 1)
    sp = token_spans(full, model.config)
    vis, vis_end, L = sp["visual"], sp["vis_end"], sp["L"]
    ev = torch.tensor([int(vis[o]) for o in range(n_vis) if o not in keep],
                      device=device)
    attn2d = torch.ones(1, L, dtype=torch.long, device=device)
    pos = mrope_position_ids(model, full, ins["image_grid_thw"], attn2d)
    m4 = causal_mask_4d(L, device, torch.float16)
    if ev.numel():
        m4 = evict_columns(m4, ev, row_start=vis_end + 1)
    out = model(input_ids=full, attention_mask=m4, position_ids=pos,
                pixel_values=ins["pixel_values"],
                image_grid_thw=ins["image_grid_thw"], use_cache=False)
    n_d = d_ids.shape[1]
    lg = out.logits[0, L - n_d - 1:L - 1].float()      # 각 초안 토큰 예측 위치
    pred = lg.argmax(-1)
    match = (pred == d_ids[0]).all().item()
    top2 = torch.topk(lg, 2, dim=-1).values
    margin = float((top2[:, 0] - top2[:, 1]).min())
    n_match = int((pred == d_ids[0]).sum())
    return bool(match), margin, n_match, n_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True, help="base_accuracy 예측 glob")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/smoke/draft_verify.jsonl")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    base_pred = {}
    for f in glob.glob(a.base):
        for l in open(f):
            r = json.loads(l)
            base_pred[r["question_id"]] = r["prediction"]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    if a.limit:
        rows = rows[:a.limit]
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if a.resume and os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(str(json.loads(line)["sample_id"]))
            except Exception:
                pass
        print(f"[resume] {len(done)}장 건너뜀", flush=True)
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    NL, NKV, HD = kv_dims(model)

    with open(out_path, "a" if a.resume else "w") as f:
        for di, row in enumerate(rows):
            if str(row["sample_id"]) in done:
                continue
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs_src = row["questions"][1:4]
            qs_eval = row["questions"][4:6]       # 배치 시나리오 = held-out 질문
            if len(qs_src) < 2 or not qs_eval:
                continue
            s1_src = [S.score_s1(model, processor, img, q["question"] + BRIEF,
                                 a.device).cpu() for q in qs_src]
            n_vis = s1_src[0].shape[0]
            shape = KVShape(layers=NL, batch=1, kv_heads=NKV, tokens=n_vis,
                            head_dim=HD)
            e = dense_storage(shape)
            fb = e.payload_bytes + e.metadata_bytes + e.position_bytes
            k = max_keep_for_budget(shape, int(a.budget * fb), "sparse")
            union = set().union(*[set(torch.topk(s, min(k, n_vis)).indices.tolist())
                                  for s in s1_src])
            for qB in qs_eval:
                for qA in qs_src:
                    draft = base_pred.get(qA["question_id"])
                    if not draft:
                        continue
                    acc, margin, n_match, n_d = verify_draft(
                        model, processor, img, qB["question"], draft, union,
                        n_vis, a.device)
                    f.write(json.dumps({
                        "sample_id": row["sample_id"],
                        "qA_id": qA["question_id"], "qB_id": qB["question_id"],
                        "draft": draft, "gold_B": qB["answers"],
                        "accepted": acc, "min_margin": round(margin, 4),
                        "n_match": n_match, "n_draft_tokens": n_d,
                        "draft_em_vs_goldB": exact_match(draft, qB["answers"]),
                        "budget": a.budget, "keep_tokens": len(union)},
                        ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} {time.time()-t0:.0f}s",
                  flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
