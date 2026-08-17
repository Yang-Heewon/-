"""끝-끝 캐스케이드 평가 — 3단 사다리를 실제 파이프라인으로 돌린다 (논문 §4).

각 held-out 질문에 대해:
  0) [질문 + 초안후보들] 병렬 검증 forward — 이 forward 하나에서
     ① 초안 greedy 일치 여부 ② 첫 답 토큰의 a4 margin 이 동시에 나온다.
  1단: 어느 초안이든 전일치 → 즉시 그 답 (보증: 2단과 동일 출력)
  2단: a4 ≥ τ → 보관 KV로 생성한 답
  3단: a4 < τ → 원본 이미지(무마스크)로 생성한 답
기록: 선택된 tier, 답, EM, 각 경로의 예상 비용(결과 19 실측 상수).
기준선은 같은 기록에서 유도: 항상-2단(폴백 없음), 항상-3단, 무작위 폴백.

  python -m vlm_diagnosis.exps.cascade_eval \
    --manifest experiments/manifests/screenqa_discovery.jsonl \
    --base "results/discovery/vast/qwen25vl_sqa_base.shard*.jsonl" --device cuda:2
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
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.kv_baselines import KVShape, dense_storage, max_keep_for_budget
from vlm_diagnosis.core.metrics import anls, exact_match
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."


@torch.no_grad()
def tier01_forward(model, processor, img, question, drafts, keep, n_vis, device):
    """검증 forward 1회: (수락된 초안 or None, a4 margin).
    drafts를 이어붙여 한 시퀀스에서 각 초안 구간을 병렬 검증한다."""
    tok = processor.tokenizer
    ins = S.vlm_inputs(processor, img, question + BRIEF, device)
    q_len = ins["input_ids"].shape[1]
    spans = []
    ids_list = [ins["input_ids"]]
    cur = q_len
    for d in drafts:
        di = tok(d, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if di.shape[1] == 0:
            continue
        ids_list.append(di)
        spans.append((cur, cur + di.shape[1], d, di))
        cur += di.shape[1]
    full = torch.cat(ids_list, 1)
    sp = token_spans(full, model.config)
    vis, vis_end, L = sp["visual"], sp["vis_end"], sp["L"]
    evict = torch.tensor([int(vis[o]) for o in range(n_vis) if o not in keep],
                         device=device)
    attn2d = torch.ones(1, L, dtype=torch.long, device=device)
    pos = mrope_position_ids(model, full, ins["image_grid_thw"], attn2d)
    m4 = causal_mask_4d(L, device, torch.float16)
    if evict.numel():
        m4 = evict_columns(m4, evict, row_start=vis_end + 1)
    # 초안 구간끼리는 서로 못 보게 (각 초안이 질문 직후에 오는 것처럼) —
    # 구간 s..e 의 행은 [0..q_len)과 자기 구간만 본다
    for (s, e, _, _) in spans:
        m4[0, 0, s:e, q_len:s] = torch.finfo(torch.float16).min
    out = model(input_ids=full, attention_mask=m4, position_ids=pos,
                pixel_values=ins["pixel_values"],
                image_grid_thw=ins["image_grid_thw"], use_cache=False)
    lg = out.logits[0].float()
    # a4: 질문 마지막 토큰 위치(첫 답 토큰 예측)의 margin
    t2 = torch.topk(lg[q_len - 1], 2).values
    a4 = float(t2[0] - t2[1])
    accepted = None
    for (s, e, dtext, di) in spans:
        pred = lg[s - 1:e - 1].argmax(-1)
        # 첫 토큰은 질문 마지막 위치(q_len-1)의 예측이어야 하나, 구간이 질문
        # 직후가 아니므로: 구간 s의 이전 위치는 다른 초안일 수 있음 → 첫 토큰은
        # q_len-1 위치 예측과 비교, 나머지는 구간 내부 비교
        first_ok = int(lg[q_len - 1].argmax()) == int(di[0, 0])
        rest_ok = bool((pred[1:] == di[0, 1:]).all()) if di.shape[1] > 1 else True
        # 구간 내부 첫 위치(s-1)는 이전 구간 마지막이라 무효 — 위 first_ok로 대체
        inner = bool((lg[s:e - 1].argmax(-1) == di[0, 1:]).all()) if di.shape[1] > 1 else True
        if first_ok and inner:
            accepted = dtext
            break
    return accepted, a4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=1.914)   # 파일럿 보정값 (프로토콜)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", default="results/discovery/cascade.jsonl")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    base_pred = {}
    for f in glob.glob(a.base):
        for l in open(f):
            r = json.loads(l)
            base_pred[r["question_id"]] = r["prediction"]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    rows = rows[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
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
            qs_eval = row["questions"][4:6]
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
            drafts = [base_pred[q["question_id"]] for q in qs_src
                      if base_pred.get(q["question_id"])]
            for qB in qs_eval:
                accepted, a4 = tier01_forward(model, processor, img,
                                              qB["question"], drafts, union,
                                              n_vis, a.device)
                ins = S.vlm_inputs(processor, img, qB["question"] + BRIEF, a.device)
                sp = token_spans(ins["input_ids"], model.config)
                evict = torch.tensor(
                    [int(sp["visual"][o]) for o in range(n_vis) if o not in union],
                    device=a.device)
                # 항상-2단/3단 기준선용 답도 함께 기록 (같은 표본에서 공정 비교)
                ans2 = greedy_generate_masked(model, processor, ins,
                                              max_new_tokens=a.max_new_tokens,
                                              evict_cols=evict,
                                              row_start=sp["vis_end"] + 1)
                ans3 = greedy_generate_masked(model, processor, ins,
                                              max_new_tokens=a.max_new_tokens)
                if accepted is not None:
                    tier, ans = 1, accepted
                elif a4 >= a.tau:
                    tier, ans = 2, ans2
                else:
                    tier, ans = 3, ans3
                f.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "question_id": qB["question_id"], "gold": qB["answers"],
                    "tier": tier, "a4": round(a4, 4),
                    "draft_accepted": accepted is not None,
                    "answer": ans, "em": exact_match(ans, qB["answers"]),
                    "anls": anls(ans, qB["answers"]),
                    "em_tier2": exact_match(ans2, qB["answers"]),
                    "em_tier3": exact_match(ans3, qB["answers"]),
                    "budget": a.budget, "tau": a.tau,
                    "keep_tokens": len(union), "n_visual": n_vis},
                    ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} {time.time()-t0:.0f}s",
                  flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
