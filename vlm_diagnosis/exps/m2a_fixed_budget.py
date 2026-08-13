"""M2-A Track 1 — 고정 byte 예산에서 selector별 sparse 보존 성능 (02_M2A §3).

한 문서에 대해:
  write 시점  : q0 에피소드(질문+모델의 실제 답)까지가 "과거". 여기까지의 정보로
                각 selector가 남길 시각 토큰을 고른다.
  read 시점   : 나머지 질문들(q1..)을 새 질문으로 간주해 생성·채점한다.
                q0 자체는 answer-carryover 위험이 있어 평가에서 제외한다.

selector (G01 안 A; 고를 때 볼 수 있는 정보 순):
  random          아무것도 안 봄 (통제 하한)
  spatial_uniform 위치만 (row-major 등간격)
  knorm           key 벡터 크기 (작은 norm 우선 보존 — kvpress KnormPress 관례)
  s5              재구성 prompt("화면 상세 설명")가 준 attention (KVzip-VLM 적응)
  h2o             과거 에피소드(q0+답) 동안 받은 누적 attention (source-aware, F_w)
  s1              평가 질문 자신의 attention (read-time 참조선 — 저장 압축 아님)

예산: serialized bytes 기준 (sparse index·position metadata 포함,
      core.kv_baselines 회계). 평가: 생성 → ANLS(공식 지표) + EM + loyalty.

실행 (smoke):
  python -m vlm_diagnosis.exps.m2a_fixed_budget --limit 3 --device cuda:0
"""
import argparse
import json
import math
import os
import time
import zlib
from datetime import datetime, timezone

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.attnstat import QKCapture, recv_column_mass
from vlm_diagnosis.core.metrics import anls, exact_match, normalize_text
from vlm_diagnosis.core.kv_baselines import (
    KVShape, dense_storage, sparse_storage, max_keep_for_budget)
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."
N_LAYERS, N_KV_HEADS, HEAD_DIM = 28, 4, 128

TIMING = {"random": "write_time", "spatial_uniform": "write_time",
          "knorm": "write_time", "s5": "write_time",
          "h2o": "write_time_source_aware", "s1": "read_time"}


def _est_bytes(shape):
    e = dense_storage(shape)
    return e.payload_bytes + e.metadata_bytes + e.position_bytes


def _sparse_bytes(shape, k):
    e = sparse_storage(shape, keep_tokens=k)
    return e.payload_bytes + e.metadata_bytes + e.position_bytes


def top_k_indices(scores, k):
    return set(torch.topk(scores, min(k, scores.shape[0])).indices.tolist())


def uniform_indices(n_vis, k):
    """row-major 등간격 — 위치만 쓰는 공간 균등 통제."""
    idx = torch.linspace(0, n_vis - 1, steps=min(k, n_vis)).round().long()
    return set(idx.tolist())


@torch.no_grad()
def episode_capture(model, processor, img, q0_text, device, max_new_tokens):
    """write 에피소드: q0 → greedy 답 생성 → [img, q0, 답] TF forward에서
    (a) h2o = 질문·답 구간이 시각 토큰에 준 누적 attention,
    (b) knorm = 시각 토큰 key norm (RoPE는 회전이라 norm 불변) 을 캡처."""
    ins = S.vlm_inputs(processor, img, q0_text, device)
    a0 = greedy_generate_masked(model, processor, ins,
                                max_new_tokens=max_new_tokens)
    a_ids = processor.tokenizer(a0, add_special_tokens=False,
                                return_tensors="pt").input_ids.to(device)
    full = torch.cat([ins["input_ids"], a_ids], 1)
    sp = token_spans(full, model.config)
    vis, vis_end, L = sp["visual"], sp["vis_end"], sp["L"]
    attn = torch.ones(1, L, dtype=torch.long, device=device)
    pos = mrope_position_ids(model, full, ins["image_grid_thw"], attn)
    with QKCapture() as cap:
        model(input_ids=full, attention_mask=attn, position_ids=pos,
              pixel_values=ins["pixel_values"],
              image_grid_thw=ins["image_grid_thw"], use_cache=False)
        mass = recv_column_mass(cap.qk, row_start=vis_end + 1, row_end=L)
        h2o = mass[vis].cpu()
        knorm = torch.zeros(len(vis))
        for _, k in cap.qk:
            kn = k[0].float().norm(dim=-1).mean(0)          # (L,) 헤드 평균
            knorm += kn[vis].cpu()
        knorm = -(knorm / len(cap.qk))                       # 작은 norm 우선 보존
    return a0, h2o, knorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/m2a_diagnostic.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budgets", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--eval-questions-per-doc", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/smoke/m2a_track1.jsonl")
    a = ap.parse_args()

    budgets = [float(x) for x in a.budgets.split(",")]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    rows = rows[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
    if a.nshards > 1:
        a.out = a.out.replace(".jsonl", f".shard{a.shard}.jsonl")
    model, processor = load_qwen25vl(device=a.device, max_pixels=MAX_PIXELS)
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_id = f"m2a-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    with open(out_path, "w") as f:
        f.write(json.dumps({
            "record_type": "run_metadata", "schema_version": "1.1",
            "run_id": run_id, "stage": "M2A", "run_kind": "smoke",
            "manifest_path": a.manifest, "budgets_keep": budgets,
            "selectors": list(TIMING), "metric": "anls",
            "write_episode": "q0 + model greedy answer",
            "eval_questions": "q1..; q0 excluded (answer-carryover risk)",
            "started_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False) + "\n")

        for di, row in enumerate(rows):
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs = row["questions"]
            q0 = qs[0]
            a0, h2o, knorm = episode_capture(
                model, processor, img, q0["question"] + BRIEF,
                a.device, a.max_new_tokens)
            s5 = S.score_s5(model, processor, img, a.device).cpu()
            n_vis = s5.shape[0]
            sample_seed = zlib.crc32(
                f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF
            rnd = S.score_s0(n_vis, seed=sample_seed)
            shape = KVShape(layers=N_LAYERS, batch=1, kv_heads=N_KV_HEADS,
                            tokens=n_vis, head_dim=HEAD_DIM)
            full_bytes = _est_bytes(shape)

            for q in qs[1:1 + a.eval_questions_per_doc]:
                q_text = q["question"] + BRIEF
                ins = S.vlm_inputs(processor, img, q_text, a.device)
                sp = token_spans(ins["input_ids"], model.config)
                vis, vis_end = sp["visual"], sp["vis_end"]
                golds = q["answers"]
                pred_full = greedy_generate_masked(
                    model, processor, ins, max_new_tokens=a.max_new_tokens)
                base = {"run_id": run_id, "dataset": row["dataset"],
                        "split": "smoke", "sample_id": row["sample_id"],
                        "question_id": q["question_id"], "gold": golds,
                        "n_visual": n_vis}
                f.write(json.dumps({**base, "condition_id": "FULL_KV",
                                    "selection_timing": "none",
                                    "keep_ratio_target": 1.0,
                                    "estimated_serialized_bytes": full_bytes,
                                    "prediction": pred_full,
                                    "anls": anls(pred_full, golds),
                                    "em": exact_match(pred_full, golds)},
                                   ensure_ascii=False) + "\n")
                s1 = S.score_s1(model, processor, img, q_text, a.device).cpu()
                scores = {"random": rnd, "spatial_uniform": None,
                          "knorm": knorm, "s5": s5, "h2o": h2o, "s1": s1}
                for B in budgets:
                    k = max_keep_for_budget(shape, int(B * full_bytes), "sparse")
                    act = _sparse_bytes(shape, k)
                    for name, sc in scores.items():
                        keep = (uniform_indices(n_vis, k) if sc is None
                                else top_k_indices(sc, k))
                        evict = torch.tensor(
                            [int(vis[o]) for o in range(n_vis) if o not in keep],
                            device=a.device)
                        pred = greedy_generate_masked(
                            model, processor, ins,
                            max_new_tokens=a.max_new_tokens,
                            evict_cols=evict, row_start=vis_end + 1)
                        f.write(json.dumps({
                            **base, "condition_id": f"{name}@B{int(B*100)}",
                            "selector": name,
                            "selection_timing": TIMING[name],
                            "keep_ratio_target": B,
                            "keep_tokens": k,
                            "estimated_serialized_bytes": act,
                            "budget_utilization": round(act / (B * full_bytes), 4),
                            "prediction": pred,
                            "anls": anls(pred, golds),
                            "em": exact_match(pred, golds),
                            "loyalty": float(normalize_text(pred)
                                             == normalize_text(pred_full))},
                            ensure_ascii=False) + "\n")
                        f.flush()
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} n_vis={n_vis} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
