"""LEGACY reconstruction/S1 visual-only sweep (재현용으로 보존).

현재 dual-prefill 방식은 ``core_delta_full_kv.py``를 사용한다. 이 파일은 기존
KVzip-reconstruction 결과와의 비교 및 과거 결과 재현만을 위해 남겨 둔다.

§3 전제의 구현: 시각 KV는 한 번만 만들어진다 (image + 현재 질문 prefill). 조건마다
바뀌는 것은 **어느 시각 KV column을 질문·생성 행에서 차단할지** (mask) 뿐이다.
  core importance  C(I)   = score_kvzip (기본; --core s5 로 교체 가능) — 질문 무관, 이미지당 1회
  query relevance  Δ(I,q) = score_s1  — 현재 질문 token이 시각 KV에 준 attention, 질문당 1회
결합: core_delta_keep(C, Δ, B, alpha) — B_C=round(alpha·B) core 보호 + 나머지 query 상위.
alpha=0 → pure S1, alpha=1 → pure KVzip-VLM, 중간 → core–delta.

예산: 모든 조건이 같은 keep_count(=같은 serialized byte, kv_baselines 회계)를 쓴다.
비교군(§4.4): FULL_KV, random, spatial_uniform, alpha 격자(양끝이 KVzip-only / S1-only),
선택적 weighted-sum ablation(--wsum).

표본·평가 규약은 m2a_fixed_budget(사다리, D6)과 동일: 이미지당 q1..q3 평가(q0 제외),
BRIEF 접미사, greedy 생성, EM/ANLS/loyalty.

실행 (smoke):
  python -m vlm_diagnosis.exps.core_delta_sweep --limit 2 --device cuda:0
"""
import argparse
import json
import os
import time
import zlib
from datetime import datetime, timezone

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.metrics import anls, exact_match, normalize_text
from vlm_diagnosis.core.kv_baselines import KVShape, max_keep_for_budget
from vlm_diagnosis.core.core_delta import (
    core_delta_keep, weighted_sum_keep, visual_kv_invariance)
from vlm_diagnosis.core import signals as S
from vlm_diagnosis.exps.m2a_fixed_budget import (
    _est_bytes, _sparse_bytes, top_k_indices, uniform_indices, MAX_PIXELS, BRIEF)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _fmt(x: float) -> str:
    return f"{x:g}"


def _core_scores(model, processor, img, device, which):
    out = {}
    if "kvzip" in which:
        out["kvzip"] = S.score_kvzip(model, processor, img, device).cpu()
    if "s5" in which:
        out["s5"] = S.score_s5(model, processor, img, device).cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budgets", default="0.05", help="keep-ratio 목표 (byte 회계로 환산)")
    ap.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,1")
    ap.add_argument("--core", default="kvzip", help="쉼표 목록: kvzip,s5")
    ap.add_argument("--wsum", action="store_true",
                    help="약한 형태 ablation: rank 가중합 top-k (alpha 격자의 중간값 재사용)")
    ap.add_argument("--no-controls", action="store_true", help="random/spatial_uniform 생략")
    ap.add_argument("--query-from", default="self", choices=["self", "q0"],
                    help="query 점수의 출처. self: 평가 질문 자신의 S1 (§4 상보성 검정). "
                         "q0: 과거 질문 q0의 S1로 cache를 만들고 q1..을 평가 (§11 '재사용 core' 검정 — "
                         "core 이름에 Q0 접미사, alpha=0이 M3 transfer 조건과 동일)")
    ap.add_argument("--eval-questions-per-doc", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/smoke/core_delta.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-kv-invariance", action="store_true",
                    help="§3 검증(첫 표본에서 시각 KV 불변성 측정) 생략")
    a = ap.parse_args()

    budgets = [float(x) for x in a.budgets.split(",")]
    alphas = sorted({float(x) for x in a.alphas.split(",")})
    cores = [c.strip() for c in a.core.split(",") if c.strip()]
    for c in cores:
        assert c in ("kvzip", "s5"), c
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    rows = rows[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
    if a.nshards > 1:
        a.out = a.out.replace(".jsonl", f".shard{a.shard}.jsonl")
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    N_LAYERS, N_KV_HEADS, HEAD_DIM = kv_dims(model)
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_id = f"cd-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    done = set()
    if a.resume and os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                if r.get("record_type") not in ("run_metadata", "kv_invariance"):
                    done.add(str(r["sample_id"]))
            except Exception:
                pass
        print(f"[resume] 이미 완료한 표본 {len(done)}개 건너뜀", flush=True)

    with open(out_path, "a" if a.resume else "w") as f:
        f.write(json.dumps({
            "record_type": "run_metadata", "schema_version": "1.0",
            "run_id": run_id, "stage": "CORE_DELTA_PHASE_A", "run_kind": "smoke",
            "model": a.model, "manifest_path": a.manifest,
            "budgets_keep": budgets, "alphas": alphas, "core_scores": cores,
            "query_score": "s1", "query_from": a.query_from, "wsum_ablation": a.wsum,
            "selection": "core_delta_keep: round(alpha*B) core top-k protected, "
                         "remaining budget filled by query top-k among unselected; |keep|==B",
            "eviction": "V2 mask — visual KV computed once (full prefill), columns "
                        "blocked for rows >= vis_end+1 (question + generation)",
            "eval_questions": "q1..q{n}; q0 excluded (same frozen sample as D6 ladder)",
            "metric": "em (primary), anls, loyalty",
            "started_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False) + "\n")

        inv_done = a.skip_kv_invariance
        for di, row in enumerate(rows):
            if str(row["sample_id"]) in done:
                continue
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs = row["questions"]
            eval_qs = qs[1:1 + a.eval_questions_per_doc]
            if not eval_qs:
                continue

            if not inv_done:
                # §3: 질문을 바꿔도 시각 KV는 같은가 — 첫 표본에서 수치로 확인·기록
                inv = visual_kv_invariance(
                    model, processor, img, eval_qs[0]["question"] + BRIEF, "x", a.device)
                f.write(json.dumps({"record_type": "kv_invariance", "run_id": run_id,
                                    "sample_id": row["sample_id"], **inv}) + "\n")
                f.flush()
                print(f"[kv_invariance] {inv}", flush=True)
                inv_done = True
                torch.cuda.empty_cache()

            core = _core_scores(model, processor, img, a.device, cores)
            n_vis = next(iter(core.values())).shape[0]
            s1_past, past_qid, tag = None, None, ""
            if a.query_from == "q0":
                # 재사용 검정: 과거 질문 q0의 attention으로 delta를 고르고 q1..을 평가
                s1_past = S.score_s1(model, processor, img, qs[0]["question"] + BRIEF,
                                     a.device).cpu()
                past_qid, tag = qs[0]["question_id"], "Q0"
            sample_seed = zlib.crc32(f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF
            rnd = S.score_s0(n_vis, seed=sample_seed)
            shape = KVShape(layers=N_LAYERS, batch=1, kv_heads=N_KV_HEADS,
                            tokens=n_vis, head_dim=HEAD_DIM)
            full_bytes = _est_bytes(shape)

            for q in eval_qs:
                q_text = q["question"] + BRIEF
                ins = S.vlm_inputs(processor, img, q_text, a.device)
                sp = token_spans(ins["input_ids"], model.config)
                vis, vis_end = sp["visual"], sp["vis_end"]
                assert len(vis) == n_vis, (len(vis), n_vis)
                golds = q["answers"]
                pred_full = greedy_generate_masked(
                    model, processor, ins, max_new_tokens=a.max_new_tokens)
                base = {"run_id": run_id, "model": a.model, "dataset": row["dataset"],
                        "sample_id": row["sample_id"], "question_id": q["question_id"],
                        "gold": golds, "n_visual": n_vis}
                f.write(json.dumps({**base, "condition_id": "FULL_KV", "selector": "full",
                                    "keep_ratio_target": 1.0, "keep_tokens": n_vis,
                                    "estimated_serialized_bytes": full_bytes,
                                    "prediction": pred_full,
                                    "anls": anls(pred_full, golds),
                                    "em": exact_match(pred_full, golds)},
                                   ensure_ascii=False) + "\n")
                if s1_past is None:
                    s1 = S.score_s1(model, processor, img, q_text, a.device).cpu()
                else:
                    s1 = s1_past
                    base["query_from"] = "q0"
                    base["past_question_id"] = past_qid

                def run(cond_id, selector, keep, B, k, act, extra=None):
                    evict = torch.tensor(
                        [int(vis[o]) for o in range(n_vis) if o not in keep],
                        device=a.device)
                    assert len(keep) == k, (cond_id, len(keep), k)
                    pred = greedy_generate_masked(
                        model, processor, ins, max_new_tokens=a.max_new_tokens,
                        evict_cols=evict, row_start=vis_end + 1)
                    rec = {**base, "condition_id": cond_id, "selector": selector,
                           "keep_ratio_target": B, "keep_tokens": k,
                           "estimated_serialized_bytes": act,
                           "budget_utilization": round(act / (B * full_bytes), 4),
                           "prediction": pred, "anls": anls(pred, golds),
                           "em": exact_match(pred, golds),
                           "loyalty": float(normalize_text(pred) == normalize_text(pred_full))}
                    if extra:
                        rec.update(extra)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

                for B in budgets:
                    k = max_keep_for_budget(shape, int(B * full_bytes), "sparse")
                    act = _sparse_bytes(shape, k)
                    Bt = f"B{_fmt(B * 100)}"
                    if not a.no_controls:
                        run(f"random@{Bt}", "random", top_k_indices(rnd, k), B, k, act)
                        run(f"spatial_uniform@{Bt}", "spatial_uniform",
                            uniform_indices(n_vis, k), B, k, act)
                    for cname, cscore in core.items():
                        for al in alphas:
                            keep, info = core_delta_keep(cscore, s1, k, al)
                            run(f"cd_{cname}{tag}_a{_fmt(al)}@{Bt}", f"cd_{cname}{tag}",
                                keep, B, k, act,
                                extra={"alpha": al, "core": cname,
                                       "core_delta": info.as_dict()})
                        if a.wsum:
                            for w in alphas:
                                if w in (0.0, 1.0):
                                    continue   # 양끝은 alpha 격자와 동일
                                keep = weighted_sum_keep(cscore, s1, k, w)
                                run(f"wsum_{cname}{tag}_w{_fmt(w)}@{Bt}", f"wsum_{cname}{tag}",
                                    keep, B, k, act, extra={"w_core": w, "core": cname})
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} n_vis={n_vis} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
