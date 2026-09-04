"""Core–Delta 누적 선택 판 — 같은 화면에 질문이 반복될 때, 전체 캐시(DRAM)에서 무엇을 골라 쓸지를
질문마다 갱신한다. 두 운용 방식 × 이미지/기록 비율 alpha.

전제: 전체 KV는 DRAM에 있다. 실험 대상은 "선택"(= 압축)이다. 각 질문은 그 시점에 골라 둔
조각만 보고 답하며, 그 질문은 선택에 관여하지 않은 새 질문이다.

기준 두 가지 (둘 다 전체 캐시 위에서 잼)
  이미지 기준 : 질문 없이 이미지만으로 계산 (kvzip = 설명 생성 후 최대 attention / image = 시각
                행 attention 평균). 화면당 한 번.
  기록 기준   : 이전 질문들과 그 답의 행이 각 조각을 참고한 attention 평균을 질문별로 계산해
                두고, 단계 t에서는 이전 질문 1..t-1 의 평균을 쓴다.
결합: 예산 중 round(alpha·예산)를 이미지 기준 순위로, 나머지를 기록 기준 순위로 (정확히 예산).
      alpha=1 이미지만(KVzip), alpha=0 기록만.

운용 방식
  keep : 질문마다 20%를 다시 고른다 (크기 유지, 구성 교체).
  grow : 질문마다 20%를 더 고른다 (이미 있는 것은 유지, 나머지에서 추가) → 20, 40, 60, 80%.
단계 1은 기록이 없으므로 두 방식·모든 alpha가 같은 집합(이미지 기준 20%)이다.

대조: random_keep(20% 고정), random_grow(20%씩 무작위 추가), oracle(그 질문 자신의 attention,
      같은 크기), FULL. sink 4 token은 항상 포함.

  python -m vlm_diagnosis.exps.core_delta_incremental --limit 1 --device cuda:0
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
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.attnstat import QKCapture
from vlm_diagnosis.core.metrics import anls, exact_match, normalize_text
from vlm_diagnosis.core.kv_select import (
    per_head_column_stats, select_triples, greedy_generate_perhead, kept_composition, kv_bytes)
from vlm_diagnosis.core import signals as S
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS, BRIEF
from vlm_diagnosis.exps.core_delta_dram import core_stats_kvzip, core_stats_image, N_SINK

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _fmt(x: float) -> str:
    return f"{x:g}"


@torch.no_grad()
def answer_rows_stats(model, processor, img, ins, answer, device, P_shared):
    """[이미지 + 질문 + 답] forward에서 질문·머리말·답 행이 각 접두 조각에 준 attention 평균."""
    if answer.strip():
        a_ids = processor.tokenizer(answer, add_special_tokens=False,
                                    return_tensors="pt").input_ids.to(device)
        full = torch.cat([ins["input_ids"], a_ids], 1)
    else:
        full = ins["input_ids"]
    L = full.shape[1]
    attn = torch.ones(1, L, dtype=torch.long, device=device)
    pos = mrope_position_ids(model, full, ins["image_grid_thw"], attn)
    with QKCapture() as cap:
        model(input_ids=full, attention_mask=attn, position_ids=pos,
              pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"],
              use_cache=False)
        mean, _ = per_head_column_stats(cap.qk, P_shared, L)
    return mean[:, :, :P_shared].clone(), L - P_shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4, help="질문 수 (q1..q_steps 순서대로)")
    ap.add_argument("--budget-step", type=float, default=0.2, help="한 단계 예산 (접두 KV 대비)")
    ap.add_argument("--alphas", default="1,0.75,0.5,0.25,0")
    ap.add_argument("--schemes", default="keep,grow")
    ap.add_argument("--image-signal", default="kvzip", choices=["kvzip", "image"])
    ap.add_argument("--hist-rows", default="episode", choices=["episode", "question"],
                    help="기록 기준에 쓰는 행: episode=질문+답, question=질문만")
    ap.add_argument("--no-sink-protect", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/smoke/core_delta_incremental.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reverse", action="store_true",
                    help="shard 내 이미지를 역순으로 처리 (같은 shard를 두 GPU가 양끝에서 처리할 때)")
    a = ap.parse_args()

    alphas = sorted({float(x) for x in a.alphas.split(",")}, reverse=True)
    schemes = [s.strip() for s in a.schemes.split(",") if s.strip()]
    assert set(schemes) <= {"keep", "grow"}
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    rows = rows[a.shard::a.nshards]
    if a.reverse:
        rows = rows[::-1]
    if a.limit:
        rows = rows[:a.limit]
    if a.nshards > 1:
        a.out = a.out.replace(".jsonl", f".shard{a.shard}.jsonl")
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    N_LAYERS, N_KV_HEADS, HEAD_DIM = kv_dims(model)
    PER_TOKEN = N_LAYERS * N_KV_HEADS
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_id = f"inc-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    done = set()
    if a.resume and os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                if r.get("record_type") != "run_metadata":
                    done.add(str(r["sample_id"]))
            except Exception:
                pass
        print(f"[resume] 이미 완료한 표본 {len(done)}개 건너뜀", flush=True)

    with open(out_path, "a" if a.resume else "w") as f:
        f.write(json.dumps({
            "record_type": "run_metadata", "schema_version": "1.0",
            "run_id": run_id, "stage": "CORE_DELTA_INCREMENTAL", "run_kind": "smoke",
            "model": a.model, "manifest_path": a.manifest,
            "kv_dims": {"layers": N_LAYERS, "kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM},
            "steps": a.steps, "budget_step": a.budget_step, "alphas": alphas, "schemes": schemes,
            "image_signal": a.image_signal, "hist_rows": a.hist_rows,
            "sink_protect": not a.no_sink_protect,
            "protocol": "full KV assumed in DRAM; before answering question t the active set is "
                        "re-selected (keep: budget_step) or extended (grow: t*budget_step) using "
                        "alpha*image-signal + (1-alpha)*mean attention of questions 1..t-1 (+answers) "
                        "measured on the full cache; question t is unseen at selection time; "
                        "question rows and generated rows see only the active set",
            "condition_id": "inc_<scheme>_a<alpha>_t<t>; random_<scheme>_t<t>; oracle_<scheme>_t<t>; FULL_KV",
            "started_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False) + "\n")

        for di, row in enumerate(rows):
            if str(row["sample_id"]) in done:
                continue
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs = row["questions"][:a.steps]
            if len(qs) < 2:
                continue
            # ---- 이미지 기준 (화면당 한 번)
            if a.image_signal == "kvzip":
                img_sig, P_shared, n_desc = core_stats_kvzip(model, processor, img, a.device)
            else:
                img_sig, P_shared, n_desc = core_stats_image(model, processor, img, a.device)
            prefix_triples = PER_TOKEN * P_shared
            T_step = int(round(a.budget_step * prefix_triples))
            sample_seed = zlib.crc32(f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF
            g = torch.Generator().manual_seed(sample_seed)
            rand_scores = torch.rand((N_LAYERS, N_KV_HEADS, P_shared), generator=g)
            sink_forced = torch.zeros((N_LAYERS, N_KV_HEADS, P_shared), dtype=torch.bool)
            if not a.no_sink_protect:
                sink_forced[:, :, :N_SINK] = True

            # ---- 질문별 전체-캐시 실행: 기준 답, 자기 attention(oracle용), 기록 기준용 attention
            per_q = []
            for q in qs:
                q_text = q["question"] + BRIEF
                ins = S.vlm_inputs(processor, img, q_text, a.device)
                sp = token_spans(ins["input_ids"], model.config)
                assert sp["vis_end"] + 2 == P_shared
                P = int(sp["L"])
                with QKCapture() as cap:
                    pred_full = greedy_generate_masked(model, processor, ins,
                                                       max_new_tokens=a.max_new_tokens)
                    q_mean, _ = per_head_column_stats(cap.qk[:N_LAYERS], P_shared, P)
                del cap
                s1 = q_mean[:, :, :P_shared].clone()
                if a.hist_rows == "episode":
                    hist, n_rows = answer_rows_stats(model, processor, img, ins, pred_full,
                                                     a.device, P_shared)
                else:
                    hist, n_rows = s1, P - P_shared
                per_q.append({"q": q, "ins": ins, "vis": sp["visual"], "P": P, "pred_full": pred_full,
                              "s1": s1, "hist": hist, "n_rows": n_rows})

            def img_top(T, forced=None):
                keep, _ = select_triples(img_sig, torch.zeros_like(img_sig), T, 1.0,
                                         sink_forced if forced is None else (forced | sink_forced))
                return keep

            def rand_top(T, forced=None):
                keep, _ = select_triples(rand_scores, torch.zeros_like(rand_scores), T, 1.0,
                                         sink_forced if forced is None else (forced | sink_forced))
                return keep

            def mixed(T, hist_mean, al, forced=None):
                fz = sink_forced if forced is None else (forced | sink_forced)
                keep, sel = select_triples(img_sig, hist_mean, T, al, fz)
                return keep

            grow_state = {al: None for al in alphas}      # alpha별 누적 집합
            grow_rand = None
            for t, item in enumerate(per_q, start=1):
                q, ins, vis, P = item["q"], item["ins"], item["vis"], item["P"]
                golds = q["answers"]
                pred_full = item["pred_full"]
                n_q = P - P_shared
                base = {"run_id": run_id, "model": a.model, "dataset": row["dataset"],
                        "sample_id": row["sample_id"], "question_id": q["question_id"], "step": t,
                        "history_question_ids": [x["q"]["question_id"] for x in per_q[:t - 1]],
                        "gold": golds, "n_prefix": P_shared, "n_visual": int(len(vis)),
                        "n_question": n_q, "prefix_triples": prefix_triples}
                f.write(json.dumps({**base, "condition_id": "FULL_KV", "selector": "full",
                                    "scheme": None, "alpha": None, "size_frac": 1.0,
                                    "prediction": pred_full, "anls": anls(pred_full, golds),
                                    "em": exact_match(pred_full, golds)},
                                   ensure_ascii=False) + "\n")
                hist_mean = (torch.stack([x["hist"] for x in per_q[:t - 1]]).mean(0)
                             if t > 1 else None)

                cache = {}    # 같은 집합은 한 번만 생성 (집합의 해시로 공유)

                def run(cond_id, selector, scheme, al, keep_prefix, size_frac, extra=None):
                    key = hash(keep_prefix.numpy().tobytes())
                    if key in cache:
                        pred, comp, shared = cache[key][0], cache[key][1], True
                    else:
                        keep = torch.ones((N_LAYERS, N_KV_HEADS, P), dtype=torch.bool)
                        keep[:, :, :P_shared] = keep_prefix
                        pred = greedy_generate_perhead(model, processor, ins, keep, P_shared,
                                                       max_new_tokens=a.max_new_tokens)
                        comp = kept_composition(keep_prefix, vis, N_SINK)
                        cache[key] = (pred, comp)
                        shared = False
                    rec = {**base, "condition_id": cond_id, "selector": selector, "scheme": scheme,
                           "alpha": al, "size_frac": size_frac,
                           "kept_bytes": kv_bytes(comp["kept_triples"], HEAD_DIM),
                           "shared_generation": shared,
                           "prediction": pred, "anls": anls(pred, golds),
                           "em": exact_match(pred, golds),
                           "loyalty": float(normalize_text(pred) == normalize_text(pred_full)),
                           **comp}
                    if extra:
                        rec.update(extra)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

                for al in alphas:
                    # keep: 매 단계 예산 T_step 으로 재선택
                    if "keep" in schemes:
                        K = img_top(T_step) if (t == 1 or al == 1.0) else mixed(T_step, hist_mean, al)
                        run(f"inc_keep_a{_fmt(al)}_t{t}", "incremental_keep", "keep", al, K, a.budget_step)
                    # grow: 이전 집합 유지 + T_step 추가
                    if "grow" in schemes:
                        prev = grow_state[al]
                        if prev is None:
                            K = img_top(T_step)
                        else:
                            T_total = int(prev.sum()) + T_step
                            K = img_top(T_total, forced=prev) if al == 1.0 else mixed(T_total, hist_mean, al, forced=prev)
                            assert bool((prev & ~K).sum() == 0)
                        grow_state[al] = K
                        run(f"inc_grow_a{_fmt(al)}_t{t}", "incremental_grow", "grow", al, K,
                            min(1.0, t * a.budget_step))
                # 대조군
                run(f"random_keep_t{t}", "random", "keep", None, rand_top(T_step), a.budget_step)
                grow_rand = rand_top(T_step) if grow_rand is None else rand_top(int(grow_rand.sum()) + T_step, forced=grow_rand)
                run(f"random_grow_t{t}", "random", "grow", None, grow_rand, min(1.0, t * a.budget_step))
                s1 = item["s1"]
                ok, _ = select_triples(s1, torch.zeros_like(s1), T_step, 1.0, sink_forced)
                run(f"oracle_keep_t{t}", "oracle_current_question", "keep", None, ok, a.budget_step)
                if t > 1:
                    og, _ = select_triples(s1, torch.zeros_like(s1), min(t * T_step, prefix_triples), 1.0, sink_forced)
                    run(f"oracle_grow_t{t}", "oracle_current_question", "grow", None, og,
                        min(1.0, t * a.budget_step))
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} prefix={P_shared} steps={len(per_q)} "
                  f"desc={n_desc} {time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
