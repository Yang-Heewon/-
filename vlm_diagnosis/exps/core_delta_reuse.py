"""Core–Delta 재사용 판 — 쓰기 시점에 두 기준(이미지만 / 이미지+과거 질문)으로 고르고 나머지는
삭제한 뒤, 아직 보지 않은 새 질문으로 평가한다 (KVzip의 query-agnostic 다중 질문 규약).

쓰기 시점 (새 질문은 모름)
  이미지 기준   : 이미지만 넣었을 때의 중요도. 두 가지 신호를 나란히 계산한다.
                    kvzip = 범용 지시("보이는 내용을 그대로 옮겨라")로 설명을 생성시켜 설명 행이
                            각 접두 조각에 준 attention의 최대 (KVzip 재구성 점수의 이미지 적응)
                    image = 접두(system + vision 경계 + 시각)만 prefill하고 시각 행들이 각 접두
                            조각에 준 attention의 평균 (생성 없음)
  과거 질문 기준: 이미지 + 과거 질문 q0(+ 모델이 그때 낸 답)을 prefill하고, 질문·답 행들이 각
                  접두 조각에 준 attention의 평균 (D6 사다리의 h2o와 같은 source-aware 신호).
  결합          : 예산 B(접두 KV 세 짝 수 대비 비율) 중 round(alpha·B)를 이미지 기준 순위로 먼저,
                  나머지를 아직 안 뽑힌 것 중 과거 질문 기준 순위로 채운다 (정확히 B). 나머지 삭제.
                  alpha=1 이미지만(KVzip 해당), alpha=0 과거 질문만(h2o 해당). sink 4 token 항상 포함.
읽기 시점 (새 질문 q1..q3 도착)
  남긴 B만으로 답한다. 질문 행과 생성 행 모두 남긴 조각만 본다 (row_start = 질문 첫 token).
  질문 token 자체는 새로 만들어지므로 항상 보존 (예산 밖).
기준선: random(+sink), oracle_s1(새 질문 자신의 attention으로 고른 상한), FULL.
평가 질문은 D6 사다리와 같이 q1..q3, q0는 쓰기 에피소드로만 쓴다.

  python -m vlm_diagnosis.exps.core_delta_reuse --limit 1 --device cuda:0
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
    per_head_column_stats, select_triples, select_tokens, greedy_generate_perhead,
    kept_composition, kv_bytes, index_bytes)
from vlm_diagnosis.core import signals as S
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS, BRIEF
from vlm_diagnosis.exps.core_delta_dram import core_stats_kvzip, core_stats_image, N_SINK

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _fmt(x: float) -> str:
    return f"{x:g}"


@torch.no_grad()
def past_episode_stats(model, processor, img, q0_text, device, max_new_tokens, rows="episode"):
    """이미지 + 과거 질문 q0 (+ 모델 답) prefill에서 질문·답 행이 각 접두 조각에 준 attention.
    rows='episode'면 질문+머리말+답 행, 'question'이면 질문+머리말 행만.
    반환 (mean[:, :, :P_shared], P_shared, a0, n_rows)."""
    ins = S.vlm_inputs(processor, img, q0_text, device)
    a0 = greedy_generate_masked(model, processor, ins, max_new_tokens=max_new_tokens)
    P_q = ins["input_ids"].shape[1]
    if rows == "episode" and a0.strip():
        a_ids = processor.tokenizer(a0, add_special_tokens=False,
                                    return_tensors="pt").input_ids.to(device)
        full = torch.cat([ins["input_ids"], a_ids], 1)
    else:
        full = ins["input_ids"]
    sp = token_spans(full, model.config)
    P_shared = sp["vis_end"] + 2
    L = full.shape[1]
    attn = torch.ones(1, L, dtype=torch.long, device=device)
    pos = mrope_position_ids(model, full, ins["image_grid_thw"], attn)
    with QKCapture() as cap:
        model(input_ids=full, attention_mask=attn, position_ids=pos,
              pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"],
              use_cache=False)
        mean, _ = per_head_column_stats(cap.qk, P_shared, L)
    return mean[:, :, :P_shared].clone(), P_shared, a0, L - P_shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budgets", default="0.05,0.1,0.2", help="B: 남길 양(접두 KV 대비 비율), 나머지 삭제")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1", help="이미지 기준 몫. 1=이미지만, 0=과거 질문만")
    ap.add_argument("--image-signals", default="kvzip,image", help="쉼표 목록: kvzip,image")
    ap.add_argument("--past-rows", default="episode", choices=["episode", "question"])
    ap.add_argument("--token-cells", default="0.1:0.5,0.1:1", help="token 단위로도 돌릴 (B:alpha) 칸, 첫 이미지 신호")
    ap.add_argument("--no-sink-protect", action="store_true")
    ap.add_argument("--eval-questions-per-doc", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/smoke/core_delta_reuse.jsonl")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    budgets = sorted({float(x) for x in a.budgets.split(",")})
    alphas = sorted({float(x) for x in a.alphas.split(",")})
    sigs = [s.strip() for s in a.image_signals.split(",") if s.strip()]
    assert sigs and set(sigs) <= {"kvzip", "image"}, sigs
    tok_cells = [tuple(float(v) for v in c.split(":")) for c in a.token_cells.split(",") if c]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    rows = rows[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
    if a.nshards > 1:
        a.out = a.out.replace(".jsonl", f".shard{a.shard}.jsonl")
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    N_LAYERS, N_KV_HEADS, HEAD_DIM = kv_dims(model)
    PER_TOKEN = N_LAYERS * N_KV_HEADS
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_id = f"ru-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

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
            "run_id": run_id, "stage": "CORE_DELTA_REUSE", "run_kind": "smoke",
            "model": a.model, "manifest_path": a.manifest,
            "kv_dims": {"layers": N_LAYERS, "kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM},
            "budgets": budgets, "alphas": alphas, "image_signals": sigs,
            "past_rows": a.past_rows, "token_cells": tok_cells,
            "sink_protect": not a.no_sink_protect,
            "protocol": "write time: select B from image-prefix triples using image-only signal "
                        "(alpha share) + past-question(q0 episode) signal (rest), delete the rest; "
                        "read time: unseen questions q1..q3 answered with the kept B only "
                        "(question rows and generated rows restricted; question tokens always kept)",
            "signals": {
                "kvzip": "per (layer, kv_head) MAX attention from generated repeat-description rows",
                "image": "per (layer, kv_head) MEAN attention from visual rows in image-only prefill",
                "past": "per (layer, kv_head) MEAN attention from q0(+answer) rows in image+q0 prefill"},
            "condition_id": "ru_<sig>_a<alpha>@B<B> (alpha=0 -> ru_past_a0@B); random@B; oracle_s1@B; "
                            "ruT_<sig>_a<alpha>@B (token granularity)",
            "eval_questions": "q1..q{n}; q0 = write-time episode only",
            "started_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False) + "\n")

        for di, row in enumerate(rows):
            if str(row["sample_id"]) in done:
                continue
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs = row["questions"]
            q0 = qs[0]
            eval_qs = qs[1:1 + a.eval_questions_per_doc]
            if not eval_qs:
                continue
            # ---- 쓰기 시점 신호
            img_sig, P_shared, n_desc = {}, None, None
            if "kvzip" in sigs:
                sc, P_shared, n_desc = core_stats_kvzip(model, processor, img, a.device)
                img_sig["kvzip"] = sc
            if "image" in sigs:
                sc, P2, _ = core_stats_image(model, processor, img, a.device)
                assert P_shared is None or P2 == P_shared
                P_shared = P2
                img_sig["image"] = sc
            past_sig, P3, a0, n_past = past_episode_stats(
                model, processor, img, q0["question"] + BRIEF, a.device, a.max_new_tokens,
                a.past_rows)
            assert P3 == P_shared, (P3, P_shared)
            prefix_triples = PER_TOKEN * P_shared
            sample_seed = zlib.crc32(f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF
            g = torch.Generator().manual_seed(sample_seed)
            rand_scores = torch.rand((N_LAYERS, N_KV_HEADS, P_shared), generator=g)
            sink_forced = torch.zeros((N_LAYERS, N_KV_HEADS, P_shared), dtype=torch.bool)
            sink_forced_tok = torch.zeros(P_shared, dtype=torch.bool)
            if not a.no_sink_protect:
                sink_forced[:, :, :N_SINK] = True
                sink_forced_tok[:N_SINK] = True
            img_tok = {s: (sc.amax(dim=(0, 1)) if s == "kvzip" else sc.mean(dim=(0, 1)))
                       for s, sc in img_sig.items()}
            past_tok = past_sig.mean(dim=(0, 1))

            # 쓰기 시점 선택은 질문과 무관 → 이미지당 한 번 만들어 모든 평가 질문에 재사용
            keeps = {}
            for B in budgets:
                T = int(round(B * prefix_triples))
                Bt = f"B{_fmt(B*100)}"
                keep, _ = select_triples(rand_scores, torch.zeros_like(rand_scores), T, 1.0, sink_forced)
                keeps[f"random@{Bt}"] = ("random", None, B, None, "head", keep)
                for si, s in enumerate(sigs):
                    for al in alphas:
                        if al == 0 and si > 0:
                            continue
                        keep, sel = select_triples(img_sig[s], past_sig, T, al, sink_forced)
                        name = f"ru_past_a0@{Bt}" if al == 0 else f"ru_{s}_a{_fmt(al)}@{Bt}"
                        keeps[name] = ("core_delta", None if al == 0 else s, B, al, "head", keep)
                for Bc, al in tok_cells:
                    if abs(Bc - B) > 1e-9:
                        continue
                    B_tok = T // PER_TOKEN
                    keep, _ = select_tokens(img_tok[sigs[0]], past_tok, B_tok, al,
                                            N_LAYERS, N_KV_HEADS, sink_forced_tok)
                    keeps[f"ruT_{sigs[0]}_a{_fmt(al)}@{Bt}"] = ("core_delta_token", sigs[0], B, al, "token", keep)

            for q in eval_qs:
                q_text = q["question"] + BRIEF
                ins = S.vlm_inputs(processor, img, q_text, a.device)
                sp = token_spans(ins["input_ids"], model.config)
                vis, vis_end, P = sp["visual"], sp["vis_end"], int(sp["L"])
                assert vis_end + 2 == P_shared, (vis_end, P_shared)
                n_q = P - P_shared
                golds = q["answers"]
                with QKCapture() as cap:
                    pred_full = greedy_generate_masked(
                        model, processor, ins, max_new_tokens=a.max_new_tokens)
                    q_mean, _ = per_head_column_stats(cap.qk[:N_LAYERS], P_shared, P)
                del cap
                s1_prefix = q_mean[:, :, :P_shared].clone()
                base = {"run_id": run_id, "model": a.model, "dataset": row["dataset"],
                        "sample_id": row["sample_id"], "question_id": q["question_id"],
                        "past_question_id": q0["question_id"], "past_answer": a0,
                        "gold": golds, "n_prefix": P_shared, "n_visual": int(len(vis)),
                        "n_question": n_q, "n_past_rows": n_past, "prefix_triples": prefix_triples}
                f.write(json.dumps({**base, "condition_id": "FULL_KV", "selector": "full",
                                    "keep_frac": 1.0, "kept_triples": prefix_triples,
                                    "prediction": pred_full, "anls": anls(pred_full, golds),
                                    "em": exact_match(pred_full, golds)},
                                   ensure_ascii=False) + "\n")

                def run(cond_id, selector, sig, B, al, gran, keep_prefix):
                    keep = torch.ones((N_LAYERS, N_KV_HEADS, P), dtype=torch.bool)
                    keep[:, :, :P_shared] = keep_prefix
                    pred = greedy_generate_perhead(model, processor, ins, keep, P_shared,
                                                   max_new_tokens=a.max_new_tokens)
                    comp = kept_composition(keep_prefix, vis, N_SINK)
                    rec = {**base, "condition_id": cond_id, "selector": selector,
                           "image_signal": sig, "granularity": gran, "keep_frac": B, "alpha": al,
                           "kept_bytes": kv_bytes(comp["kept_triples"], HEAD_DIM),
                           "index_bytes": index_bytes(N_LAYERS, N_KV_HEADS, P_shared, gran),
                           "prediction": pred, "anls": anls(pred, golds),
                           "em": exact_match(pred, golds),
                           "loyalty": float(normalize_text(pred) == normalize_text(pred_full)),
                           **comp}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

                for cond_id, (selector, sig, B, al, gran, keep) in keeps.items():
                    run(cond_id, selector, sig, B, al, gran, keep)
                # 상한: 새 질문 자신의 attention으로 고른 것 (읽기 시점 정보 사용, 방법이 아님)
                for B in budgets:
                    T = int(round(B * prefix_triples))
                    keep, _ = select_triples(s1_prefix, torch.zeros_like(s1_prefix), T, 1.0, sink_forced)
                    run(f"oracle_s1@B{_fmt(B*100)}", "oracle_current_question", None, B, None, "head", keep)
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} prefix={P_shared} vis={len(vis)} "
                  f"desc={n_desc} past_rows={n_past} {time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
