"""Core–Delta DRAM 판 — 쓰기 시점에 DRAM에 둘 양 B만 남기고 나머지는 삭제, 두 core 신호를 나란히.

이 파일은 core_delta_write_read.py(다른 세션이 image-only prefill 신호로 개정)와 독립적으로,
같은 격자를 **두 가지 쓰기 시점 신호**로 돌린다.
  kvzip : 질문 없이 "보이는 내용을 그대로 옮겨 적어라" 지시로 설명을 생성시키고, 설명 행이
          각 접두 열에 준 attention의 최대값 (층, KV head별). KVzip 원저의 재구성 점수 적응
          (사용자 원안 VLM_idea v1 §2·§4, D6에서 GUI 최강 write-time 신호).
  image : system+image 경계까지만 prefill하고, 시각 token 행들이 각 접두 열에 준 attention의
          평균 (층, KV head별). 생성 없음 (다른 세션의 dual-prefill 안).
읽기 시점 delta 는 둘 다 같은 joint prefill(이미지+질문)의 질문 행 attention 평균으로 고른다.

설정 (두 저장소 구조)
  쓰기 시점 (질문 없음) : 접두 KV(system + vision 경계 + 시각)에서 core 신호 순위로 DRAM 보관분
                          B(예: 10/20/40%)를 고르고 나머지는 삭제. 같은 순위의 상위 C가 GPU 상주 core.
  읽기 시점 (질문 도착) : DRAM 보관분 B 안에서 core에 없는 것 중 질문 점수 상위 D를 GPU로 올림.
  답 생성               : 질문 행과 생성 행 모두 core ∪ delta만 본다. 질문 token은 항상 보존
                          (읽기 시점에 새로 생김; C·D 예산 밖, 뜨거운 캐시 크기에는 포함).
비교군 : core만 B (= 같은 저장량의 KVzip, B 전부 GPU) / core만 C+D (= 같은 GPU 사용량의 KVzip)
         / delta만 (C=0) / 무작위 core / token 단위 일부 칸 / B=1 전부 보관(참고).
sink(앞 4 token)는 core에 항상 포함. 비율은 접두 KV 세 짝(층×KV head×token) 수 기준.

  python -m vlm_diagnosis.exps.core_delta_dram --limit 1 --device cuda:0 --core-signals kvzip,image
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

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
REPEAT_PROMPT = ("Repeat the entire visible content of the image exactly, "
                 "including all text.")
N_SINK = 4


def _fmt(x: float) -> str:
    return f"{x:g}"


def _parse_grid(s):
    return sorted({float(x) for x in s.split(",")}) if s else []


@torch.no_grad()
def core_stats_kvzip(model, processor, img, device, max_new_tokens=96):
    """KVzip식: 범용 지시로 설명 생성 → 설명 행이 각 열에 준 attention 최대 (층, KV head, 열).
    반환 (peak[:, :, :P_shared], P_shared, n_desc)."""
    ins = S.vlm_inputs(processor, img, REPEAT_PROMPT, device)
    gen = model.generate(**{k: v for k, v in ins.items()},
                         max_new_tokens=max_new_tokens, do_sample=False)
    sp = token_spans(gen, model.config)
    P = ins["input_ids"].shape[1]
    attn = torch.ones(1, gen.shape[1], dtype=torch.long, device=device)
    pos = mrope_position_ids(model, gen, ins["image_grid_thw"], attn)
    with QKCapture() as cap:
        model(input_ids=gen, attention_mask=attn, position_ids=pos,
              pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"],
              use_cache=False)
        _, peak = per_head_column_stats(cap.qk, P, gen.shape[1])
    P_shared = sp["vis_end"] + 2
    return peak[:, :, :P_shared].clone(), P_shared, gen.shape[1] - P


@torch.no_grad()
def core_stats_image(model, processor, img, device):
    """image-only prefill: 접두(system + vision 경계 + 시각)만 넣고, 시각 행들이 각 접두 열에
    준 attention 평균 (층, KV head, 열). 생성 없음. 반환 (mean, P_shared, n_image_rows)."""
    ins = S.vlm_inputs(processor, img, "x", device)
    sp = token_spans(ins["input_ids"], model.config)
    P_shared = sp["vis_end"] + 2
    ids = ins["input_ids"][:, :P_shared]
    attn = torch.ones_like(ids)
    pos = mrope_position_ids(model, ids, ins["image_grid_thw"], attn)
    with QKCapture() as cap:
        model(input_ids=ids, attention_mask=attn, position_ids=pos,
              pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"],
              use_cache=False)
        v0 = int(sp["visual"].min())
        mean, _ = per_head_column_stats(cap.qk, v0, P_shared)
    return mean.clone(), P_shared, P_shared - v0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--core-signals", default="kvzip,image", help="쉼표 목록: kvzip,image")
    ap.add_argument("--cold-budgets", default="0.1,0.2,0.4",
                    help="B: DRAM 보관분(접두 KV 대비 비율), 나머지 삭제. 1 = 전부 보관(참고)")
    ap.add_argument("--core-sizes", default="0,0.025,0.05", help="C: GPU 상주 core 크기")
    ap.add_argument("--delta-sizes", default="0.01,0.025", help="D: 질문 시 DRAM→GPU delta 크기")
    ap.add_argument("--core-only-sizes", default="0.025,0.035,0.05,0.06,0.075,0.1,0.2,0.4",
                    help="core만(delta 0) 크기 — 같은 저장량/같은 GPU 사용량의 KVzip 기준선")
    ap.add_argument("--random-core-cells", default="0.2:0.05:0.01", help="무작위 core (B:C:D)")
    ap.add_argument("--token-cells", default="0.2:0:0.01,0.2:0.05:0.01",
                    help="token 단위(공통 마스크) (B:C:D)")
    ap.add_argument("--no-sink-protect", action="store_true")
    ap.add_argument("--eval-questions-per-doc", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/smoke/core_delta_dram.jsonl")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    signals = [s.strip() for s in a.core_signals.split(",") if s.strip()]
    assert set(signals) <= {"kvzip", "image"}, signals
    C_list, D_list, B_list = _parse_grid(a.core_sizes), _parse_grid(a.delta_sizes), _parse_grid(a.cold_budgets)
    core_only = _parse_grid(a.core_only_sizes)
    rand_cells = [tuple(float(v) for v in c.split(":")) for c in a.random_core_cells.split(",") if c]
    tok_cells = [tuple(float(v) for v in c.split(":")) for c in a.token_cells.split(",") if c]
    for B, C, D in rand_cells + tok_cells:
        assert C + D <= B + 1e-9, f"cell B{B}:C{C}:D{D} — core+delta가 DRAM 보관량을 넘음"
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
    run_id = f"wd-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

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
            "run_id": run_id, "stage": "CORE_DELTA_DRAM", "run_kind": "smoke",
            "model": a.model, "manifest_path": a.manifest,
            "kv_dims": {"layers": N_LAYERS, "kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM},
            "core_signals": signals, "cold_budgets": B_list, "core_sizes": C_list,
            "delta_sizes": D_list, "core_only_sizes": core_only,
            "random_core_cells": rand_cells, "token_cells": tok_cells,
            "sink_protect_in_core": not a.no_sink_protect,
            "core_signal_defs": {
                "kvzip": "per (layer, kv_head) MAX attention from generated repeat-description rows "
                         "(generic prompt, no question) to prefix columns",
                "image": "per (layer, kv_head) MEAN attention from visual-token rows to prefix "
                         "columns in an image-only prefill (no text, no generation)"},
            "read_time_delta": "top-D by mean question+header-row attention (joint prefill) among "
                               "DRAM-tier triples not in core (exact scores; selector cost not modeled)",
            "cold_tier": "B<1: top-B by the core signal kept in DRAM, rest DELETED; core = top-C of "
                         "the same ranking; delta only inside B. B=1: everything retained",
            "semantics": "question rows and generated rows see only core ∪ delta (row_start = first "
                         "question token); question tokens always kept, not counted in C/D",
            "condition_id": "wd_<signal>[B<B>]_C<C>_D<D>; wdR = random core; wdT = token granularity",
            "eval_questions": "q1..q{n}; q0 excluded",
            "started_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False) + "\n")

        for di, row in enumerate(rows):
            if str(row["sample_id"]) in done:
                continue
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs = row["questions"]
            eval_qs = qs[1:1 + a.eval_questions_per_doc]
            if not eval_qs:
                continue
            # ---- 쓰기 시점: 두 신호 (둘 다 질문 없음)
            core_scores, P_shared, n_desc = {}, None, None
            if "kvzip" in signals:
                sc, P_shared, n_desc = core_stats_kvzip(model, processor, img, a.device)
                core_scores["kvzip"] = sc
            if "image" in signals:
                sc, P2, _ = core_stats_image(model, processor, img, a.device)
                assert P_shared is None or P2 == P_shared, (P2, P_shared)
                P_shared = P2
                core_scores["image"] = sc
            prefix_triples = PER_TOKEN * P_shared
            sample_seed = zlib.crc32(f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF
            g = torch.Generator().manual_seed(sample_seed)
            rand_scores = torch.rand((N_LAYERS, N_KV_HEADS, P_shared), generator=g)
            sink_forced = torch.zeros((N_LAYERS, N_KV_HEADS, P_shared), dtype=torch.bool)
            sink_forced_tok = torch.zeros(P_shared, dtype=torch.bool)
            if not a.no_sink_protect:
                sink_forced[:, :, :N_SINK] = True
                sink_forced_tok[:N_SINK] = True

            def keep_head(frac, scores):
                T = int(round(frac * prefix_triples))
                if T <= 0:
                    return torch.zeros((N_LAYERS, N_KV_HEADS, P_shared), dtype=torch.bool)
                keep, _ = select_triples(scores, torch.zeros_like(scores), T, 1.0, sink_forced)
                return keep

            def keep_tok(frac, scores_tok):
                B_tok = int(round(frac * prefix_triples)) // PER_TOKEN
                if B_tok <= 0:
                    return torch.zeros(P_shared, dtype=torch.bool)
                keep, _ = select_tokens(scores_tok, torch.zeros(P_shared), B_tok, 1.0,
                                        N_LAYERS, N_KV_HEADS, sink_forced_tok)
                return keep[0, 0].clone()

            sizes_needed = set(C_list) | set(core_only) | {c for _, c, _ in rand_cells}
            per_signal = {}
            for sig, sc in core_scores.items():
                sc_tok = sc.amax(dim=(0, 1)) if sig == "kvzip" else sc.mean(dim=(0, 1))
                per_signal[sig] = {
                    "cores": {C: keep_head(C, sc) for C in sizes_needed},
                    "colds": {B: keep_head(B, sc) for B in B_list if B < 1},
                    "cores_tok": {C: keep_tok(C, sc_tok) for _, C, _ in tok_cells},
                    "colds_tok": {B: keep_tok(B, sc_tok) for B, _, _ in tok_cells if B < 1},
                }
            cores_rand = {C: keep_head(C, rand_scores) for _, C, _ in rand_cells}

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
                query_prefix = q_mean[:, :, :P_shared].clone()
                query_tok = query_prefix.mean(dim=(0, 1))
                base = {"run_id": run_id, "model": a.model, "dataset": row["dataset"],
                        "sample_id": row["sample_id"], "question_id": q["question_id"],
                        "gold": golds, "n_prefix": P_shared, "n_visual": int(len(vis)),
                        "n_question": n_q, "prefix_triples": prefix_triples}
                f.write(json.dumps({**base, "condition_id": "FULL_KV", "selector": "full",
                                    "core_frac": 1.0, "delta_frac": 0.0, "cold_frac": 1.0,
                                    "hot_triples": PER_TOKEN * P, "fetch_triples": 0,
                                    "hot_bytes": kv_bytes(PER_TOKEN * P, HEAD_DIM),
                                    "prediction": pred_full, "anls": anls(pred_full, golds),
                                    "em": exact_match(pred_full, golds)},
                                   ensure_ascii=False) + "\n")

                def run(cond_id, selector, keep_prefix, sig, B, C, D, gran, core_triples, n_delta):
                    keep = torch.ones((N_LAYERS, N_KV_HEADS, P), dtype=torch.bool)
                    keep[:, :, :P_shared] = keep_prefix
                    pred = greedy_generate_perhead(model, processor, ins, keep, P_shared,
                                                   max_new_tokens=a.max_new_tokens)
                    comp = kept_composition(keep_prefix, vis, N_SINK)
                    stored = comp["kept_triples"]
                    rec = {**base, "condition_id": cond_id, "selector": selector,
                           "core_signal": sig, "granularity": gran,
                           "cold_frac": B, "core_frac": C, "delta_frac": D,
                           "stored_triples": stored, "core_triples": int(core_triples),
                           "fetch_triples": int(n_delta), "fetch_bytes": kv_bytes(n_delta, HEAD_DIM),
                           "hot_triples": stored + PER_TOKEN * n_q,
                           "hot_bytes": kv_bytes(stored + PER_TOKEN * n_q, HEAD_DIM),
                           "index_bytes": index_bytes(N_LAYERS, N_KV_HEADS, P_shared, gran),
                           "prediction": pred, "anls": anls(pred, golds),
                           "em": exact_match(pred, golds),
                           "loyalty": float(normalize_text(pred) == normalize_text(pred_full)),
                           **{k: v for k, v in comp.items() if k != "kept_triples"}}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

                def with_delta_head(core_keep, D, cold_keep):
                    T_D = int(round(D * prefix_triples))
                    if T_D <= 0:
                        return core_keep.clone(), 0
                    qs_ = query_prefix
                    if cold_keep is not None:
                        assert bool((core_keep & ~cold_keep).sum() == 0)
                        assert T_D <= int((cold_keep & ~core_keep).sum())
                        qs_ = query_prefix.masked_fill(~cold_keep, float("-inf"))
                    keep, sel = select_triples(torch.full_like(qs_, float("-inf")), qs_,
                                               int(core_keep.sum()) + T_D, 0.0, core_keep)
                    return keep, sel.query_count

                def with_delta_tok(core_tok_keep, D, cold_tok):
                    B_tok = int(round(D * prefix_triples)) // PER_TOKEN
                    if B_tok <= 0:
                        return core_tok_keep[None, None, :].expand(N_LAYERS, N_KV_HEADS, P_shared).clone(), 0
                    qt = query_tok
                    if cold_tok is not None:
                        assert bool((core_tok_keep & ~cold_tok).sum() == 0)
                        assert B_tok <= int((cold_tok & ~core_tok_keep).sum())
                        qt = query_tok.masked_fill(~cold_tok, float("-inf"))
                    keep, sel = select_tokens(torch.full((P_shared,), float("-inf")), qt,
                                              int(core_tok_keep.sum()) + B_tok, 0.0,
                                              N_LAYERS, N_KV_HEADS, core_tok_keep)
                    return keep, sel.query_count

                def btag(B):
                    return "" if B >= 1 else f"B{_fmt(B*100)}"

                for sig, ps in per_signal.items():
                    # core만 (delta 0): B 무관, KVzip 기준선 겸용
                    for C in sorted(set(c for c in C_list if c > 0) | set(core_only)):
                        run(f"wd_{sig}_C{_fmt(C*100)}_D0", "core_only", ps["cores"][C], sig, 1.0, C,
                            0.0, "head", ps["cores"][C].sum(), 0)
                    for B in B_list:
                        cold = ps["colds"].get(B)
                        for C in C_list:
                            if C > B + 1e-9:
                                continue
                            for D in D_list:
                                if D == 0 or C + D > B + 1e-9:
                                    continue
                                keep, n_delta = with_delta_head(ps["cores"][C], D, cold)
                                run(f"wd_{sig}{btag(B)}_C{_fmt(C*100)}_D{_fmt(D*100)}", "core_delta",
                                    keep, sig, B, C, D, "head", ps["cores"][C].sum(), n_delta)
                    for B, C, D in rand_cells:
                        cold = ps["colds"].get(B)
                        rc = cores_rand[C] & cold if cold is not None else cores_rand[C]
                        # 무작위 core도 DRAM 보관분 안에서만 뽑히도록 보관분과 교집합 후 크기 보정
                        if cold is not None and int(rc.sum()) < int(cores_rand[C].sum()):
                            need = int(cores_rand[C].sum()) - int(rc.sum())
                            sc = rand_scores.masked_fill(~(cold & ~rc), float("-inf"))
                            extra, _ = select_triples(sc, torch.zeros_like(sc), int(rc.sum()) + need,
                                                      1.0, rc)
                            rc = extra
                        keep, n_delta = with_delta_head(rc, D, cold)
                        run(f"wdR_{sig}{btag(B)}_C{_fmt(C*100)}_D{_fmt(D*100)}", "random_core_delta",
                            keep, sig, B, C, D, "head", rc.sum(), n_delta)
                    for B, C, D in tok_cells:
                        cold_t = ps["colds_tok"].get(B)
                        keep, n_delta = with_delta_tok(ps["cores_tok"][C], D, cold_t)
                        run(f"wdT_{sig}{btag(B)}_C{_fmt(C*100)}_D{_fmt(D*100)}", "core_delta_token",
                            keep, sig, B, C, D, "token", int(ps["cores_tok"][C].sum()) * PER_TOKEN,
                            n_delta)
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} prefix={P_shared} vis={len(vis)} "
                  f"desc_tokens={n_desc} {time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
