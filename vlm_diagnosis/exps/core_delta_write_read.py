"""Dual-prefill 쓰기/읽기 판 — 두 prefill의 중요 항목만 합쳐 사용한다.

설정 (두 저장소 구조, VLM_idea.md §8 후반·§11)
  쓰기 시점 (질문 없음) : system+image 경계까지만 정확히 한 번 prefill하고, image 행 attention
                          으로 중요한 **core**(크기 C)를 고른다. 생성/reconstruction은 없다.
  읽기 시점 (질문 도착) : image+기존 text prefix를 정확히 한 번 prefill하고, text 행 attention
                          으로 중요한 **delta**(크기 D)를 고른다. core와 겹치는 항목은 제거하고
                          joint 순위의 다음 항목으로 채워 정확히 C+D를 유지한다.
  답 생성               : 질문 행과 생성 행 모두 core ∪ delta만 본다 (질문 도착 전에 잘라낸 규약).
                          질문 token 자체는 읽기 시점에 새로 만들어지므로 항상 보존하고, 저장
                          예산(C, D)에는 넣지 않되 뜨거운 캐시 크기에는 포함해 기록한다.

무엇을 비교하나 — 격자 (C, D) 에서
  (0, D)  delta만: 읽기 시점에 질문 점수로만 가져옴 (core의 가치를 재는 기준선)
  (C, 0)  core만 : image-only prefill 중요도만 (delta의 가치를 재는 기준선)
  (C, D)  core–delta
  무작위 core + delta : core의 '내용'이 중요한지, 단순히 개수가 늘어난 것인지 분리
  token 단위 일부 격자점 : 물리적으로 잘라낼 수 있는 공통 마스크 판

가정 (기록): delta 선택은 전체 접두 KV에 대한 정확한 질문 attention 점수를 쓴다. 차가운
저장소에서 점수를 계산하는 비용(selector overhead)은 이 실험에서 모델링하지 않는다.
sink(앞 4 token)는 core에 항상 포함한다(StreamingLLM 관례). C·D 비율은 접두 KV 세 짝 수 기준.

실행 (smoke):
  python -m vlm_diagnosis.exps.core_delta_write_read --limit 1 --device cuda:0
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
from vlm_diagnosis.core.attnstat import QKCapture
from vlm_diagnosis.core.metrics import anls, exact_match, normalize_text
from vlm_diagnosis.core.kv_select import (
    per_head_column_stats, select_triples, select_tokens, greedy_generate_perhead,
    kept_composition, kv_bytes, index_bytes)
from vlm_diagnosis.core import signals as S
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS, BRIEF
from vlm_diagnosis.exps.core_delta_full_kv import image_prefill_stats, N_SINK

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _fmt(x: float) -> str:
    return f"{x:g}"


def _parse_grid(s):
    return sorted({float(x) for x in s.split(",")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--core-sizes", default="0,0.025,0.05,0.1",
                    help="C: 쓰기 시점 core 크기 (접두 KV 세 짝 수 대비 비율)")
    ap.add_argument("--delta-sizes", default="0,0.01,0.025,0.05",
                    help="D: 읽기 시점 delta 크기 (접두 KV 세 짝 수 대비 비율)")
    ap.add_argument("--cold-budgets", default="1",
                    help="B: 쓰기 시점에 DRAM에 남길 양(접두 KV 대비 비율), 나머지는 삭제. "
                         "1 = 전부 보관(참고치). 예: 0.1,0.2,0.4. core⊂B, delta는 B 안에서만.")
    ap.add_argument("--core-only-sizes", default="",
                    help="추가로 돌릴 image-prefill core만(delta 0) 크기 (예: 0.035,0.06)")
    ap.add_argument("--random-core-cells", default="1:0.05:0.01,1:0.05:0.025",
                    help="무작위 core 대조를 돌릴 (B:C:D) 칸")
    ap.add_argument("--token-cells", default="1:0:0.01,1:0:0.025,1:0.05:0.01,1:0.05:0.025",
                    help="token 단위(공통 마스크)로도 돌릴 (B:C:D) 칸")
    ap.add_argument("--no-sink-protect", action="store_true",
                    help="core에 sink 4 token을 강제 포함하지 않음 (ablation)")
    ap.add_argument("--eval-questions-per-doc", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/smoke/core_delta_write_read.jsonl")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    C_list = _parse_grid(a.core_sizes)
    D_list = _parse_grid(a.delta_sizes)
    B_list = _parse_grid(a.cold_budgets)
    core_only = _parse_grid(a.core_only_sizes) if a.core_only_sizes else []
    rand_cells = [tuple(float(v) for v in c.split(":")) for c in a.random_core_cells.split(",") if c]
    tok_cells = [tuple(float(v) for v in c.split(":")) for c in a.token_cells.split(",") if c]
    for B, C, D in rand_cells + tok_cells:
        assert C + D <= B + 1e-9, f"cell B{B}:C{C}:D{D} — core+delta가 DRAM 보관량을 넘음"
    for B in B_list:
        for C in C_list:
            for D in D_list:
                assert C > B + 1e-9 or C + D <= B + 1e-9 or D == 0, \
                    f"B{B} C{C} D{D}: delta가 B 안에 들어갈 자리가 없음"
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
    run_id = f"wr-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

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
            "run_id": run_id, "stage": "DUAL_PREFILL_WRITE_READ", "run_kind": "smoke",
            "model": a.model, "manifest_path": a.manifest,
            "kv_dims": {"layers": N_LAYERS, "kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM},
            "core_sizes": C_list, "delta_sizes": D_list, "cold_budgets": B_list,
            "core_only_sizes": core_only, "random_core_cells": rand_cells,
            "token_cells": tok_cells, "sink_protect_in_core": not a.no_sink_protect,
            "signal_prefills": "one image-only prefill per image plus one joint image+existing-text "
                               "prefill per evaluated text prefix; no reconstruction generation",
            "cold_tier": "B<1: image-prefill top-B by the same core score kept in DRAM, rest "
                         "DELETED; core = top-C of the same ranking (nested); delta chosen only "
                         "inside B. B=1: everything retained (reference)",
            "stored_prefix": "system + vision_start + visual + vision_end (everything before the question)",
            "write_time_core": "per (layer, kv_head) mean attention received from image rows "
                               "during the image-only prefill; sink tokens forced into core",
            "read_time_delta": "top-D by mean question-row attention among prefix triples not in core "
                               "(exact scores over full prefix; selector cost not modeled)",
            "semantics": "question rows and generated rows see only core ∪ delta (row_start = "
                         "first question token); question tokens always kept, not counted in C/D",
            "cache_provenance": "core/delta are selection masks; retained K/V values use the "
                                "canonical joint prefill coordinates and K/V pairs stay together",
            "budget_unit": "fraction of prefix triples = layers*kv_heads*prefix_tokens",
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
            # ---- 쓰기 시점: text 없이 image-only prefill을 정확히 한 번 실행
            core_prefix, image_prefix_ids, n_image_rows = image_prefill_stats(
                model, processor, img, a.device)
            P_shared = image_prefix_ids.shape[1]
            core_tok = core_prefix.mean(dim=(0, 1))
            prefix_triples = PER_TOKEN * P_shared
            sample_seed = zlib.crc32(f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF
            g = torch.Generator().manual_seed(sample_seed)
            rand_core_scores = torch.rand((N_LAYERS, N_KV_HEADS, P_shared), generator=g)
            sink_forced = torch.zeros((N_LAYERS, N_KV_HEADS, P_shared), dtype=torch.bool)
            sink_forced_tok = torch.zeros(P_shared, dtype=torch.bool)
            if not a.no_sink_protect:
                sink_forced[:, :, :N_SINK] = True
                sink_forced_tok[:N_SINK] = True

            def core_keep_head(C, scores):
                T = int(round(C * prefix_triples))
                if T <= 0:
                    return torch.zeros((N_LAYERS, N_KV_HEADS, P_shared), dtype=torch.bool)
                keep, _ = select_triples(scores, torch.zeros_like(scores), T, 1.0, sink_forced)
                return keep

            def core_keep_tok(C):
                B_tok = int(round(C * prefix_triples)) // PER_TOKEN
                if B_tok <= 0:
                    return torch.zeros(P_shared, dtype=torch.bool)
                keep, _ = select_tokens(core_tok, torch.zeros(P_shared), B_tok, 1.0,
                                        N_LAYERS, N_KV_HEADS, sink_forced_tok)
                return keep[0, 0].clone()

            core_sizes_needed = set(C_list) | set(core_only) | {c for _, c, _ in rand_cells}
            cores_head = {C: core_keep_head(C, core_prefix) for C in core_sizes_needed}
            colds_head = {B: core_keep_head(B, core_prefix) for B in B_list if B < 1}
            cores_rand = {C: core_keep_head(C, rand_core_scores) for _, C, _ in rand_cells}
            cores_tok = {C: core_keep_tok(C) for _, C, _ in tok_cells}
            colds_tok = {B: core_keep_tok(B) for B, _, _ in tok_cells if B < 1}

            for q in eval_qs:
                q_text = q["question"] + BRIEF
                ins = S.vlm_inputs(processor, img, q_text, a.device)
                sp = token_spans(ins["input_ids"], model.config)
                vis, vis_end, P = sp["visual"], sp["vis_end"], int(sp["L"])
                if vis_end + 2 != P_shared or not torch.equal(
                        ins["input_ids"][:, :P_shared].cpu(), image_prefix_ids):
                    raise RuntimeError(
                        "image-only and image+text prefills do not share an exact image prefix"
                    )
                n_q = P - P_shared                                   # 질문 + 머리말 token 수
                golds = q["answers"]
                # ---- 읽기 시점: FULL 예측의 prefill에서 질문 점수를 캡처 (질문 행 → 접두 열)
                with QKCapture() as cap:
                    pred_full = greedy_generate_masked(
                        model, processor, ins, max_new_tokens=a.max_new_tokens)
                    q_mean, _ = per_head_column_stats(cap.qk[:N_LAYERS], P_shared, P)
                del cap
                query_prefix = q_mean[:, :, :P_shared].clone()        # (L, H, P_shared)
                query_tok = query_prefix.mean(dim=(0, 1))
                base = {"run_id": run_id, "model": a.model, "dataset": row["dataset"],
                        "sample_id": row["sample_id"], "question_id": q["question_id"],
                        "gold": golds, "n_prefix": P_shared, "n_visual": int(len(vis)),
                        "n_question": n_q, "prefix_triples": prefix_triples}
                f.write(json.dumps({**base, "condition_id": "FULL_KV", "selector": "full",
                                    "core_frac": 1.0, "delta_frac": 0.0,
                                    "hot_triples": PER_TOKEN * P, "fetch_triples": 0,
                                    "hot_bytes": kv_bytes(PER_TOKEN * P, HEAD_DIM),
                                    "prediction": pred_full, "anls": anls(pred_full, golds),
                                    "em": exact_match(pred_full, golds)},
                                   ensure_ascii=False) + "\n")

                def run(cond_id, selector, keep_prefix, C, D, gran, extra=None):
                    keep = torch.ones((N_LAYERS, N_KV_HEADS, P), dtype=torch.bool)
                    keep[:, :, :P_shared] = keep_prefix
                    pred = greedy_generate_perhead(model, processor, ins, keep, P_shared,
                                                   max_new_tokens=a.max_new_tokens)
                    comp = kept_composition(keep_prefix, vis, N_SINK)
                    stored = comp["kept_triples"]
                    rec = {**base, "condition_id": cond_id, "selector": selector,
                           "granularity": gran, "core_frac": C, "delta_frac": D,
                           "stored_triples": stored,
                           "hot_triples": stored + PER_TOKEN * n_q,
                           "hot_bytes": kv_bytes(stored + PER_TOKEN * n_q, HEAD_DIM),
                           "index_bytes": index_bytes(N_LAYERS, N_KV_HEADS, P_shared, gran),
                           "prediction": pred, "anls": anls(pred, golds),
                           "em": exact_match(pred, golds),
                           "loyalty": float(normalize_text(pred) == normalize_text(pred_full)),
                           **{k: v for k, v in comp.items() if k != "kept_triples"}}
                    if extra:
                        rec.update(extra)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

                def with_delta_head(core_keep, D, cold_keep=None):
                    """Image set과 joint top-D를 union하고 겹친 수만큼 joint에서 backfill.

                    구현상 ``core_keep``을 forced로 두고 joint 상위 D개를 고르면 독립
                    top-D 합집합의 중복을 joint 다음 순위로 채운 것과 정확히 같다.
                    ``cold_keep``이 있으면 joint 후보는 그 DRAM 보관분으로 제한한다.
                    """
                    T_D = int(round(D * prefix_triples))
                    if T_D <= 0:
                        return core_keep.clone(), 0
                    qs = query_prefix
                    if cold_keep is not None:
                        assert bool((core_keep & ~cold_keep).sum() == 0), "core가 DRAM 보관분 밖에 있음"
                        avail = int((cold_keep & ~core_keep).sum())
                        assert T_D <= avail, (T_D, avail)
                        qs = query_prefix.masked_fill(~cold_keep, float("-inf"))
                    keep, sel = select_triples(torch.full_like(qs, float("-inf")), qs,
                                               int(core_keep.sum()) + T_D, 0.0, core_keep)
                    if cold_keep is not None:
                        assert bool((keep & ~cold_keep).sum() == 0)
                    return keep, sel.query_count

                def with_delta_tok(core_tok_keep, D, cold_tok=None):
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

                # core만 (delta 0) 칸: DRAM 보관량 B와 무관 → 한 번만 실행
                for C in sorted(set(c for c in C_list if c > 0) | set(core_only)):
                    run(f"wr_C{_fmt(C*100)}_D0", "core_only_head", cores_head[C], C, 0.0, "head",
                        {"cold_frac": 1.0, "core_triples": int(cores_head[C].sum()),
                         "fetch_triples": 0, "fetch_bytes": 0})
                for B in B_list:
                    cold = colds_head.get(B)                       # None = 전부 보관
                    for C in C_list:
                        if C > B + 1e-9:
                            continue
                        for D in D_list:
                            if D == 0 or C + D > B + 1e-9:
                                continue
                            keep, n_delta = with_delta_head(cores_head[C], D, cold)
                            run(f"wr{btag(B)}_C{_fmt(C*100)}_D{_fmt(D*100)}", "core_delta_head",
                                keep, C, D, "head",
                                {"cold_frac": B, "core_triples": int(cores_head[C].sum()),
                                 "fetch_triples": n_delta, "fetch_bytes": kv_bytes(n_delta, HEAD_DIM)})
                for B, C, D in rand_cells:
                    cold = colds_head.get(B)
                    keep, n_delta = with_delta_head(cores_rand[C], D, cold)
                    run(f"wrR{btag(B)}_C{_fmt(C*100)}_D{_fmt(D*100)}", "random_core_delta_head",
                        keep, C, D, "head",
                        {"cold_frac": B, "core_triples": int(cores_rand[C].sum()),
                         "fetch_triples": n_delta, "fetch_bytes": kv_bytes(n_delta, HEAD_DIM)})
                for B, C, D in tok_cells:
                    cold_t = colds_tok.get(B)
                    keep, n_delta = with_delta_tok(cores_tok[C], D, cold_t)
                    run(f"wrT{btag(B)}_C{_fmt(C*100)}_D{_fmt(D*100)}", "core_delta_token",
                        keep, C, D, "token",
                        {"cold_frac": B, "core_triples": int(cores_tok[C].sum()) * PER_TOKEN,
                         "fetch_triples": n_delta, "fetch_bytes": kv_bytes(n_delta, HEAD_DIM)})
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} prefix={P_shared} vis={len(vis)} "
                  f"image_rows={n_image_rows} {time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
