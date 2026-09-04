"""Dual-prefill 전체 KV 판 — 프롬프트의 모든 token × 모든 층 × 모든 KV head.

무엇이 달라졌나 (시각 token만 압축하던 core_delta_sweep 대비)
  후보     : system 문구, vision 경계, 시각 token, 질문, assistant 머리말 전부 (보호 없음이 기본)
  단위     : (층, KV head, token) 세 짝. head 단위(H)는 층·head마다 다른 token을 남기고,
             token 단위(T)는 모든 층·head가 같은 token을 남긴다 (공통 마스크). 둘 다 실행.
  예산     : 프롬프트 전체 KV의 세 짝 수 × 비율 (K·V 본체 byte와 비례). 조건 간 동일.
  시점     : 기본 'generation' = 이미지+질문을 한 번 prefill해 캐시를 만든 뒤 잘라내고, 생성
             행만 남은 캐시를 본다 (§8의 주장 범위: prefill 뒤 active cache·decode 절감).
             --row-start question 이면 질문 행부터 차단 (이전 시각 전용 판·D6 사다리 규약).
점수 (생성/reconstruction 없이 정확히 두 signal prefill에서 측정)
  image : system+image 경계까지만 한 번 prefill하고, image 행들이 각 prefix 열에 준 attention
          평균을 (층, KV head)별로 집계. 뒤 text 열에는 이 점수가 정의되지 않는다.
  joint : 같은 image+기존 text prefix를 한 번 prefill하고, text-prefix 행들이 전체 prompt 열에
          준 attention 평균을 (층, KV head)별로 집계한다.
결합 : 각 순위에서 정해진 몫을 독립 top-k한 뒤 합집합·중복 제거하고, 빈 예산은 joint 순위로
       채워 최종 세 짝 수를 정확히 B로 맞춘다. 실제 K/V 값은 joint prefill cache에서 취한다.
비교군: random(세 짝 무작위), random_sink(앞 4 token 보호 + 무작위), uniform_tok(등간격 token).

실행 (smoke):
  python -m vlm_diagnosis.exps.core_delta_full_kv --limit 1 --device cuda:0
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
    per_head_column_stats, select_dual_prefill_triples,
    select_dual_prefill_tokens, select_triples, uniform_token_keep,
    greedy_generate_perhead, kept_composition, kv_bytes, index_bytes)
from vlm_diagnosis.core import signals as S
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS, BRIEF

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
N_SINK = 4


def _fmt(x: float) -> str:
    return f"{x:g}"


@torch.no_grad()
def image_prefill_stats(model, processor, img, device):
    """Run one true image-only prefix prefill and return its attention score.

    A dummy text value is used only to materialize the model's canonical chat
    template; input IDs are sliced through ``vision_end`` before the forward,
    so no user text or assistant header enters this prefill.  Returned IDs are
    used to fail closed if a joint prompt does not share the exact prefix.
    """
    ins = S.vlm_inputs(processor, img, "x", device)
    sp = token_spans(ins["input_ids"], model.config)
    prefix_len = sp["vis_end"] + 2  # visual tokens plus the closing vision boundary
    prefix_ids = ins["input_ids"][:, :prefix_len]
    attn = torch.ones_like(prefix_ids)
    pos = mrope_position_ids(model, prefix_ids, ins["image_grid_thw"], attn)
    with QKCapture() as cap:
        model(input_ids=prefix_ids, attention_mask=attn, position_ids=pos,
              pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"],
              use_cache=False)
        expected_layers, expected_heads, _ = kv_dims(model)
        if len(cap.qk) != expected_layers:
            raise RuntimeError(
                f"image-only prefill captured {len(cap.qk)} attention layers; "
                f"expected {expected_layers}"
            )
        mean, _ = per_head_column_stats(
            cap.qk, int(sp["visual"].min()), prefix_len)
    if mean.shape[:2] != (expected_layers, expected_heads):
        raise RuntimeError(
            f"image-only score shape {tuple(mean.shape)} disagrees with "
            f"KV dimensions {(expected_layers, expected_heads)}"
        )
    return mean, prefix_ids.detach().cpu(), prefix_len - int(sp["visual"].min())


def align_image_prefill_score(image_score, prompt_len: int):
    """Place an image-prefix score in the full joint-prompt coordinate space."""
    score = torch.as_tensor(image_score)
    if score.ndim != 3:
        raise ValueError(f"image score must have shape (layers, heads, prefix), got {score.shape}")
    prefix_len = score.shape[-1]
    if prefix_len > prompt_len:
        raise ValueError(
            f"image prefix ({prefix_len}) is longer than joint prompt ({prompt_len})"
        )
    aligned = torch.full((*score.shape[:-1], prompt_len), float("-inf"), dtype=score.dtype)
    aligned[..., :prefix_len] = score.cpu()
    eligible = torch.zeros_like(aligned, dtype=torch.bool)
    eligible[..., :prefix_len] = True
    return aligned, eligible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budgets", default="0.05,0.1", help="프롬프트 전체 KV 세 짝 수 대비 비율")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1",
                    help="image-only prefill 몫; 0=joint-only, 1=image-only")
    ap.add_argument("--granularity", default="head,token", help="쉼표 목록: head,token")
    ap.add_argument("--row-start", default="generation", choices=["generation", "question"],
                    help="generation: prefill 후 잘라내기(생성 행만 제한). question: 질문 행부터 제한")
    ap.add_argument("--protect", default="none", choices=["none", "sink", "text", "sink+text"],
                    help="항상 보존(예산에 포함)할 token: sink=앞 4개, text=시각 아닌 모든 token")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--eval-questions-per-doc", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/smoke/core_delta_full_kv.jsonl")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    budgets = [float(x) for x in a.budgets.split(",")]
    alphas = sorted({float(x) for x in a.alphas.split(",")})
    grans = [g.strip() for g in a.granularity.split(",") if g.strip()]
    if not budgets or any(B <= 0 or B > 1 for B in budgets):
        raise ValueError("every budget must be in (0,1]")
    if not alphas or any(al < 0 or al > 1 for al in alphas):
        raise ValueError("every alpha/image fraction must be in [0,1]")
    if not grans or set(grans) - {"head", "token"}:
        raise ValueError("granularity must contain only head and/or token")
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
    run_id = f"fk-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

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
            "run_id": run_id, "stage": "DUAL_PREFILL_FULL_KV", "run_kind": "smoke",
            "model": a.model, "manifest_path": a.manifest,
            "kv_dims": {"layers": N_LAYERS, "kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM},
            "budgets_frac_of_prompt_kv": budgets, "alphas": alphas, "granularity": grans,
            "row_start": a.row_start, "protect": a.protect,
            "candidates": "all prompt tokens (system, vision bounds, visual, question, "
                          "assistant header) x all layers x all KV heads",
            "signal_prefills": 2,
            "image_prefill_score": "mean attention received from image rows during one "
                                     "system+image-only prefix prefill",
            "joint_prefill_score": "mean attention received from existing text-prefix and "
                                     "assistant-header rows during one image+text prefill",
            "selection": "independent image/joint top-k union; de-duplicate and backfill from "
                         "joint ranking; |keep|==budget",
            "cache_provenance": "selection masks are combined; retained K/V values come from "
                                "the canonical joint image+text prefill (K and V stay paired)",
            "budget_unit": "triples = layers*kv_heads*tokens; bytes = triples*2*head_dim*2",
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
            image_prefix_score, image_prefix_ids, n_image_rows = image_prefill_stats(
                model, processor, img, a.device)
            P_shared = image_prefix_ids.shape[1]
            sample_seed = zlib.crc32(f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF
            g = torch.Generator().manual_seed(sample_seed)

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
                golds = q["answers"]
                # FULL 예측의 prefill에서 query 점수를 함께 캡처 (prefill 1회 = 방법의 실제 비용)
                with QKCapture() as cap:
                    pred_full = greedy_generate_masked(
                        model, processor, ins, max_new_tokens=a.max_new_tokens)
                    if len(cap.qk) < N_LAYERS:
                        raise RuntimeError(
                            f"joint prefill captured {len(cap.qk)} attention layers; "
                            f"expected at least {N_LAYERS}"
                        )
                    joint_mean, _ = per_head_column_stats(
                        cap.qk[:N_LAYERS], P_shared, P)
                del cap
                image_score, image_eligible = align_image_prefill_score(
                    image_prefix_score, P)
                joint_score = joint_mean                              # (L, Hkv, P)
                image_tok = image_score.mean(dim=(0, 1))
                joint_tok = joint_score.mean(dim=(0, 1))
                image_eligible_tok = image_eligible[0, 0]
                is_vis = torch.zeros(P, dtype=torch.bool); is_vis[vis.cpu()] = True
                forced_tok = torch.zeros(P, dtype=torch.bool)
                if "sink" in a.protect:
                    forced_tok[:N_SINK] = True
                if "text" in a.protect:
                    forced_tok |= ~is_vis
                forced = forced_tok[None, None, :].expand(N_LAYERS, N_KV_HEADS, P).clone()
                n_text = int((~is_vis).sum())
                full_triples = PER_TOKEN * P
                row_start = P if a.row_start == "generation" else P_shared
                base = {"run_id": run_id, "model": a.model, "dataset": row["dataset"],
                        "sample_id": row["sample_id"], "question_id": q["question_id"],
                        "gold": golds, "n_prompt": P, "n_visual": int(len(vis)),
                        "n_text": n_text, "row_start": a.row_start}
                f.write(json.dumps({**base, "condition_id": "FULL_KV", "selector": "full",
                                    "keep_ratio_target": 1.0, "kept_triples": full_triples,
                                    "payload_bytes": kv_bytes(full_triples, HEAD_DIM),
                                    "prediction": pred_full, "anls": anls(pred_full, golds),
                                    "em": exact_match(pred_full, golds)},
                                   ensure_ascii=False) + "\n")

                def run(cond_id, selector, keep, B, gran, extra=None):
                    pred = greedy_generate_perhead(model, processor, ins, keep, row_start,
                                                   max_new_tokens=a.max_new_tokens)
                    comp = kept_composition(keep, vis, N_SINK)
                    rec = {**base, "condition_id": cond_id, "selector": selector,
                           "keep_ratio_target": B, "granularity": gran,
                           "payload_bytes": kv_bytes(comp["kept_triples"], HEAD_DIM),
                           "index_bytes": index_bytes(N_LAYERS, N_KV_HEADS, P, gran),
                           "budget_utilization": round(comp["kept_triples"] / (B * full_triples), 4),
                           "prediction": pred, "anls": anls(pred, golds),
                           "em": exact_match(pred, golds),
                           "loyalty": float(normalize_text(pred) == normalize_text(pred_full)),
                           **comp}
                    if extra:
                        rec.update(extra)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

                for B in budgets:
                    T = int(round(B * full_triples))
                    B_tok = T // PER_TOKEN
                    if int(forced.sum()) > T or int(forced_tok.sum()) > B_tok:
                        raise ValueError(
                            f"--protect={a.protect} requires more entries than budget {B:g}"
                        )
                    Bt = f"B{_fmt(B * 100)}"
                    if not a.no_controls:
                        rnd = torch.rand((N_LAYERS, N_KV_HEADS, P), generator=g)
                        keep, sel = select_triples(torch.full_like(rnd, float("-inf")), rnd, T,
                                                   0.0, forced)
                        run(f"random@{Bt}", "random", keep, B, "head",
                            {"selection": sel.as_dict()})
                        forced_sink = forced.clone(); forced_sink[:, :, :N_SINK] = True
                        if int(forced_sink.sum()) > T:
                            raise ValueError(f"sink protection exceeds budget {B:g}")
                        keep, sel = select_triples(torch.full_like(rnd, float("-inf")), rnd, T,
                                                   0.0, forced_sink)
                        run(f"random_sink@{Bt}", "random_sink", keep, B, "head",
                            {"selection": sel.as_dict()})
                        keep = uniform_token_keep(P, B_tok, N_LAYERS, N_KV_HEADS, forced_tok)
                        run(f"uniform_tok@{Bt}", "uniform_tok", keep, B, "token")
                    for gran in grans:
                        tag = "H" if gran == "head" else "T"
                        for al in alphas:
                            if gran == "head":
                                keep, sel = select_dual_prefill_triples(
                                    image_score, joint_score, T, al, forced,
                                    image_eligible=image_eligible)
                            else:
                                keep, sel = select_dual_prefill_tokens(
                                    image_tok, joint_tok, B_tok, al,
                                    N_LAYERS, N_KV_HEADS, forced_tok,
                                    image_eligible_tok=image_eligible_tok)
                            run(f"cd_dualprefill{tag}_a{_fmt(al)}@{Bt}",
                                f"cd_dualprefill{tag}", keep, B, gran,
                                {"alpha": al, "image_fraction": al,
                                 "selection": sel.as_dict()})
                        if not a.no_controls:
                            # image-only endpoint with sink protection: separate the
                            # two-prefill signal from missing anchor-token effects.
                            fs_tok = forced_tok.clone(); fs_tok[:N_SINK] = True
                            if gran == "head":
                                fs = fs_tok[None, None, :].expand(N_LAYERS, N_KV_HEADS, P).clone()
                                keep, sel = select_dual_prefill_triples(
                                    image_score, joint_score, T, 1.0, fs,
                                    image_eligible=image_eligible)
                            else:
                                keep, sel = select_dual_prefill_tokens(
                                    image_tok, joint_tok, B_tok, 1.0,
                                    N_LAYERS, N_KV_HEADS, fs_tok,
                                    image_eligible_tok=image_eligible_tok)
                            run(f"image_sink{tag}@{Bt}", f"image_sink{tag}", keep, B, gran,
                                {"alpha": 1.0, "image_fraction": 1.0, "protect": "sink",
                                 "selection": sel.as_dict()})
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} P={P} vis={len(vis)} "
                  f"image_rows={n_image_rows} {time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
