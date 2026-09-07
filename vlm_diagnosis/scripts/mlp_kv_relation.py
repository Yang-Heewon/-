"""MLP 통계와 KV 의 관계 자료 — 남기는 실험이 아니라 관측만.

token (층 l, 위치 i) 마다
  MLP 쪽 : mlp_norm, R, D, hidden_rel, hidden_cos           (단일 prefill 수집기)
  KV 쪽  : K norm, V norm (head 평균 및 head별)
           attn_pre  = prefill 중 뒤 조각 행들이 받은 attention (읽는 목적 없음)
           attn_desc = 설명문 생성 행이 받은 attention        (재구성 = 작동하는 필요도 대리)
           attn_q    = 실제 질문 행이 받은 attention (질문 3개 평균) = 진짜 읽기 필요도
관계:
  (a) 같은 층 상관 (Spearman, 보호 token 제외, 층별·전체)
  (b) 층 이동 상관: MLP[l] vs K/V norm[l+1], vs attn[l+1] (MLP 출력은 다음 층 KV 에 들어감)
  (c) MLP norm 10분위별 attn_q 평균 (어느 분위든 다른가)
  (d) 시각 / 비시각 token 분포
  (e) K/V norm 과 attn_q 의 관계 (MLP 없이 KV 자체가 읽힘을 예측하나)

  python -m vlm_diagnosis.scripts.mlp_kv_relation --limit 30 --device cuda:0
"""
import argparse, json, os
from collections import defaultdict

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.context_only_cache import prefill_context
from vlm_diagnosis.core.kv_select import per_head_column_stats
from vlm_diagnosis.core.attnstat import QKCapture
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.signals import vlm_inputs
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.static_pair_select import spearman_avg_rank, protected_positions
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS, BRIEF
from vlm_diagnosis.exps.core_delta_dram import REPEAT_PROMPT

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


@torch.no_grad()
def collect(model, processor, img, qs, device):
    pre = prefill_context(model, processor, img, device, collect_dynamics=True, capture_qk=True)
    P = pre["spans"]["P"]; v0 = int(pre["spans"]["visual"].min())
    dyn = pre["dynamics"]
    attn_pre, _ = per_head_column_stats(pre["qk"], v0, P)                       # (L,H,P)
    knorm = torch.stack([k[0].float().norm(dim=-1).cpu() for k, _ in pre["kv"]])
    vnorm = torch.stack([v[0].float().norm(dim=-1).cpu() for _, v in pre["kv"]])
    pre["qk"] = None
    # 설명문 재구성 행
    ins1 = vlm_inputs(processor, img, REPEAT_PROMPT, device)
    gen = model.generate(**{k: v for k, v in ins1.items()}, max_new_tokens=96, do_sample=False)
    P1, Lg = ins1["input_ids"].shape[1], gen.shape[1]
    attn = torch.ones(1, Lg, dtype=torch.long, device=device)
    pos = mrope_position_ids(model, gen, ins1["image_grid_thw"], attn)
    with QKCapture() as cap:
        model(input_ids=gen, attention_mask=attn, position_ids=pos, pixel_values=ins1["pixel_values"],
              image_grid_thw=ins1["image_grid_thw"], use_cache=False)
        desc_mean, desc_peak = per_head_column_stats(cap.qk, P1, Lg)
    del cap
    # 실제 질문 행 (질문 3개 평균)
    aq = []
    for q in qs:
        ins = vlm_inputs(processor, img, q["question"] + BRIEF, device)
        L_ = ins["input_ids"].shape[1]
        assert token_spans(ins["input_ids"], model.config)["vis_end"] + 2 == P
        pos = mrope_position_ids(model, ins["input_ids"], ins["image_grid_thw"], torch.ones_like(ins["input_ids"]))
        with QKCapture() as cap:
            model(input_ids=ins["input_ids"], attention_mask=torch.ones_like(ins["input_ids"]), position_ids=pos,
                  pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"], use_cache=False)
            m, _ = per_head_column_stats(cap.qk, P, L_)
        del cap
        aq.append(m[:, :, :P])
    vis = torch.zeros(P, dtype=torch.bool); vis[pre["spans"]["visual"]] = True
    return {"P": P, "visual": vis, "prefix_ids": pre["prefix_ids"][0],
            "mlp_norm": dyn.mlp_norm, "R": dyn.R, "D": dyn.D, "hidden_rel": dyn.hidden_rel, "hidden_cos": dyn.hidden_cos,
            "knorm": knorm[:, :, :P], "vnorm": vnorm[:, :, :P], "attn_pre": attn_pre[:, :, :P],
            "attn_desc": desc_mean[:, :, :P], "attn_desc_peak": desc_peak[:, :, :P], "attn_q": torch.stack(aq).mean(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--limit", type=int, default=30); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl"); ap.add_argument("--out", default="results/context_only/mlp_kv_relation")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))][:a.limit]
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    vis_mod = model.model.visual if hasattr(model.model, "visual") else None
    if vis_mod is not None and hasattr(vis_mod, "config"):
        vis_mod.config._attn_implementation = "sdpa"
    placeholders = {int(model.config.image_token_id)}
    special = set(int(s) for s in processor.tokenizer.all_special_ids) - placeholders
    data = []
    for i, row in enumerate(rows):
        img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
        d = collect(model, processor, img, row["questions"][1:4], a.device)
        d["protected"] = protected_positions(d["prefix_ids"][None], special, 4)
        d["sample_id"] = row["sample_id"]
        data.append({k: (v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in d.items()})
        print(f"[{i+1}/{len(rows)}] {row['sample_id']} P={d['P']}", flush=True)
    out = os.path.join(ROOT, a.out)
    torch.save(data, out + ".pt")
    print(f"[saved] {out}.pt")


if __name__ == "__main__":
    main()
