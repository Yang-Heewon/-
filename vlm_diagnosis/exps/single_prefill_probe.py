"""단일 prefill 신호 판 — "이미지 + 짧은 고정 probe 문장" 한 번의 forward에서 나오는 통계로
KV 조각(층, KV head, token)을 고를 수 있는가. 기준은 20% 유지(추가로 5%), 화면 172장(Qwen2.5-VL).

대조 (기존)
  kvzip         재구성: 범용 지시 + 설명문 생성 → 설명문 행의 최대 attention          (2 pass, 생성 포함)
  attn1         이미지만 넣었을 때 뒤 조각 행들이 앞 조각을 본 평균                   (1 pass, 목적 없음)
  mlp / knorm   MLP 쓰기 비율 / K 벡터 크기                                           (1 pass)
  random / oracle(그 질문 자신의 attention, 상한)

새 신호 (모두 1 pass: 이미지 + 고정 문장 ≤ 40 token, 생성 없음)
  probe_repeat      "그대로 옮겨라" 지시문 행이 조각을 본 최대 attention  (kvzip 과 같은 forward 의 지시문 행만)
  probe_repeat_mean 같은 행들의 평균
  probe_summ        "핵심 내용을 요약하라" 지시문 행의 최대
  probe_q           "이 화면은 무엇이며 무엇을 할 수 있는가" 질문 행의 최대
  probe_max         세 probe 를 층별 정규화 후 최대
  qmean             기대 attention: probe 행 query 의 head별 평균 벡터 q̄ 로 softmax(q̄·K/√d) (행렬 없이 벡터 내적)
  kcover            K 공간 커버리지: head마다 farthest-point 순서 (먼저 뽑힌 조각일수록 높은 점수)
  hcos              층을 지나며 잔차 방향이 얼마나 바뀌었나 1−cos(x_in, x_out), 한 층 뒤 KV 에 대응
  mlp_d             문서의 D = |R_l − R_{l−1}| (MLP 비율의 층간 변화), 한 층 뒤 KV 에 대응

  python -m vlm_diagnosis.exps.single_prefill_probe --limit 1 --device cuda:0
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

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.attnstat import QKCapture
from vlm_diagnosis.core.metrics import anls, exact_match, normalize_text
from vlm_diagnosis.core.kv_select import (
    per_head_column_stats, select_triples, greedy_generate_perhead, kept_composition)
from vlm_diagnosis.core import signals as S
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS, BRIEF
from vlm_diagnosis.exps.core_delta_dram import REPEAT_PROMPT, N_SINK
from vlm_diagnosis.exps.mlp_signal_probe import (
    MLPWriteCapture, layer_norm_excl_sink, shift_to_kv, spearman, _lm_layers, _fmt)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

PROBES = {
    "repeat": REPEAT_PROMPT,
    "summ": "Summarize all the information shown on this screen, including every text element.",
    "q": "What is this screen about, and what can the user do here?",
}


class ResidualChangeCapture:
    """층마다 token별 1 − cos(층 입력 잔차, 층 출력 잔차) 를 기록."""

    def __init__(self, model):
        self.layers = _lm_layers(model)
        self.x_in, self.change, self.handles = {}, {}, []

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            def pre(m, args, kwargs, li=li):
                h = args[0] if args else kwargs["hidden_states"]
                self.x_in[li] = h[0].float()
            def post(m, args, out, li=li):
                o = out[0] if isinstance(out, (tuple, list)) else out
                cos = torch.nn.functional.cosine_similarity(self.x_in[li], o[0].float(), dim=-1)
                self.change[li] = (1.0 - cos).cpu()
                del self.x_in[li]
            self.handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(post))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False

    def tensor(self):
        return torch.stack([self.change[l] for l in range(len(self.layers))])  # (L, T)


@torch.no_grad()
def expected_attention(qk, row_start, row_end, n_cols):
    """probe 행 [row_start,row_end) 의 query 를 head별 평균한 q̄ 로 softmax(q̄·K/√d) 를 열 [0,n_cols) 에 대해 계산.
    그룹 내 query head 는 최대. 반환 (L, Hkv, n_cols)."""
    out = []
    for q, k in qk:
        q = q[0].float(); k = k[0].float()
        Hq, _, d = q.shape; Hkv = k.shape[0]; g = Hq // Hkv
        qbar = q[:, row_start:row_end].mean(1)                                  # (Hq, d)
        kk = k[:, :n_cols].repeat_interleave(g, dim=0)                          # (Hq, n, d)
        w = torch.einsum("hd,hnd->hn", qbar, kk) / math.sqrt(d)
        p = w.softmax(-1)
        out.append(p.view(Hkv, g, n_cols).amax(1).cpu())
    return torch.stack(out)


@torch.no_grad()
def k_coverage_order(qk, n_cols, n_steps, n_sink=N_SINK):
    """head마다 K 벡터의 farthest-point 순서. 점수 = n_cols − 뽑힌 순번 (미선택은 남은 최소거리로 아래 순위).
    반환 (L, Hkv, n_cols)."""
    L = len(qk); Hkv = qk[0][1].shape[1]
    K = torch.stack([k[0, :, :n_cols].float() for _, k in qk]).view(L * Hkv, n_cols, -1)  # (LH, n, d)
    dev = K.device
    LH = K.shape[0]
    K = K / K.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    score = torch.zeros(LH, n_cols, device=dev)
    mind = torch.full((LH, n_cols), float("inf"), device=dev)
    ar = torch.arange(LH, device=dev)
    cur = torch.full((LH,), n_sink, dtype=torch.long, device=dev)   # 첫 대표 = 첫 비-sink 조각
    for step in range(n_steps):
        score[ar, cur] = float(n_cols - step)
        c = K[ar, cur]                                                # (LH, d)
        dist = 1.0 - torch.einsum("bd,bnd->bn", c, K)                 # 코사인 거리
        mind = torch.minimum(mind, dist)
        mind[ar, cur] = -1.0
        cur = mind.argmax(-1)
    rest = score == 0
    score[rest] = (mind.clamp(min=0) * 0.5)[rest]                     # 미선택: 0~0.5 사이, 선택보다 항상 아래
    return score.view(L, Hkv, n_cols).cpu()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budgets", default="0.2,0.05")
    ap.add_argument("--signals", default="kvzip,attn1,mlp,knorm,probe_repeat,probe_repeat_mean,probe_summ,probe_q,"
                                         "probe_max,qmean,kcover,hcos,mlp_d")
    ap.add_argument("--eval-questions-per-doc", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/smoke/single_prefill_probe.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()

    budgets = sorted({float(x) for x in a.budgets.split(",")}, reverse=True)
    signals = [s for s in a.signals.split(",") if s]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))][a.shard::a.nshards]
    if a.reverse:
        rows = rows[::-1]
    if a.limit:
        rows = rows[:a.limit]
    if a.nshards > 1:
        a.out = a.out.replace(".jsonl", f".shard{a.shard}.jsonl")
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    vis = model.model.visual if hasattr(model.model, "visual") else getattr(model, "visual", None)
    if vis is not None and hasattr(vis, "config"):
        vis.config._attn_implementation = "sdpa"
    N_LAYERS, N_KV_HEADS, HEAD_DIM = kv_dims(model)
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_id = f"sp1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    done = set()
    if a.resume and os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                if r.get("record_type") not in ("run_metadata", "signal_diag"):
                    done.add(str(r["sample_id"]))
            except Exception:
                pass

    def forward_capture(ins, ids=None, with_hidden=False):
        """forward 한 번 + (q,k) 캡처 (+ MLP 비율, 잔차 방향 변화)."""
        ids = ins["input_ids"] if ids is None else ids
        attn = torch.ones_like(ids)
        pos = mrope_position_ids(model, ids, ins["image_grid_thw"], attn)
        caps = [QKCapture()]
        if with_hidden:
            caps += [MLPWriteCapture(model), ResidualChangeCapture(model)]
        for c in caps:
            c.__enter__()
        try:
            model(input_ids=ids, attention_mask=attn, position_ids=pos,
                  pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"], use_cache=False)
        finally:
            pass
        return caps

    def close(caps):
        for c in caps:
            c.__exit__(None, None, None)

    with open(out_path, "a" if a.resume else "w") as f:
        f.write(json.dumps({"record_type": "run_metadata", "run_id": run_id, "stage": "SINGLE_PREFILL_PROBE",
                            "model": a.model, "manifest_path": a.manifest, "budgets": budgets,
                            "signals": signals, "granularity": "head", "sink_protect": N_SINK,
                            "probes": PROBES,
                            "semantics": "question rows and generated rows see only the kept set",
                            "started_at": datetime.now(timezone.utc).isoformat()}) + "\n")
        for di, row in enumerate(rows):
            if str(row["sample_id"]) in done:
                continue
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs = row["questions"][1:1 + a.eval_questions_per_doc]
            # ---- 1-pass A: 이미지 접두만 (attn1, mlp, knorm, hcos, mlp_d)
            ins0 = S.vlm_inputs(processor, img, "x", a.device)
            sp0 = token_spans(ins0["input_ids"], model.config)
            P_shared = sp0["vis_end"] + 2
            v0 = int(sp0["visual"].min())
            ids0 = ins0["input_ids"][:, :P_shared]
            cap, mw, rc = forward_capture(ins0, ids0, with_hidden=True)
            attn1, _ = per_head_column_stats(cap.qk, v0, P_shared)
            knorm = torch.stack([k[0].float().norm(dim=-1).cpu() for _, k in cap.qk])
            n_cover = int(round(max(budgets) * P_shared)) + 8
            kcover = k_coverage_order(cap.qk, P_shared, n_cover)
            mlp_ratio = mw.ratio()[:, :P_shared]
            hcos_LT = rc.tensor()[:, :P_shared]
            close([cap, mw, rc]); del cap, mw, rc
            mlp_kv = shift_to_kv(mlp_ratio, N_KV_HEADS)
            mlp_d = shift_to_kv(torch.cat([mlp_ratio[:1] * 0, (mlp_ratio[1:] - mlp_ratio[:-1]).abs()]), N_KV_HEADS)
            hcos = shift_to_kv(hcos_LT, N_KV_HEADS)
            # ---- 1-pass B: 이미지 + probe 문장 (생성 없음) — repeat 는 kvzip 생성 forward 에서 함께 얻음
            probe_peak, probe_mean, probe_len = {}, {}, {}
            for pname, ptext in PROBES.items():
                if pname == "repeat":
                    continue
                insp = S.vlm_inputs(processor, img, ptext, a.device)
                Pp = insp["input_ids"].shape[1]
                assert token_spans(insp["input_ids"], model.config)["vis_end"] + 2 == P_shared
                cap, = forward_capture(insp)
                m, pk = per_head_column_stats(cap.qk, P_shared, Pp)
                probe_mean[pname], probe_peak[pname] = m[:, :, :P_shared].clone(), pk[:, :, :P_shared].clone()
                probe_len[pname] = Pp - P_shared
                close([cap]); del cap
            # ---- 2-pass: 재구성 설명문 생성 + forward (kvzip); 같은 forward 의 지시문 행 = probe_repeat
            ins1 = S.vlm_inputs(processor, img, REPEAT_PROMPT, a.device)
            gen = model.generate(**{k: v for k, v in ins1.items()}, max_new_tokens=96, do_sample=False)
            P1 = ins1["input_ids"].shape[1]; Lg = gen.shape[1]
            cap, = forward_capture(ins1, gen)
            _, kvzip = per_head_column_stats(cap.qk, P1, Lg)
            m, pk = per_head_column_stats(cap.qk, P_shared, P1)
            probe_mean["repeat"], probe_peak["repeat"] = m[:, :, :P_shared].clone(), pk[:, :, :P_shared].clone()
            probe_len["repeat"] = P1 - P_shared
            qmean = expected_attention(cap.qk, P_shared, P1, P_shared)
            close([cap]); del cap
            kvzip = kvzip[:, :, :P_shared].clone()
            probe_max = torch.stack([layer_norm_excl_sink(probe_peak[p]) for p in PROBES]).amax(0)
            sig = {
                "kvzip": kvzip, "attn1": attn1, "mlp": mlp_kv, "knorm": knorm,
                "probe_repeat": probe_peak["repeat"], "probe_repeat_mean": probe_mean["repeat"],
                "probe_summ": probe_peak["summ"], "probe_q": probe_peak["q"], "probe_max": probe_max,
                "qmean": qmean, "kcover": kcover, "hcos": hcos, "mlp_d": mlp_d,
            }
            sig = {k: v for k, v in sig.items() if k in signals}
            prefix_triples = N_LAYERS * N_KV_HEADS * P_shared
            sink_forced = torch.zeros((N_LAYERS, N_KV_HEADS, P_shared), dtype=torch.bool)
            sink_forced[:, :, :N_SINK] = True
            seed = zlib.crc32(f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF
            rnd = torch.rand((N_LAYERS, N_KV_HEADS, P_shared), generator=torch.Generator().manual_seed(seed))
            keeps = {}
            diag = {"record_type": "signal_diag", "run_id": run_id, "sample_id": row["sample_id"],
                    "n_prefix": P_shared, "probe_len": probe_len, "desc_len": Lg - P1,
                    "spearman_vs_kvzip": {}, "top_overlap_vs_kvzip": {}}
            for B in budgets:
                T = int(round(B * prefix_triples)); Bt = f"B{_fmt(B*100)}"
                ref, _ = select_triples(sig["kvzip"], torch.zeros_like(kvzip), T, 1.0, sink_forced) if "kvzip" in sig else (None, None)
                for name, sc in sig.items():
                    keep, _ = select_triples(sc, torch.zeros_like(sc), T, 1.0, sink_forced)
                    keeps[f"{name}@{Bt}"] = (name, B, keep)
                    if ref is not None:
                        diag["top_overlap_vs_kvzip"][f"{name}@{Bt}"] = float((keep & ref).sum() / max(1, int(ref.sum())))
                keep, _ = select_triples(rnd, torch.zeros_like(rnd), T, 1.0, sink_forced)
                keeps[f"random@{Bt}"] = ("random", B, keep)
            if "kvzip" in sig:
                for name, sc in sig.items():
                    diag["spearman_vs_kvzip"][name] = spearman(sc[:, :, N_SINK:], sig["kvzip"][:, :, N_SINK:])
            f.write(json.dumps(diag) + "\n")

            for q in qs:
                q_text = q["question"] + BRIEF
                ins = S.vlm_inputs(processor, img, q_text, a.device)
                sp = token_spans(ins["input_ids"], model.config)
                vis_idx, vis_end, P = sp["visual"], sp["vis_end"], int(sp["L"])
                assert vis_end + 2 == P_shared
                golds = q["answers"]
                with QKCapture() as cap:
                    pred_full = greedy_generate_masked(model, processor, ins, max_new_tokens=a.max_new_tokens)
                    q_mean, _ = per_head_column_stats(cap.qk[:N_LAYERS], P_shared, P)
                del cap
                s1 = q_mean[:, :, :P_shared].clone()
                base = {"run_id": run_id, "model": a.model, "dataset": row["dataset"], "sample_id": row["sample_id"],
                        "question_id": q["question_id"], "gold": golds, "n_prefix": P_shared, "n_visual": int(len(vis_idx))}
                f.write(json.dumps({**base, "condition_id": "FULL_KV", "selector": "full", "prediction": pred_full,
                                    "anls": anls(pred_full, golds), "em": exact_match(pred_full, golds)}, ensure_ascii=False) + "\n")

                def run(cond_id, selector, B, keep_prefix):
                    keep = torch.ones((N_LAYERS, N_KV_HEADS, P), dtype=torch.bool)
                    keep[:, :, :P_shared] = keep_prefix
                    pred = greedy_generate_perhead(model, processor, ins, keep, P_shared, max_new_tokens=a.max_new_tokens)
                    comp = kept_composition(keep_prefix, vis_idx, N_SINK)
                    f.write(json.dumps({**base, "condition_id": cond_id, "selector": selector, "keep_frac": B,
                                        "prediction": pred, "anls": anls(pred, golds), "em": exact_match(pred, golds),
                                        "loyalty": float(normalize_text(pred) == normalize_text(pred_full)),
                                        **{k: v for k, v in comp.items() if k != "per_layer_kept"}},
                                       ensure_ascii=False) + "\n")
                    f.flush()

                for cond_id, (name, B, keep) in keeps.items():
                    run(cond_id, name, B, keep)
                for B in budgets:
                    T = int(round(B * prefix_triples))
                    keep, _ = select_triples(s1, torch.zeros_like(s1), T, 1.0, sink_forced)
                    run(f"oracle@B{_fmt(B*100)}", "oracle", B, keep)
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} prefix={P_shared} probe={probe_len} desc={Lg-P1} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
