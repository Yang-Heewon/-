"""MLP 신호가 KV 솎아내기에 도움이 되는가 — 같은 캐시, 같은 예산(5%, 층·KV head 단위)에서
신호 9개로 고른 결과를 비교한다. 질문은 선택에 관여하지 않는다 (남긴 조각만 보고 답함).

신호 (조각 = (층 l, KV head h, token j), 이미지 접두 = system + vision 경계 + 시각)
  kvzip        재구성 최대 attention: 범용 지시로 설명문 생성 → 설명문 행이 조각을 참고한 최대 (2 pass)
  kvzip_x_mlp  kvzip × g(MLP 쓰기 비율[l-1, j])                                                (2 pass)
  kvzip_rowmlp 설명문 행 q마다 w_q = 그 행의 MLP 비율로 가중한 뒤 열 최대: max_q w_q·a[q→j]  (2 pass)
  attn1        접두 prefill 안에서 뒤 token 행들이 조각 j를 참고한 평균                         (1 pass)
  attn1_x_mlp  attn1 × g(MLP 비율)                                                             (1 pass)
  mlp          MLP 쓰기 비율만. r[l, j] = ‖MLP_l(RMSNorm(h))‖ / ‖h‖ 를 층 l+1 의 KV에 대응(head 공통) (1 pass)
  knorm        K 벡터 크기(head별)                                                              (1 pass)
  random / oracle(그 질문 자신의 attention)                                                    대조
g(r) = 층별로 sink 제외 최대값으로 나눈 [0,1] 값.

진단: 신호마다 kvzip 점수와의 Spearman 순위 상관과 상위 5% 조각 겹침 비율을 화면마다 기록.

  python -m vlm_diagnosis.exps.mlp_signal_probe --limit 1 --device cuda:0
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

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _fmt(x):
    return f"{x:g}"


def _lm_layers(model):
    core = model.model
    return (core.language_model if hasattr(core, "language_model") else core).layers


class MLPWriteCapture:
    """한 forward 동안 층마다 token별 ‖MLP 출력‖ / ‖MLP 적용 전 잔차‖ 를 기록 (모델 수정 없음)."""

    def __init__(self, model):
        self.layers = _lm_layers(model)
        self.base, self.delta, self.handles = {}, {}, []

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(
                lambda m, a, o, li=li: self.base.__setitem__(li, a[0][0].float().norm(dim=-1).cpu())))
            self.handles.append(layer.mlp.register_forward_hook(
                lambda m, a, o, li=li: self.delta.__setitem__(li, o[0].float().norm(dim=-1).cpu())))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False

    def ratio(self):
        n = len(self.layers)
        assert len(self.base) == n and len(self.delta) == n, (len(self.base), len(self.delta), n)
        return torch.stack([self.delta[l] / self.base[l].clamp(min=1e-6) for l in range(n)])  # (L, T)


@torch.no_grad()
def per_head_column_max_rowweighted(qk, row_start, row_end, row_w, chunk=256):
    """열 최대를 취하기 전에 행마다 가중치 row_w[l, q] 를 곱한다. 반환 (L, Hkv, T)."""
    peaks = []
    for li, (q, k) in enumerate(qk):
        q = q[0].float(); k = k[0].float()
        Hq, L, d = q.shape
        Hkv = k.shape[0]; g = Hq // Hkv
        kk = k.repeat_interleave(g, dim=0)
        cols = torch.arange(L, device=q.device)
        m_acc = torch.zeros(Hq, L, device=q.device)
        for s0 in range(row_start, row_end, chunk):
            e = min(s0 + chunk, row_end)
            w = q[:, s0:e] @ kk.transpose(-1, -2) / math.sqrt(d)
            rows = torch.arange(s0, e, device=q.device)
            w.masked_fill_(cols[None, None, :] > rows[None, :, None], float("-inf"))
            p = w.softmax(-1) * row_w[li, s0:e].to(q.device)[None, :, None]
            m_acc = torch.maximum(m_acc, p.amax(1))
            del w, p
        peaks.append(m_acc.view(Hkv, g, L).amax(1).cpu())
    return torch.stack(peaks)


def layer_norm_excl_sink(x, n_sink=N_SINK):
    """(L, H, T) 점수를 층별로 sink 제외 최대값으로 나눠 [0,1]로."""
    out = x.clone()
    for l in range(x.shape[0]):
        mx = x[l, :, n_sink:].max().clamp(min=1e-12)
        out[l] = (x[l] / mx).clamp(max=1.0)
    return out


def layer_rank_excl_sink(x, n_sink=N_SINK):
    """(L, H, T) 점수를 층별로 sink 제외 순위 백분위(0~1)로. sink는 1."""
    out = torch.ones_like(x)
    for l in range(x.shape[0]):
        v = x[l, :, n_sink:].flatten()
        r = torch.argsort(torch.argsort(v)).float() / max(1, v.numel() - 1)
        out[l, :, n_sink:] = r.view(x.shape[1], -1)
    return out


def shift_to_kv(ratio_LT, n_heads):
    """MLP 비율 (L, T): 층 l의 MLP 출력은 층 l+1의 KV에 반영 → kv[l] = ratio[l-1], kv[0] = ratio[0]."""
    L, T = ratio_LT.shape
    shifted = torch.cat([ratio_LT[:1], ratio_LT[:-1]], dim=0)
    return shifted[:, None, :].expand(L, n_heads, T).clone()


def spearman(a, b, n=20000, seed=0):
    a = a.flatten(); b = b.flatten()
    if a.numel() > n:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(a.numel(), generator=g)[:n]
        a, b = a[idx], b[idx]
    ra = torch.argsort(torch.argsort(a)).float(); rb = torch.argsort(torch.argsort(b)).float()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budgets", default="0.05")
    ap.add_argument("--signals", default="kvzip,kvzip_x_mlp,kvzip_rowmlp,attn1,attn1_x_mlp,mlp,knorm")
    ap.add_argument("--eval-questions-per-doc", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/smoke/mlp_signal_probe.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()

    budgets = sorted({float(x) for x in a.budgets.split(",")})
    signals = [s for s in a.signals.split(",") if s]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))][a.shard::a.nshards]
    if a.reverse:
        rows = rows[::-1]
    if a.limit:
        rows = rows[:a.limit]
    if a.nshards > 1:
        a.out = a.out.replace(".jsonl", f".shard{a.shard}.jsonl")
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    # 비전 인코더의 eager full attention(패치 1만 개, fp32 softmax)이 32GB에서 한계에 걸리므로
    # 인코더만 메모리 효율 커널(sdpa)로 바꾼다. 언어 모델은 eager 그대로(attention 캡처 필요).
    vis = model.model.visual if hasattr(model.model, "visual") else getattr(model, "visual", None)
    if vis is not None and hasattr(vis, "config"):
        vis.config._attn_implementation = "sdpa"
    N_LAYERS, N_KV_HEADS, HEAD_DIM = kv_dims(model)
    def _mem(tag):
        if torch.cuda.is_available():
            print(f"[mem] {tag}: alloc={torch.cuda.memory_allocated()/2**30:.2f}G "
                  f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G peak={torch.cuda.max_memory_allocated()/2**30:.2f}G "
                  f"device={torch.cuda.get_device_name(0)} free/total={[round(x/2**30,2) for x in torch.cuda.mem_get_info()]}", flush=True)
    _mem("after load")
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_id = f"mlp-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    done = set()
    if a.resume and os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                if r.get("record_type") not in ("run_metadata", "signal_diag"):
                    done.add(str(r["sample_id"]))
            except Exception:
                pass

    with open(out_path, "a" if a.resume else "w") as f:
        f.write(json.dumps({"record_type": "run_metadata", "run_id": run_id, "stage": "MLP_SIGNAL_PROBE",
                            "model": a.model, "manifest_path": a.manifest, "budgets": budgets,
                            "signals": signals, "granularity": "head", "sink_protect": N_SINK,
                            "semantics": "question rows and generated rows see only the kept set",
                            "mlp_ratio": "‖mlp_out‖/‖pre-mlp residual‖ per (layer, token), shifted one layer to KV",
                            "started_at": datetime.now(timezone.utc).isoformat()}) + "\n")
        for di, row in enumerate(rows):
            if str(row["sample_id"]) in done:
                continue
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs = row["questions"][1:1 + a.eval_questions_per_doc]
            # ---- 1-pass: 이미지 접두만 forward (attention 수신량, MLP 비율, K-norm)
            ins0 = S.vlm_inputs(processor, img, "x", a.device)
            sp0 = token_spans(ins0["input_ids"], model.config)
            P_shared = sp0["vis_end"] + 2
            v0 = int(sp0["visual"].min())
            ids = ins0["input_ids"][:, :P_shared]
            attn = torch.ones_like(ids)
            pos = mrope_position_ids(model, ids, ins0["image_grid_thw"], attn)
            _mem("before first forward")
            with QKCapture() as cap, MLPWriteCapture(model) as mw:
                model(input_ids=ids, attention_mask=attn, position_ids=pos,
                      pixel_values=ins0["pixel_values"], image_grid_thw=ins0["image_grid_thw"], use_cache=False)
                _mem("after first forward")
                attn1, attn1_max = per_head_column_stats(cap.qk, v0, P_shared)  # (L,H,P) 뒤 행들의 평균 / 최대 참조
                knorm = torch.stack([k[0].float().norm(dim=-1).cpu() for _, k in cap.qk])  # (L,H,P)
            mlp_ratio = mw.ratio()[:, :P_shared]                                 # (L,P)
            del cap, mw
            mlp_kv = shift_to_kv(mlp_ratio, N_KV_HEADS)                           # (L,H,P)
            # ---- 2-pass: 재구성 설명문 생성 + 캡처 (행 MLP 비율 포함)
            ins1 = S.vlm_inputs(processor, img, REPEAT_PROMPT, a.device)
            gen = model.generate(**{k: v for k, v in ins1.items()}, max_new_tokens=96, do_sample=False)
            P1 = ins1["input_ids"].shape[1]; Lg = gen.shape[1]
            attn_g = torch.ones(1, Lg, dtype=torch.long, device=a.device)
            pos_g = mrope_position_ids(model, gen, ins1["image_grid_thw"], attn_g)
            with QKCapture() as cap, MLPWriteCapture(model) as mw:
                model(input_ids=gen, attention_mask=attn_g, position_ids=pos_g,
                      pixel_values=ins1["pixel_values"], image_grid_thw=ins1["image_grid_thw"], use_cache=False)
                _, kvzip = per_head_column_stats(cap.qk, P1, Lg)
                row_r = mw.ratio()                                                 # (L, Lg)
                row_w = torch.stack([(row_r[l] / row_r[l, P1:Lg].max().clamp(min=1e-6)).clamp(max=1.0)
                                     for l in range(N_LAYERS)])
                kvzip_rowmlp = per_head_column_max_rowweighted(cap.qk, P1, Lg, row_w)
            del cap, mw
            kvzip = kvzip[:, :, :P_shared].clone(); kvzip_rowmlp = kvzip_rowmlp[:, :, :P_shared].clone()
            g_mlp = layer_norm_excl_sink(mlp_kv)
            g_mlp_rank = layer_rank_excl_sink(mlp_kv) ** 0.5          # 순위 정규화, 세기 γ=0.5
            sig = {
                "kvzip": kvzip,
                "attn1_max": attn1_max,
                "attn1_max_x_mlp": layer_norm_excl_sink(attn1_max) * g_mlp,
                "attn1_x_mlprank": layer_norm_excl_sink(attn1) * g_mlp_rank,
                "attn1_max_x_mlprank": layer_norm_excl_sink(attn1_max) * g_mlp_rank,
                "kvzip_x_mlp": layer_norm_excl_sink(kvzip) * g_mlp,
                "kvzip_rowmlp": kvzip_rowmlp,
                "attn1": attn1,
                "attn1_x_mlp": layer_norm_excl_sink(attn1) * g_mlp,
                "mlp": mlp_kv,
                "knorm": knorm,
            }
            sig = {k: v for k, v in sig.items() if k in signals}
            prefix_triples = N_LAYERS * N_KV_HEADS * P_shared
            sink_forced = torch.zeros((N_LAYERS, N_KV_HEADS, P_shared), dtype=torch.bool)
            sink_forced[:, :, :N_SINK] = True
            seed = zlib.crc32(f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF
            rnd = torch.rand((N_LAYERS, N_KV_HEADS, P_shared), generator=torch.Generator().manual_seed(seed))
            keeps = {}
            diag = {"record_type": "signal_diag", "run_id": run_id, "sample_id": row["sample_id"],
                    "n_prefix": P_shared, "spearman_vs_kvzip": {}, "top_overlap_vs_kvzip": {}}
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
                vis, vis_end, P = sp["visual"], sp["vis_end"], int(sp["L"])
                assert vis_end + 2 == P_shared
                golds = q["answers"]
                with QKCapture() as cap:
                    pred_full = greedy_generate_masked(model, processor, ins, max_new_tokens=a.max_new_tokens)
                    q_mean, _ = per_head_column_stats(cap.qk[:N_LAYERS], P_shared, P)
                del cap
                s1 = q_mean[:, :, :P_shared].clone()
                base = {"run_id": run_id, "model": a.model, "dataset": row["dataset"], "sample_id": row["sample_id"],
                        "question_id": q["question_id"], "gold": golds, "n_prefix": P_shared, "n_visual": int(len(vis))}
                f.write(json.dumps({**base, "condition_id": "FULL_KV", "selector": "full", "prediction": pred_full,
                                    "anls": anls(pred_full, golds), "em": exact_match(pred_full, golds)}, ensure_ascii=False) + "\n")

                def run(cond_id, selector, B, keep_prefix):
                    keep = torch.ones((N_LAYERS, N_KV_HEADS, P), dtype=torch.bool)
                    keep[:, :, :P_shared] = keep_prefix
                    pred = greedy_generate_perhead(model, processor, ins, keep, P_shared, max_new_tokens=a.max_new_tokens)
                    comp = kept_composition(keep_prefix, vis, N_SINK)
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
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} prefix={P_shared} desc={Lg-P1} {time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
