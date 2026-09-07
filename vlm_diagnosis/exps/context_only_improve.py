"""Context-only 개선 판 — 재구성 + 다양성 채우기, K 공간 다양성만, Expected-Attention식 단일 prefill 추정,
MLP 스파이크 자동 보호. 물리 삭제 harness(context_only_cache) 위에서, 화면(ScreenQA)·자연 사진(GQA) 공통.
stage 이름은 'sweep' 으로 기록하고 v2 metadata(expected_conditions 명시)로 남긴다.

조건 (유지율 k 마다):
  random | recon_desc | recon_desc/div30 | recon_desc/div50 | kcover | expattn | expattn_v
  recon_desc+auto | random+auto        (+auto = 보호 집합을 앞 4개 고정 대신 MLP 스파이크 ∪ special 로)

  python -m vlm_diagnosis.exps.context_only_improve --manifest experiments/manifests/gqa_discovery.jsonl --split dev --device cuda:0
"""
import argparse
import hashlib
import math
import os
import time
from datetime import datetime, timezone

import torch
import transformers
from PIL import Image

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.context_only_cache import prefill_context, build_memory, report_dict, _sync, method_scores
from vlm_diagnosis.core.context_only_protocol import ExperimentLog, parity_tolerances
from vlm_diagnosis.core.static_pair_select import spearman_avg_rank
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS, BRIEF
from vlm_diagnosis.exps.context_only_kv import (
    ROOT, DEV_CONTEXTS, _git, _sha, cond_id, load_rows, external_scores, _answer_records)


def build_conditions(keep_ratios, seeds, panel="improve"):
    conds = [("full", "none", "none", 1.0, 0)]
    if panel == "hidden":
        for k in keep_ratios:
            for s in seeds:
                conds.append(("random", "keep_high", "global", k, s))
            conds += [("kcover", "keep_high", "global", k, 0), ("hidden", "keep_high", "hidcover", k, 0),
                      ("hidden", "keep_high", "hid2k", k, 0), ("recon_desc", "keep_high", "global", k, 0),
                      ("recon_desc", "keep_high", "hidquota", k, 0)]
        return conds
    for k in keep_ratios:
        for s in seeds:
            conds.append(("random", "keep_high", "global", k, s))
        conds += [("recon_desc", "keep_high", "global", k, 0), ("recon_desc", "keep_high", "div30", k, 0),
                  ("recon_desc", "keep_high", "div50", k, 0), ("kcover", "keep_high", "global", k, 0),
                  ("expattn", "keep_high", "global", k, 0), ("expattn_v", "keep_high", "global", k, 0),
                  ("recon_desc+auto", "keep_high", "global", k, 0), ("random+auto", "keep_high", "global", k, 0)]
    return conds


@torch.no_grad()
def expected_attention_scores(qk, P, window=64):
    """Expected-Attention 식 근사 (Liu et al. 2025 의 정신을 따른 단일 prefill 추정; 공식 구현 아님).
    미래 query 를 context 마지막 window 행의 (RoPE 적용 후) query 분포 N(μ, Σ) 로 보고
    E[exp(qᵀk/√d)] = exp(μᵀk/√d + kᵀΣk/(2d)) 를 열 j 마다 계산, softmax 후 그룹 내 query head 최대.
    반환 (L, Hkv, P) 와 같은 것에 ‖v‖ 를 곱한 value-aware 판."""
    out, out_v = [], []
    for q, k in qk:
        q = q[0].float(); k = k[0, :, :P].float()
        Hq, _, d = q.shape; Hkv = k.shape[0]; g = Hq // Hkv
        w = q[:, max(0, P - window):P]                                  # (Hq, W, d)
        mu = w.mean(1)                                                  # (Hq, d)
        wc = w - mu[:, None]
        sigma = torch.einsum("hwd,hwe->hde", wc, wc) / max(1, w.shape[1] - 1)   # (Hq, d, d)
        kk = k.repeat_interleave(g, dim=0)                              # (Hq, P, d)
        lin = torch.einsum("hd,hpd->hp", mu, kk) / math.sqrt(d)
        quad = torch.einsum("hpd,hde,hpe->hp", kk, sigma, kk) / (2 * d)
        p = (lin + quad).softmax(-1)                                    # (Hq, P)
        out.append(p.view(Hkv, g, P).amax(1).cpu())
    return torch.stack(out)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--split", default="dev", choices=["dev", "eval", "all"])
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--keep-ratios", default="0.2,0.05")
    ap.add_argument("--random-seeds", default="0")
    ap.add_argument("--questions-per-context", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--spike-factor", type=float, default=10.0)
    ap.add_argument("--expattn-window", type=int, default=64)
    ap.add_argument("--panel", default="improve", choices=["improve", "hidden"])
    ap.add_argument("--hidden-layers", default="1-9", help="hidden 묶음 (예: 1-9 early, 10-18 mid, 19-27 late)")
    ap.add_argument("--hidden-clusters", type=int, default=32)
    ap.add_argument("--hidden-share", type=float, default=0.5)
    ap.add_argument("--out", default=None); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    h0, h1 = [int(x) for x in a.hidden_layers.split("-")]

    keep_ratios = sorted({float(x) for x in a.keep_ratios.split(",")}, reverse=True)
    seeds = [int(x) for x in a.random_seeds.split(",")]
    rows = load_rows(a)
    dom = os.path.basename(a.manifest).split("_")[0]
    out_rel = a.out or f"results/context_only/{a.panel}_{a.model}_{dom}_{a.split}.jsonl"
    if a.nshards > 1:
        out_rel = out_rel.replace(".jsonl", f".shard{a.shard}.jsonl")
    out_path = os.path.join(ROOT, out_rel)
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    vis = model.model.visual if hasattr(model.model, "visual") else None
    if vis is not None and hasattr(vis, "config"):
        vis.config._attn_implementation = "sdpa"
    L, H, _ = kv_dims(model)
    placeholders = {int(getattr(model.config, k)) for k in ("image_token_id", "video_token_id") if getattr(model.config, k, None) is not None}
    special_ids = set(int(s) for s in processor.tokenizer.all_special_ids) - placeholders
    conds = build_conditions(keep_ratios, seeds, a.panel)
    # 작은 이미지 제외: 최소 유지율에서 예산이 보호 쌍(앞 4 + special, 층·head 전체)보다 작으면 명세상 오류이므로
    # 실행 전에 걸러 metadata 에 기록한다 (조용히 빼지 않음). token 수는 processor 만으로 계산.
    from vlm_diagnosis.core.signals import vlm_inputs
    from vlm_diagnosis.core.spans import token_spans
    from vlm_diagnosis.core.static_pair_select import protected_positions
    kept_rows, excluded = [], []
    for r in rows:
        img = Image.open(os.path.join(ROOT, r["image"])).convert("RGB")
        ins = vlm_inputs(processor, img, "x", "cpu")
        P = int(token_spans(ins["input_ids"], model.config)["vis_end"]) + 2
        n_prot = int(protected_positions(ins["input_ids"][:, :P], special_ids, 4).sum()) * L * H
        if int(round(min(keep_ratios) * L * H * P)) < n_prot:
            excluded.append({"context_id": str(r["sample_id"]), "n_tokens": P, "reason": "budget < protected pairs at min keep_ratio"})
        else:
            kept_rows.append(r)
    rows = kept_rows
    print(f"[filter] excluded {len(excluded)} small contexts: {[e['context_id'] for e in excluded]}", flush=True)
    tc = getattr(model.config, "text_config", model.config)
    dtype = next(model.parameters()).dtype
    impl_files = ["vlm_diagnosis/exps/context_only_improve.py", "vlm_diagnosis/core/context_only_cache.py",
                  "vlm_diagnosis/core/static_pair_select.py", "vlm_diagnosis/core/mlp_dynamics.py", "vlm_diagnosis/core/ragged_kv.py"]
    meta = {
        "run_id": f"co-improve-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", "stage": "sweep", "model": a.model,
        "model_id": getattr(model.config, "_name_or_path", ""), "model_revision": str(getattr(model.config, "_commit_hash", None) or "local"),
        "tokenizer_revision": str(getattr(processor.tokenizer, "_commit_hash", None) or "local"),
        "implementation_sha256": hashlib.sha256(b"".join(open(os.path.join(ROOT, f), "rb").read() for f in impl_files)).hexdigest(),
        "code_revision": _git(["rev-parse", "HEAD"]), "code_dirty": bool(_git(["status", "--porcelain"])),
        "transformers": transformers.__version__, "torch": torch.__version__, "device": a.device, "dtype": str(dtype),
        "attn_backend": tc._attn_implementation, "manifest": a.manifest, "manifest_sha256": _sha(os.path.join(ROOT, a.manifest)),
        "split": a.split, "dev_contexts": DEV_CONTEXTS, "n_contexts": len(rows), "shard": a.shard, "nshards": a.nshards,
        "keep_ratios": keep_ratios, "methods": sorted({c[0] for c in conds if c[0] != "full"}), "random_seeds": seeds, "eps": a.eps,
        "protect": f"fixed: prefix first 4 + special ids (placeholders excluded); auto: MLP spike (R > {a.spike_factor:g}x layer median) + special ids",
        "protected_special_ids": sorted(special_ids), "spike_factor": a.spike_factor, "expattn_window": a.expattn_window,
        "granularity": "kv_pair", "storage": "physical ragged deletion",
        "budget_rule": "B = round(keep_ratio * L*Hkv*T), protected included", "tie_rule": "seeded permutation",
        "nll_rule": "answers[0] body tokens, EOS excluded, teacher forced",
        "questions": f"manifest questions[1:1+{a.questions_per_context}] + BRIEF", "question_start": 1,
        "questions_per_context": a.questions_per_context, "brief": BRIEF, "max_new_tokens": a.max_new_tokens, "decode": "greedy",
        "parity_tolerances": parity_tolerances(dtype), "started_at": datetime.now(timezone.utc).isoformat(),
        "expected_conditions": [cond_id(*c) for c in conds],
        "expected_context_ids": [str(r["sample_id"]) for r in rows], "excluded_contexts": excluded,
        "expected_question_ids_by_context": {str(r["sample_id"]): [q["question_id"] for q in r["questions"][1:1 + a.questions_per_context]] for r in rows},
        "diversity": "importance top-(1-f)B then per (layer, head) farthest-point fill on K (cosine), f in {0.3, 0.5}",
        "kcover": "per (layer, head) farthest-point order on K, start = token 4, steps = 0.25T + 8",
        "expattn": "N(mu, Sigma) of last-window post-RoPE queries; exp(mu.k/sqrt(d) + k.Sigma.k/(2d)); softmax; group max; _v = x ||v||",
        "panel": a.panel, "hidden_layers": a.hidden_layers, "hidden_clusters": a.hidden_clusters, "hidden_share": a.hidden_share,
        "hidden": "F_i = normalize(sum_l normalize(x_l,i)) over hidden layers → PCA64 → k-means C clusters; hidcover = token k-center shared by all heads; "
                  "hid2k = per (layer, head): share of budget split equally over clusters, K farthest-point inside, rest global K farthest-point; "
                  "hidquota = same quota with reconstruction score inside clusters",
    }
    log = ExperimentLog(out_path, meta, resume=a.resume)
    t_all = time.time()
    for di, row in enumerate(rows):
        cid = str(row["sample_id"])
        if cid in log.done:
            continue
        t0 = time.time()
        img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
        qs = row["questions"][1:1 + a.questions_per_context]
        hid_cap = None
        if a.panel == "hidden":
            from vlm_diagnosis.scripts.hidden_k_complementarity import HiddenCapture
            hid_cap = HiddenCapture(model); hid_cap.__enter__()
        try:
            pre = prefill_context(model, processor, img, a.device, collect_dynamics=True, capture_qk=True, eps=a.eps)
        finally:
            if hid_cap is not None:
                hid_cap.__exit__(None, None, None)
        P = pre["spans"]["P"]
        ext, ext_cost = external_scores(model, processor, img, pre, {"recon_desc"}, a.device, L, H)
        if hid_cap is not None:
            from vlm_diagnosis.scripts.late_mlp_semantic_probe import nrm, pca
            from vlm_diagnosis.core.static_pair_select import kmeans_labels
            _sync(a.device); t1 = time.perf_counter()
            F = nrm(torch.stack([nrm(hid_cap.x[l].float()[:P]) for l in range(h0, h1 + 1)]).sum(0))
            z = pca(F, 64)
            ext["hidden_z"] = z; ext["hidden_clusters"] = kmeans_labels(z, a.hidden_clusters, seed=0); ext["hidden_share"] = a.hidden_share
            ext_cost["hidden_feature_seconds"] = time.perf_counter() - t1
            del hid_cap
        _sync(a.device); t1 = time.perf_counter()
        ea = expected_attention_scores(pre["qk"], P, a.expattn_window)
        vnorm = torch.stack([v[0, :, :P].float().norm(dim=-1).cpu() for _, v in pre["kv"]])
        ext["expattn"], ext["expattn_v"] = ea, ea * vnorm
        _sync(a.device); ext_cost["expattn_seconds"] = time.perf_counter() - t1
        pre["qk"] = None
        diag = {"record_type": "diagnostic", "context_id": cid, "n_tokens": P, "n_visual": int(pre["spans"]["visual"].numel()),
                "prefill_seconds": pre["prefill_seconds"], "residual_max_rel_err": float(pre["dynamics"].residual_max_rel_err.max()),
                "dtype": pre["dynamics"].dtype, **ext_cost, "spearman_vs_recon_desc": {}}
        for m in (("kcover",) if a.panel == "hidden" else ("expattn", "expattn_v", "kcover")):
            s, _, _ = method_scores(m, pre, L, H, 0, ext)
            diag["spearman_vs_recon_desc"][m] = spearman_avg_rank(s[:, :, 4:], ext["recon_desc"][:, :, 4:])
        log.emit(diag)
        pred_full = {}
        for (method, direction, selector, k, seed) in conds:
            cond = cond_id(method, direction, selector, k, seed)
            base_method, pmode = (method.split("+")[0], "auto") if "+auto" in method else (method, "fixed")
            mem, rep, _ = build_memory(model, processor, pre, cid, base_method, k, seed, a.device,
                                       direction=direction if method != "full" else "keep_high",
                                       selector=selector if method != "full" else "global", special_ids=special_ids,
                                       extra_scores=ext, protect_mode=pmode if method != "full" else "fixed", spike_factor=a.spike_factor)
            rep.method = method
            log.emit({"record_type": "build", "context_id": cid, "condition": cond, **report_dict(rep)})
            _answer_records(model, mem, cond, method, direction, selector, k, seed, cid, qs, a, log.emit, pred_full)
            del mem
        del pre
        log.emit({"record_type": "context_done", "context_id": cid, "seconds": time.time() - t0})
        print(f"[{di+1}/{len(rows)}] {cid} P={P} {time.time()-t0:.0f}s", flush=True)
    log.close()
    print(f"[saved] {out_path} ({(time.time()-t_all)/60:.1f} min)")


if __name__ == "__main__":
    main()
