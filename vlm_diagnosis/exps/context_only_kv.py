"""Context-only KV 압축 runner (docs/CONTEXT-ONLY-KV-COMPRESSION.md 단계 1–5, VLM 화면 표본).

  --stage full      단계 1: cached-suffix 경로와 일반 FULL forward 의 logits/NLL parity
  --stage probe     단계 2: 단일 prefill 통계 (R, D, token 표, heatmap). 답변 평가 없음
  --stage deletion  단계 4: 고정 유지율(기본 20%)에서 낮은/높은 점수 삭제, random×5, 대조군
  --stage sweep     단계 5: 유지율 × 방법 품질 곡선 (context-only 방법; attn1/recon_desc 는 명시 flag)
  --stage profile   단계 5 비용: 한 방법의 build/query 비용 (dense 해제, 다른 방법 동반 실행 없음)

  python -m vlm_diagnosis.exps.context_only_kv --stage deletion --split dev --keep-ratios 0.2 --device cuda:0
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import torch
import transformers
from PIL import Image

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.metrics import exact_match, normalize_text
from vlm_diagnosis.core.context_only_cache import (
    prefill_context, build_memory, compress_context, answer_from_cache, answer_nll,
    dense_reference_logits, report_dict, CONTEXT_ONLY_METHODS, _sync)
from vlm_diagnosis.core.mlp_dynamics import token_table
from vlm_diagnosis.core.static_pair_select import spearman_avg_rank, protected_positions
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS, BRIEF

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCHEMA = "context_only_v1"
DEV_CONTEXTS = 40          # manifest 앞 40 화면 = 개발(dev), 나머지 = 평가(eval)
DELETION_SIGNALS = ("mlp_norm", "r", "d", "k_norm", "v_norm", "r_std", "hidden_rel", "hidden_cos", "d_shuffle", "d_anchor")


def _git(args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def cond_id(method, direction, selector, keep_ratio, seed):
    return f"{method}|{direction}|{selector}|k{keep_ratio:g}|s{seed}"


def load_rows(a):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    if a.split == "dev":
        rows = rows[:DEV_CONTEXTS]
    elif a.split == "eval":
        rows = rows[DEV_CONTEXTS:]
    rows = rows[a.shard::a.nshards]
    if a.reverse:
        rows = rows[::-1]
    if a.limit:
        rows = rows[:a.limit]
    return rows


def deletion_conditions(keep_ratio, seeds):
    conds = [("full", "none", "none", 1.0, 0)]
    for s in seeds:
        conds.append(("random", "keep_high", "global", keep_ratio, s))
    conds.append(("recent", "keep_high", "global", keep_ratio, 0))
    for sig in DELETION_SIGNALS:
        for d in ("keep_high", "keep_low"):
            conds.append((sig, d, "global", keep_ratio, 0))
    for sig in ("mlp_norm", "r", "d"):
        conds.append((sig, "keep_high", "layer_matched", keep_ratio, 0))
    conds.append(("random", "keep_high", "layer_matched", keep_ratio, 0))
    conds.append(("d", "keep_high", "boundary", keep_ratio, 0))
    conds.append(("random", "keep_high", "boundary", keep_ratio, 0))
    return conds


def sweep_conditions(keep_ratios, methods):
    conds = [("full", "none", "none", 1.0, 0)]
    for k in keep_ratios:
        for m in methods:
            conds.append((m, "keep_high", "global", k, 0))
    return conds


@torch.no_grad()
def external_scores(model, processor, img, pre, need, device, n_layers, n_heads):
    """context-only 계약 밖 또는 비용이 붙는 점수: attn1 (같은 prefill 의 attention 재계산),
    recon_desc (설명문 생성 + forward; 원 논문 KVzip 과 동일하다고 표기하지 않음)."""
    from vlm_diagnosis.core.kv_select import per_head_column_stats
    from vlm_diagnosis.core.attnstat import QKCapture
    from vlm_diagnosis.core.masked_eval import mrope_position_ids
    from vlm_diagnosis.core.signals import vlm_inputs
    from vlm_diagnosis.exps.core_delta_dram import REPEAT_PROMPT
    out, cost = {}, {}
    P = pre["spans"]["P"]
    if "attn1" in need:
        if pre["qk"] is None:
            raise RuntimeError("attn1 requires capture_qk=True in the prefill")
        _sync(device); t0 = time.perf_counter()
        v0 = int(pre["spans"]["visual"].min())
        mean, _ = per_head_column_stats(pre["qk"], v0, P)
        out["attn1"] = mean[:, :, :P].clone()
        _sync(device); cost["attn1_seconds"] = time.perf_counter() - t0
    if "recon_desc" in need:
        _sync(device); t0 = time.perf_counter()
        ins1 = vlm_inputs(processor, img, REPEAT_PROMPT, device)
        gen = model.generate(**{k: v for k, v in ins1.items()}, max_new_tokens=96, do_sample=False)
        P1, Lg = ins1["input_ids"].shape[1], gen.shape[1]
        attn = torch.ones(1, Lg, dtype=torch.long, device=device)
        pos = mrope_position_ids(model, gen, ins1["image_grid_thw"], attn)
        with QKCapture() as cap:
            model(input_ids=gen, attention_mask=attn, position_ids=pos, pixel_values=ins1["pixel_values"],
                  image_grid_thw=ins1["image_grid_thw"], use_cache=False)
            _, peak = per_head_column_stats(cap.qk, P1, Lg)
        out["recon_desc"] = peak[:, :, :P].clone()
        _sync(device); cost["recon_desc_seconds"] = time.perf_counter() - t0
        cost["recon_desc_forwards"] = 2; cost["recon_desc_generated"] = int(Lg - P1)
        del cap
    return out, cost


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["full", "probe", "deletion", "sweep", "profile"])
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--split", default="dev", choices=["dev", "eval", "all"])
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--keep-ratios", default="0.2")
    ap.add_argument("--methods", default="random,recent,k_norm,v_norm,mlp_norm,r,d")
    ap.add_argument("--with-attn1", action="store_true", help="sweep: context-only attention 점수 추가 (비용 표시)")
    ap.add_argument("--with-reconstruction", action="store_true", help="sweep: 설명문 재구성 점수 추가 (계약 밖, 별도 표기)")
    ap.add_argument("--random-seeds", default="0,1,2,3,4")
    ap.add_argument("--questions-per-context", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--protect-prefix", type=int, default=4)
    ap.add_argument("--profile-method", default="d")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    keep_ratios = sorted({float(x) for x in a.keep_ratios.split(",")}, reverse=True)
    seeds = [int(x) for x in a.random_seeds.split(",")]
    rows = load_rows(a)
    out_rel = a.out or f"results/context_only/{a.stage}_{a.model}_{a.split}.jsonl"
    if a.nshards > 1:
        out_rel = out_rel.replace(".jsonl", f".shard{a.shard}.jsonl")
    out_path = os.path.join(ROOT, out_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    vis = model.model.visual if hasattr(model.model, "visual") else None
    if vis is not None and hasattr(vis, "config"):
        vis.config._attn_implementation = "sdpa"
    N_LAYERS, N_KV, HEAD_DIM = kv_dims(model)
    # 보호 대상 special token: tokenizer special id 중 이미지/비디오 placeholder 는 제외
    # (placeholder 는 context 내용 자체이며, 보호하면 시각 쌍 전부가 예산 밖으로 빠진다)
    placeholders = {int(getattr(model.config, k)) for k in ("image_token_id", "video_token_id") if getattr(model.config, k, None) is not None}
    special_ids = set(int(s) for s in processor.tokenizer.all_special_ids) - placeholders
    run_id = f"co-{a.stage}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    done = set()
    if a.resume and os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                if r.get("record_type") == "context_done":
                    done.add(str(r["context_id"]))
            except json.JSONDecodeError:
                raise SystemExit(f"malformed line in {out_path}; refusing to resume")

    methods = [m for m in a.methods.split(",") if m]
    if a.with_attn1:
        methods.append("attn1")
    if a.with_reconstruction:
        methods.append("recon_desc")
    tc = getattr(model.config, "text_config", model.config)
    run_meta = {
        "record_type": "run", "schema_version": SCHEMA, "run_id": run_id, "stage": a.stage, "model": a.model,
        "model_id": getattr(model.config, "_name_or_path", ""), "code_revision": _git(["rev-parse", "HEAD"]),
        "code_dirty": bool(_git(["status", "--porcelain"])), "transformers": transformers.__version__,
        "torch": torch.__version__, "device": a.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "dtype": str(next(model.parameters()).dtype), "attn_backend": tc._attn_implementation,
        "manifest": a.manifest, "manifest_sha256": _sha(os.path.join(ROOT, a.manifest)), "split": a.split,
        "dev_contexts": DEV_CONTEXTS, "n_contexts": len(rows), "shard": a.shard, "nshards": a.nshards,
        "keep_ratios": keep_ratios, "methods": methods, "random_seeds": seeds, "eps": a.eps,
        "protect": f"prefix first {a.protect_prefix} + tokenizer special ids in prefix (image/video placeholders excluded)",
        "protected_special_ids": sorted(special_ids), "granularity": "kv_pair",
        "storage": "physical ragged deletion", "budget_rule": "B = round(keep_ratio * L*Hkv*T), protected included",
        "tie_rule": "seeded permutation", "nll_rule": "answers[0] body tokens, EOS excluded, teacher forced",
        "questions": f"manifest questions[1:1+{a.questions_per_context}] + BRIEF", "brief": BRIEF,
        "max_new_tokens": a.max_new_tokens, "decode": "greedy", "started_at": datetime.now(timezone.utc).isoformat(),
    }
    f = open(out_path, "a" if a.resume else "w")
    f.write(json.dumps(run_meta) + "\n"); f.flush()

    def emit(rec):
        f.write(json.dumps({"run_id": run_id, **rec}, ensure_ascii=False) + "\n"); f.flush()

    for di, row in enumerate(rows):
        cid = str(row["sample_id"])
        if cid in done:
            continue
        t_ctx = time.time()
        img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
        qs = row["questions"][1:1 + a.questions_per_context]
        try:
            if a.stage == "profile":
                run_profile(model, processor, img, cid, qs, a, emit, special_ids)
            elif a.stage == "full":
                run_full(model, processor, img, cid, qs, a, emit, special_ids)
            elif a.stage == "probe":
                run_probe(model, processor, img, cid, row, a, emit, special_ids, out_path)
            else:
                run_eval(model, processor, img, cid, qs, a, emit, special_ids, keep_ratios, seeds, methods, N_LAYERS, N_KV)
        except Exception as e:  # 기록하고 계속 (누락 condition 을 조용히 숨기지 않음)
            emit({"record_type": "error", "context_id": cid, "stage": a.stage, "error": repr(e)})
            print(f"[error] {cid}: {e!r}", flush=True)
            raise
        emit({"record_type": "context_done", "context_id": cid, "seconds": time.time() - t_ctx})
        print(f"[{di+1}/{len(rows)}] {cid} {time.time()-t_ctx:.0f}s", flush=True)
    f.close()
    print(f"[saved] {out_path}")


def _answer_records(model, mem, cond, method, direction, selector, k, seed, cid, qs, a, emit, pred_full):
    for q in qs:
        q_text = q["question"] + BRIEF
        golds = q["answers"]
        b = mem.clone_owned()
        g = answer_from_cache(model, b, q_text, a.device, a.max_new_tokens)
        del b
        b = mem.clone_owned()
        try:
            n = answer_nll(model, b, q_text, golds[0], a.device)
        except ValueError as e:
            n = {"nll": None, "n_answer_tokens": 0, "error": repr(e)}
        del b
        rec = {"record_type": "answer", "context_id": cid, "question_id": q["question_id"], "condition": cond,
               "method": method, "direction": direction, "selector": selector, "keep_ratio": k, "seed": seed,
               "prediction": g["prediction"], "gold": golds, "em": exact_match(g["prediction"], golds),
               "nll": n["nll"], "n_answer_tokens": n["n_answer_tokens"], "generated_tokens": g["generated_tokens"],
               "hit_generation_limit": g["hit_generation_limit"], "query_seconds": g["query_seconds"],
               "decode_seconds": g["decode_seconds"], "query_peak_bytes": g["query_peak_bytes"], "status": "ok"}
        if pred_full is not None and q["question_id"] in pred_full:
            rec["loyalty"] = float(normalize_text(g["prediction"]) == normalize_text(pred_full[q["question_id"]]))
        else:
            pred_full[q["question_id"]] = g["prediction"]
        emit(rec)


def run_eval(model, processor, img, cid, qs, a, emit, special_ids, keep_ratios, seeds, methods, L, H):
    """deletion / sweep: prefill 1회 (평가 harness 는 여러 방법이 같은 통계를 공유), 방법마다 실제 삭제 캐시 생성."""
    conds = deletion_conditions(keep_ratios[0], seeds) if a.stage == "deletion" else sweep_conditions(keep_ratios, methods)
    need_ext = {m for m in ("attn1", "recon_desc") if any(c[0] == m for c in conds)}
    pre = prefill_context(model, processor, img, a.device, collect_dynamics=True, capture_qk="attn1" in need_ext, eps=a.eps)
    ext, ext_cost = external_scores(model, processor, img, pre, need_ext, a.device, L, H) if need_ext else ({}, {})
    if pre["qk"] is not None:
        pre["qk"] = None
    dyn = pre["dynamics"]
    diag = {"record_type": "diagnostic", "context_id": cid, "n_tokens": pre["spans"]["P"],
            "n_visual": int(pre["spans"]["visual"].numel()), "prefill_seconds": pre["prefill_seconds"],
            "residual_max_rel_err": float(dyn.residual_max_rel_err.max()), "dtype": dyn.dtype, **ext_cost}
    if "recon_desc" in ext:
        from vlm_diagnosis.core.context_only_cache import method_scores
        ref = ext["recon_desc"]
        diag["spearman_vs_recon_desc"] = {}
        for m in methods:
            if m in ("recon_desc",):
                continue
            s, _, _ = method_scores(m, pre, L, H, 0, ext)
            diag["spearman_vs_recon_desc"][m] = spearman_avg_rank(s[:, :, 4:], ref[:, :, 4:])
    emit(diag)
    pred_full = {}
    for (method, direction, selector, k, seed) in conds:
        cond = cond_id(method, direction, selector, k, seed)
        mem, rep, _ = build_memory(model, processor, pre, cid, method, k, seed, a.device, direction=direction if method != "full" else "keep_high",
                                   selector=selector if method != "full" else "global", special_ids=special_ids,
                                   n_prefix_protect=a.protect_prefix, extra_scores=ext)
        emit({"record_type": "build", "context_id": cid, "condition": cond, **report_dict(rep)})
        _answer_records(model, mem, cond, method, direction, selector, k, seed, cid, qs, a, emit, pred_full)
        del mem
    del pre


def run_full(model, processor, img, cid, qs, a, emit, special_ids):
    """단계 1 parity: keep=100% ragged 경로 vs 일반 FULL forward 의 suffix logits, 정답 NLL."""
    pre = prefill_context(model, processor, img, a.device, collect_dynamics=False)
    P = pre["spans"]["P"]
    mem, rep, _ = build_memory(model, processor, pre, cid, "full", 1.0, 0, a.device, special_ids=special_ids)
    emit({"record_type": "build", "context_id": cid, "condition": cond_id("full", "none", "none", 1.0, 0), **report_dict(rep)})
    for q in qs[:2]:
        q_text = q["question"] + BRIEF
        golds = q["answers"]
        ref_logits, ref_ids, ref_pos = dense_reference_logits(model, processor, img, q_text, a.device, P)
        b = mem.clone_owned()
        suffix = b.template.suffix(q_text, first=True)
        if not torch.equal(suffix[0], ref_ids):
            raise RuntimeError("suffix token IDs differ from the dense reference")
        from vlm_diagnosis.core.context_only_cache import _ragged_forward
        from vlm_diagnosis.core.ragged_kv import RaggedAttention
        with RaggedAttention(model, b.cache, collect=False):
            out, _ = _ragged_forward(model, b, suffix, b.next_position, a.device)
        cached = out.logits[0].float()
        expected_pos = torch.arange(b.next_position, b.next_position + suffix.shape[1])
        pos_ok = bool((ref_pos == expected_pos[None].expand(3, -1)).all())
        diff = (cached - ref_logits).abs()
        argmax_agree = float((cached.argmax(-1) == ref_logits.argmax(-1)).float().mean())
        top_last = int(cached[-1].argmax()) == int(ref_logits[-1].argmax())
        del out, b
        # NLL parity: dense [prefix + question + gold] vs cached path
        tok = processor.tokenizer
        ans = tok(golds[0], add_special_tokens=False, return_tensors="pt").input_ids
        b = mem.clone_owned()
        n_c = answer_nll(model, b, q_text, golds[0], a.device)
        del b
        from vlm_diagnosis.core.signals import vlm_inputs
        from vlm_diagnosis.core.masked_eval import mrope_position_ids
        ins = vlm_inputs(processor, img, q_text, a.device)
        ids = torch.cat([ins["input_ids"], ans.to(a.device)], dim=1)
        pos = mrope_position_ids(model, ids, ins["image_grid_thw"], torch.ones_like(ids))
        o = model(input_ids=ids, position_ids=pos, attention_mask=torch.ones_like(ids), pixel_values=ins["pixel_values"],
                  image_grid_thw=ins["image_grid_thw"], use_cache=False)
        S = ins["input_ids"].shape[1]
        lp = torch.log_softmax(o.logits[0, S - 1: ids.shape[1] - 1].float(), -1).gather(1, ans[0].to(a.device)[:, None])[:, 0]
        nll_dense = float(-lp.mean())
        del o
        emit({"record_type": "parity", "context_id": cid, "question_id": q["question_id"], "n_prefix": P,
              "suffix_tokens": int(suffix.shape[1]), "positions_match": pos_ok, "logit_max_abs_diff": float(diff.max()),
              "logit_mean_abs_diff": float(diff.mean()), "argmax_agreement": argmax_agree, "first_answer_token_agree": top_last,
              "nll_cached": n_c["nll"], "nll_dense": nll_dense, "nll_abs_diff": abs(n_c["nll"] - nll_dense),
              "n_answer_tokens": n_c["n_answer_tokens"]})
    del mem, pre


def run_probe(model, processor, img, cid, row, a, emit, special_ids, out_path):
    pre = prefill_context(model, processor, img, a.device, collect_dynamics=True, eps=a.eps)
    dyn = pre["dynamics"]
    table = token_table(dyn, pre["prefix_ids"], processor.tokenizer, special_ids)
    prot = protected_positions(pre["prefix_ids"], special_ids, a.protect_prefix)
    vis = pre["spans"]["visual"]
    vis_mask = torch.zeros(pre["spans"]["P"], dtype=torch.bool); vis_mask[vis] = True
    def summ(x):
        return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}
    diag = {"record_type": "diagnostic", "context_id": cid, "n_tokens": pre["spans"]["P"], "n_visual": int(vis.numel()),
            "n_protected": int(prot.sum()), "dtype": dyn.dtype, "eps": dyn.eps,
            "residual_max_abs_err": dyn.residual_max_abs_err.tolist(), "residual_max_rel_err": dyn.residual_max_rel_err.tolist(),
            "R_layer_mean": dyn.R.mean(1).tolist(), "D_layer_mean": dyn.D.mean(1).tolist(),
            "R_visual": summ(dyn.R[:, vis_mask]), "R_nonvisual": summ(dyn.R[:, ~vis_mask]),
            "D_visual": summ(dyn.D[:, vis_mask]), "D_nonvisual": summ(dyn.D[:, ~vis_mask]),
            "spearman_R_vs_D_token": spearman_avg_rank(dyn.R.mean(0), dyn.D.mean(0)),
            "spearman_R_vs_mlpnorm_token": spearman_avg_rank(dyn.R.mean(0), dyn.mlp_norm.mean(0)),
            "top_tokens_by_D": sorted(table, key=lambda r: -r["D_mean"])[:10],
            "top_tokens_by_R": sorted(table, key=lambda r: -r["R_mean"])[:10],
            "prefill_seconds": pre["prefill_seconds"]}
    emit(diag)
    fig_dir = os.path.join(os.path.dirname(out_path), "probe_figs"); os.makedirs(fig_dir, exist_ok=True)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # raw 와 표시용(보호 token 제외 99% 분위로 clip, 표시에만 사용; selector 에는 적용하지 않음) 두 장
        free = ~prot
        panels = [("R (raw)", dyn.R, None), ("D = |R_l - R_{l-1}| (raw)", dyn.D, None),
                  ("R (display: clipped at p99 of unprotected tokens)", dyn.R, float(dyn.R[:, free].flatten().quantile(0.99))),
                  ("D (display: clipped at p99 of unprotected tokens)", dyn.D, float(dyn.D[:, free].flatten().quantile(0.99)))]
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        for ax, (name, m, vmax) in zip(axes, panels):
            im = ax.imshow(m.numpy(), aspect="auto", interpolation="nearest", vmax=vmax); ax.set_title(f"{cid}: {name}")
            ax.set_xlabel("token"); ax.set_ylabel("layer"); fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"{cid}_R_D.png"), dpi=110); plt.close(fig)
    except Exception as e:
        emit({"record_type": "error", "context_id": cid, "stage": "probe_fig", "error": repr(e)})
    json.dump(table, open(os.path.join(fig_dir, f"{cid}_tokens.json"), "w"), ensure_ascii=False)
    del pre


def run_profile(model, processor, img, cid, qs, a, emit, special_ids):
    """배포 경로 비용: compress_context (dense 해제) 1 build + 질문 비용. --profile-method plain 은 통계 없는 prefill."""
    m = a.profile_method
    k = float(a.keep_ratios.split(",")[0])
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(a.device)
    base_alloc = int(torch.cuda.memory_allocated(a.device)) if torch.cuda.is_available() else 0
    _sync(a.device); t0 = time.perf_counter()
    if m == "plain":
        pre = prefill_context(model, processor, img, a.device, collect_dynamics=False)
        mem, rep, _ = build_memory(model, processor, pre, cid, "full", 1.0, 0, a.device, special_ids=special_ids)
        del pre["kv"]; pre["kv"] = None
        _sync(a.device)
    else:
        mem, rep = compress_context(model, processor, img, m, k, 0, a.device, context_id=cid, special_ids=special_ids,
                                    n_prefix_protect=a.protect_prefix)
    build_wall = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated(a.device)) if torch.cuda.is_available() else 0
    resident = int(torch.cuda.memory_allocated(a.device)) if torch.cuda.is_available() else 0
    emit({"record_type": "build", "context_id": cid, "condition": cond_id(m, "keep_high", "global", k, 0), **report_dict(rep),
          "build_wall_seconds": build_wall, "build_peak_bytes_over_model": peak - base_alloc,
          "resident_bytes_over_model_after_build": resident - base_alloc, "persistent_bytes": mem.kv_bytes + mem.metadata_bytes})
    pred_full = {}
    _answer_records(model, mem, cond_id(m, "keep_high", "global", k, 0), m, "keep_high", "global", k, 0, cid, qs, a, emit, pred_full)
    del mem


if __name__ == "__main__":
    main()
