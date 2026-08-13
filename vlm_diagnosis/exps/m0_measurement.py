"""M0 — 측정 계약 runner (00_M0_MEASUREMENT.md §6).

검사 (m0.yaml checks):
  image_base                     : 표준 경로 greedy 생성 → task metric (+ HF generate 일치)
  mask_2d_4d                     : 2D implicit-pos vs 4D explicit-pos teacher-forced 정합
  full_mask                      : 전체 시각 차단(V2) 시 생성 점수·logp 하락
  keep100                        : 100% keep 마스크 == plain causal (정확 일치) + 반복 재현성
  cache_identity_strict          : one-shot vs prefill→serialize→load→resume (M0-E1)
  cache_equivalence_operational  : chunked prefill 변형의 수치 범위 (M0-E2)
  v1_v2                          : 질문 KV smuggling (V1 유지·V2 파괴)
  finite                         : 모든 조건 NaN/Inf 부재

실행:
  python -m vlm_diagnosis.exps.m0_measurement \
    --config experiments/configs/m0.yaml \
    --manifest experiments/manifests/m0_sanity.jsonl
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import torch
import yaml
from PIL import Image
from transformers import DynamicCache

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import (
    causal_mask_4d, evict_columns, mrope_position_ids)
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.metrics import score_sample
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
ALL_CHECKS = ("image_base", "mask_2d_4d", "full_mask", "keep100",
              "cache_identity_strict", "cache_equivalence_operational",
              "v1_v2", "finite")


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _finite(t):
    return bool(torch.isfinite(t).all())


def _canonical_answer(row):
    if row.get("acceptable_answers"):
        return row["acceptable_answers"][0]
    c = row["gen_meta"]["center"]
    return f"({c[0]}, {c[1]})"


class SampleContext:
    """한 표본의 공용 텐서: 프롬프트/정답 결합 시퀀스, 스팬, 위치, causal mask."""

    def __init__(self, model, processor, row, device):
        self.row = row
        img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
        self.ins = S.vlm_inputs(processor, img, row["question"], device)
        ans = _canonical_answer(row)
        self.ans_ids = processor.tokenizer(
            ans, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        self.full = torch.cat([self.ins["input_ids"], self.ans_ids], 1)
        sp = token_spans(self.full, model.config)
        self.vis, self.vis_end, self.L = sp["visual"], sp["vis_end"], sp["L"]
        self.P = self.ins["input_ids"].shape[1]
        self.attn2d = torch.ones(1, self.L, dtype=torch.long, device=device)
        self.pos = mrope_position_ids(
            model, self.full, self.ins["image_grid_thw"], self.attn2d)
        self.m4 = causal_mask_4d(self.L, device)
        self.kw = dict(input_ids=self.full, pixel_values=self.ins["pixel_values"],
                       image_grid_thw=self.ins["image_grid_thw"])
        self.device = device


@torch.no_grad()
def _forward_logits(model, ctx, attention_mask, position_ids, use_cache=False):
    out = model(attention_mask=attention_mask, position_ids=position_ids,
                use_cache=use_cache, **ctx.kw)
    return out


def _answer_span_stats(ctx, logits_a, logits_b):
    """정답 구간 logits 비교: max/mean abs diff + greedy 일치."""
    a = logits_a[0, ctx.P - 1:ctx.L - 1].float()
    b = logits_b[0, ctx.P - 1:ctx.L - 1].float()
    d = (a - b).abs()
    return {"max_abs_logit_diff": float(d.max()),
            "mean_abs_logit_diff": float(d.mean()),
            "prediction_equal": bool((a.argmax(-1) == b.argmax(-1)).all())}


def _answer_logp(ctx, logits):
    labels = ctx.full[0, ctx.P:]
    pred = logits[0, ctx.P - 1:ctx.L - 1].float()
    lp = torch.log_softmax(pred, -1)[torch.arange(len(labels)), labels]
    return float(lp.sum()), _finite(lp)


# ---------- 개별 검사 ----------

@torch.no_grad()
def check_image_base(model, processor, ctx, max_new_tokens):
    pred = greedy_generate_masked(model, processor, ctx.ins,
                                  max_new_tokens=max_new_tokens)
    score = score_sample(pred, ctx.row["task_type"],
                         ctx.row.get("acceptable_answers"),
                         ctx.row.get("target_bbox"))
    hf = model.generate(**{k: v for k, v in ctx.ins.items()},
                        max_new_tokens=max_new_tokens, do_sample=False)
    hf_text = processor.tokenizer.decode(
        hf[0, ctx.P:], skip_special_tokens=True).strip()
    return {"check": "image_base", "prediction": pred, "task_score": score,
            "hf_generate_text": hf_text,
            "gen_path_match": pred == hf_text}


@torch.no_grad()
def check_mask_2d_4d(model, ctx):
    a = _forward_logits(model, ctx, ctx.attn2d, None).logits
    b = _forward_logits(model, ctx, ctx.m4, ctx.pos).logits
    rec = {"check": "mask_2d_4d", **_answer_span_stats(ctx, a, b)}
    rec["logp_2d"], f1 = _answer_logp(ctx, a)
    rec["logp_4d"], f2 = _answer_logp(ctx, b)
    rec["finite"] = f1 and f2
    return rec


@torch.no_grad()
def check_full_mask(model, processor, ctx, max_new_tokens, base_score):
    evict_all = ctx.vis
    m = evict_columns(ctx.m4, evict_all, ctx.vis_end + 1)
    logits = _forward_logits(model, ctx, m, ctx.pos).logits
    lp_masked, fin = _answer_logp(ctx, logits)
    lp_full, _ = _answer_logp(
        ctx, _forward_logits(model, ctx, ctx.m4, ctx.pos).logits)
    pred = greedy_generate_masked(model, processor, ctx.ins,
                                  max_new_tokens=max_new_tokens,
                                  evict_cols=evict_all,
                                  row_start=ctx.vis_end + 1)
    score = score_sample(pred, ctx.row["task_type"],
                         ctx.row.get("acceptable_answers"),
                         ctx.row.get("target_bbox"))
    return {"check": "full_mask", "finite": fin,
            "logp_full": lp_full, "logp_masked": lp_masked,
            "delta_logp": lp_masked - lp_full,
            "base_task_score": base_score, "masked_task_score": score,
            "masked_prediction": pred}


@torch.no_grad()
def check_keep100(model, ctx):
    keep = S.topk_keep(torch.ones(len(ctx.vis)), 1.0)
    evict = torch.tensor(
        [p for o, p in enumerate(ctx.vis.tolist()) if o not in keep],
        device=ctx.device, dtype=torch.long)
    m = evict_columns(ctx.m4, evict, ctx.vis_end + 1)
    mask_equal = bool(torch.equal(m, ctx.m4))
    a = _forward_logits(model, ctx, ctx.m4, ctx.pos).logits
    b = _forward_logits(model, ctx, m, ctx.pos).logits
    rec = {"check": "keep100", "mask_exactly_equal": mask_equal,
           "n_evicted_at_100pct": int(evict.numel()),
           **_answer_span_stats(ctx, a, b)}
    rec["finite"] = _finite(a) and _finite(b)
    return rec


def _serialize_roundtrip(cache, path):
    legacy = cache.to_legacy_cache()
    torch.save([(k.cpu(), v.cpu()) for k, v in legacy], path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    bitexact = all(torch.equal(k0.cpu(), k1) and torch.equal(v0.cpu(), v1)
                   for (k0, v0), (k1, v1) in zip(legacy, loaded))
    dev = legacy[0][0].device
    rebuilt = DynamicCache.from_legacy_cache(
        tuple((k.to(dev), v.to(dev)) for k, v in loaded))
    return rebuilt, bitexact


@torch.no_grad()
def check_cache_identity_strict(model, ctx, tmp_path):
    one = model(input_ids=ctx.full, attention_mask=ctx.attn2d,
                position_ids=ctx.pos, use_cache=True,
                pixel_values=ctx.ins["pixel_values"],
                image_grid_thw=ctx.ins["image_grid_thw"])
    pre = model(input_ids=ctx.ins["input_ids"],
                attention_mask=ctx.attn2d[:, :ctx.P],
                position_ids=ctx.pos[:, :, :ctx.P], use_cache=True,
                pixel_values=ctx.ins["pixel_values"],
                image_grid_thw=ctx.ins["image_grid_thw"])
    cache, bitexact = _serialize_roundtrip(pre.past_key_values, tmp_path)
    res = model(input_ids=ctx.ans_ids, attention_mask=ctx.attn2d,
                position_ids=ctx.pos[:, :, ctx.P:], past_key_values=cache,
                use_cache=True)
    a = one.logits[0, ctx.P:].float()
    b = res.logits[0].float()
    d = (a - b).abs()
    one_legacy = one.past_key_values.to_legacy_cache()
    res_legacy = res.past_key_values.to_legacy_cache()
    kv_diff = max(float((k0[:, :, :ctx.P] - k1[:, :, :ctx.P]).abs().max())
                  for (k0, _), (k1, _) in zip(one_legacy, res_legacy))
    os.remove(tmp_path)
    return {"check": "cache_identity_strict",
            "serialize_roundtrip_bitexact": bitexact,
            "max_abs_logit_diff": float(d.max()),
            "mean_abs_logit_diff": float(d.mean()),
            "prediction_equal": bool((a.argmax(-1) == b.argmax(-1)).all()),
            "max_abs_prefix_kv_diff": kv_diff,
            "finite": _finite(one.logits) and _finite(res.logits)}


@torch.no_grad()
def check_cache_equivalence_operational(model, ctx):
    one = model(input_ids=ctx.full, attention_mask=ctx.attn2d,
                position_ids=ctx.pos, use_cache=False,
                pixel_values=ctx.ins["pixel_values"],
                image_grid_thw=ctx.ins["image_grid_thw"])
    ref = one.logits[0, ctx.P:].float()
    out = {"check": "cache_equivalence_operational", "variants": []}
    for name, split in (("split_after_image", ctx.vis_end + 3),
                        ("split_late", max(ctx.P - 4, ctx.vis_end + 2))):
        c1 = model(input_ids=ctx.full[:, :split],
                   attention_mask=ctx.attn2d[:, :split],
                   position_ids=ctx.pos[:, :, :split], use_cache=True,
                   pixel_values=ctx.ins["pixel_values"],
                   image_grid_thw=ctx.ins["image_grid_thw"])
        c2 = model(input_ids=ctx.full[:, split:ctx.P],
                   attention_mask=ctx.attn2d[:, :ctx.P],
                   position_ids=ctx.pos[:, :, split:ctx.P],
                   past_key_values=c1.past_key_values, use_cache=True)
        res = model(input_ids=ctx.ans_ids, attention_mask=ctx.attn2d,
                    position_ids=ctx.pos[:, :, ctx.P:],
                    past_key_values=c2.past_key_values, use_cache=True)
        b = res.logits[0].float()
        d = (ref - b).abs()
        out["variants"].append({
            "variant": name, "split": int(split),
            "max_abs_logit_diff": float(d.max()),
            "mean_abs_logit_diff": float(d.mean()),
            "prediction_equal": bool((ref.argmax(-1) == b.argmax(-1)).all()),
            "finite": _finite(b)})
    out["finite"] = all(v["finite"] for v in out["variants"])
    return out


@torch.no_grad()
def check_v1_v2(model, processor, ctx, max_new_tokens):
    evict_all = ctx.vis
    rec = {"check": "v1_v2"}
    lp_full, _ = _answer_logp(
        ctx, _forward_logits(model, ctx, ctx.m4, ctx.pos).logits)
    for tag, row_start in (("v1", ctx.P), ("v2", ctx.vis_end + 1)):
        m = evict_columns(ctx.m4, evict_all, row_start)
        lp, fin = _answer_logp(ctx, _forward_logits(model, ctx, m, ctx.pos).logits)
        pred = greedy_generate_masked(model, processor, ctx.ins,
                                      max_new_tokens=max_new_tokens,
                                      evict_cols=evict_all, row_start=row_start)
        score = score_sample(pred, ctx.row["task_type"],
                             ctx.row.get("acceptable_answers"),
                             ctx.row.get("target_bbox"))
        rec[f"{tag}_delta_logp"] = lp - lp_full
        rec[f"{tag}_task_score"] = score
        rec[f"{tag}_finite"] = fin
    rec["finite"] = rec["v1_finite"] and rec["v2_finite"]
    return rec


# ---------- 실행 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/m0.yaml")
    ap.add_argument("--manifest", default="experiments/manifests/m0_sanity.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--checks", default=None, help="쉼표 목록 (기본: 전부)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, a.config)))
    device = a.device or cfg["model"]["device"]
    checks = tuple(a.checks.split(",")) if a.checks else ALL_CHECKS
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    if a.limit:
        rows = rows[:a.limit]

    model, processor = load_qwen25vl(model_id=cfg["model"]["id"], device=device)
    out_path = os.path.join(ROOT, cfg["output"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_id = f"m0-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    thresholds = cfg.get("thresholds", {})

    meta = {"record_type": "run_metadata", "schema_version": "1.1",
            "run_id": run_id, "stage": "M0", "run_kind": cfg["run_kind"],
            "source_code_revision": _git_rev(),
            "config_path": a.config, "config_sha256": _sha256(os.path.join(ROOT, a.config)),
            "manifest_path": a.manifest,
            "manifest_sha256": _sha256(os.path.join(ROOT, a.manifest)),
            "model_id": cfg["model"]["id"], "model_revision": cfg["model"]["revision"],
            "processor_mode": cfg["model"]["processor_mode"],
            "dtype": cfg["model"]["dtype"], "device": device,
            "base_seed": cfg["seed"], "checks": list(checks),
            "thresholds": thresholds,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed": False}

    records = []
    t_start = time.time()
    for i, row in enumerate(rows):
        ctx = SampleContext(model, processor, row, device)
        base = {"run_id": run_id, "dataset": row["dataset"], "split": row["split"],
                "sample_id": row["sample_id"], "question_id": None,
                "task_type": row["task_type"], "condition_id": "m0_check",
                "selection_timing": "none"}
        base_score = None
        for chk in checks:
            if chk == "finite":
                continue  # 각 검사 record의 finite 필드로 집계
            t0 = time.time()
            if chk == "image_base":
                rec = check_image_base(model, processor, ctx, a.max_new_tokens)
                base_score = rec["task_score"]
            elif chk == "mask_2d_4d":
                rec = check_mask_2d_4d(model, ctx)
            elif chk == "full_mask":
                rec = check_full_mask(model, processor, ctx,
                                      a.max_new_tokens, base_score)
            elif chk == "keep100":
                rec = check_keep100(model, ctx)
            elif chk == "cache_identity_strict":
                rec = check_cache_identity_strict(
                    model, ctx, out_path + f".cache_tmp_{i}.pt")
            elif chk == "cache_equivalence_operational":
                rec = check_cache_equivalence_operational(model, ctx)
            elif chk == "v1_v2":
                rec = check_v1_v2(model, processor, ctx, a.max_new_tokens)
            else:
                raise ValueError(f"unknown check {chk}")
            rec.update(base)
            rec["threshold_used"] = None
            rec["elapsed_s"] = round(time.time() - t0, 2)
            records.append(rec)
        print(f"[{i+1}/{len(rows)}] {row['sample_id']} done "
              f"({time.time()-t_start:.0f}s)", flush=True)
        torch.cuda.empty_cache()

    meta["completed"] = True
    meta["total_elapsed_s"] = round(time.time() - t_start, 1)
    with open(out_path, "w") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}")
    _write_report(cfg, meta, records, thresholds)


def _agg(records, chk):
    return [r for r in records if r["check"] == chk]


def _write_report(cfg, meta, records, thresholds):
    lines = [f"# M0 measurement report — {meta['run_id']}", "",
             f"- source: `{meta['source_code_revision']}`",
             f"- model: `{meta['model_id']}` @ `{meta['model_revision']}`",
             f"- samples: {len({r['sample_id'] for r in records})}", ""]

    def fmt(x):
        return f"{x:.3g}" if isinstance(x, float) else str(x)

    for chk in ("image_base", "mask_2d_4d", "keep100", "cache_identity_strict",
                "cache_equivalence_operational", "full_mask", "v1_v2"):
        rs = _agg(records, chk)
        if not rs:
            continue
        lines.append(f"## {chk}")
        if chk == "image_base":
            by = {}
            for r in rs:
                by.setdefault(r["task_type"], []).append(r["task_score"])
            for t, v in sorted(by.items()):
                lines.append(f"- {t}: mean score {sum(v)/len(v):.2f} (n={len(v)})")
            n_match = sum(r["gen_path_match"] for r in rs)
            lines.append(f"- masked-generate vs HF generate 일치: {n_match}/{len(rs)}")
        elif chk in ("mask_2d_4d", "keep100", "cache_identity_strict"):
            mx = max(r["max_abs_logit_diff"] for r in rs)
            pe = sum(r["prediction_equal"] for r in rs)
            lines.append(f"- max_abs_logit_diff (worst): {fmt(mx)}")
            lines.append(f"- prediction_equal: {pe}/{len(rs)}")
            if chk == "keep100":
                ok = sum(r["mask_exactly_equal"] for r in rs)
                lines.append(f"- mask_exactly_equal: {ok}/{len(rs)}")
            if chk == "cache_identity_strict":
                bx = sum(r["serialize_roundtrip_bitexact"] for r in rs)
                kv = max(r["max_abs_prefix_kv_diff"] for r in rs)
                lines.append(f"- serialize roundtrip bitexact: {bx}/{len(rs)}")
                lines.append(f"- max prefix KV diff: {fmt(kv)}")
        elif chk == "cache_equivalence_operational":
            for variant in ("split_after_image", "split_late"):
                ds = [v["max_abs_logit_diff"] for r in rs
                      for v in r["variants"] if v["variant"] == variant]
                pe = sum(v["prediction_equal"] for r in rs
                         for v in r["variants"] if v["variant"] == variant)
                lines.append(f"- {variant}: max diff worst {fmt(max(ds))}, "
                             f"median {fmt(sorted(ds)[len(ds)//2])}, "
                             f"prediction_equal {pe}/{len(ds)}")
        elif chk == "full_mask":
            drop = [r["base_task_score"] - r["masked_task_score"] for r in rs
                    if r["base_task_score"] is not None]
            dlp = [r["delta_logp"] for r in rs]
            lines.append(f"- task score drop mean: {sum(drop)/max(len(drop),1):.2f}")
            lines.append(f"- delta_logp mean: {sum(dlp)/len(dlp):.2f}")
        elif chk == "v1_v2":
            v1 = sum(r["v1_task_score"] for r in rs) / len(rs)
            v2 = sum(r["v2_task_score"] for r in rs) / len(rs)
            lines.append(f"- V1 mean task score: {v1:.2f} (smuggling 경로)")
            lines.append(f"- V2 mean task score: {v2:.2f} (기본 semantics)")
        n_fin = sum(bool(r.get("finite", True)) for r in rs)
        lines.append(f"- finite: {n_fin}/{len(rs)}")
        lines.append("")

    lines.append("## thresholds")
    for k, v in thresholds.items():
        lines.append(f"- {k}: {v if v is not None else 'null (측정 후 고정 필요)'}")
    report = os.path.join(os.path.dirname(os.path.join(ROOT, cfg["output"])),
                          "m0_report.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[saved] {report}")


if __name__ == "__main__":
    main()
