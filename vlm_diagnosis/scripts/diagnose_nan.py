"""M0-04 — fp16 masked-path NaN localization (m0.yaml nan_diagnosis.trace 구현).

재현 대상: legacy d4 shard0, docId 4733, 질문 0, S0(random 20% keep), V2 eviction.
세 경로를 같은 입력으로 비교한다.

  full_2d   : 2D attention mask, implicit position (legacy FULL 경로)
  full_4d   : 4D causal mask + explicit mRoPE position_ids
  evict_s0  : full_4d + S0 컬럼 축출 (vis_end+1 행부터)

trace 순서 (m0.yaml):
  1. mask_rows            — 축출 mask에서 모든 query 행에 유효 key가 남는가
  2. qkv_by_layer         — layer별 q/k/v projection 출력 finite
  3. attention_prob_by_layer — post-RoPE QK^T logit의 fp32 max|.|와 fp16 overflow 수,
                               필요 시 해당 layer의 fp16 softmax NaN 행 수 재현
  4. hidden_state_by_layer — layer 입력 hidden state finite + absmax

사용:
  python -m vlm_diagnosis.scripts.diagnose_nan --device cuda:0
  python -m vlm_diagnosis.scripts.diagnose_nan --fp32-layers none   # 패치 없이 기준 지도
"""
import argparse
import json
import math
import os
import zlib

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import (
    causal_mask_4d, evict_columns, mrope_position_ids)
from vlm_diagnosis.core.attnstat import QKCapture
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
META = os.path.join(ROOT, "data", "d4_mini", "meta.jsonl")
OUT_DIR = os.path.join(ROOT, "results", "smoke", "nan_diagnosis")
FP16_MAX = 65504.0
MAX_PIXELS = 1280 * 28 * 28  # legacy d4와 동일


def _finite(t):
    return bool(torch.isfinite(t).all())


class LayerTrace:
    """layer별 hidden/q/k/v/attn/mlp finite 기록 (텐서 저장 없음 — V100 안전)."""

    ORDER = ("hidden_in", "q", "k", "v", "attn_out", "mlp_out")

    def __init__(self, model):
        lm = (model.model.language_model
              if hasattr(model.model, "language_model") else model.model)
        self.layers = lm.layers
        self.records = [{} for _ in self.layers]
        self.handles = []

    def _post(self, i, key):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.records[i][key] = _finite(t)
        return hook

    def _pre(self, i):
        def hook(_mod, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            self.records[i]["hidden_in"] = _finite(hs)
            self.records[i]["hidden_in_absmax"] = float(hs.abs().max())
        return hook

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(
                self._pre(i), with_kwargs=True))
            for key, mod in (("q", layer.self_attn.q_proj),
                             ("k", layer.self_attn.k_proj),
                             ("v", layer.self_attn.v_proj),
                             ("attn_out", layer.self_attn),
                             ("mlp_out", layer.mlp)):
                self.handles.append(mod.register_forward_hook(self._post(i, key)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False

    def first_bad(self):
        for i, rec in enumerate(self.records):
            for k in self.ORDER:
                if rec.get(k) is False:
                    return {"layer": i, "site": k}
        return None


def mask_rows_check(mask4d):
    """모든 query 행에 유효(0) key가 최소 1개 남는가."""
    visible = (mask4d[0, 0] == 0).any(dim=-1)
    bad = (~visible).nonzero(as_tuple=True)[0]
    return {"all_rows_have_key": bool(visible.all()),
            "bad_rows": bad.tolist()[:20]}


@torch.no_grad()
def qk_logit_map(qk, chunk=512):
    """layer별 post-RoPE QK^T/sqrt(d)의 fp32 max|logit|과 fp16 overflow 원소 수."""
    rows = []
    for li, (q, k) in enumerate(qk):
        q0, k0 = q[0].float(), k[0].float()
        H, L, d = q0.shape
        k0 = k0.repeat_interleave(H // k0.shape[0], dim=0)
        mx, over = 0.0, 0
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            w = q0[:, s:e] @ k0.transpose(-1, -2) / math.sqrt(d)
            mx = max(mx, float(w.abs().max()))
            over += int((w.abs() > FP16_MAX).sum())
            del w
        rows.append({"layer": li, "max_abs_logit_fp32": mx,
                     "n_over_fp16_max": over, "captured_dtype": str(q.dtype)})
    return rows


@torch.no_grad()
def reproduce_fp16_softmax(qk, layer_idx, mask4d, chunk=512):
    """의심 layer의 attention을 모델과 같은 방식(fp16 matmul+mask, fp32 softmax)으로
    재계산해 inf logit 수와 NaN이 되는 (head,row) 수를 센다."""
    q, k = qk[layer_idx]
    q0, k0 = q[0].half(), k[0].half()
    H, L, d = q0.shape
    k0 = k0.repeat_interleave(H // k0.shape[0], dim=0)
    m = mask4d[0, 0].half()
    n_inf, n_nan_rows = 0, 0
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        w = q0[:, s:e] @ k0.transpose(-1, -2) / math.sqrt(d)   # fp16 matmul (모델과 동일)
        n_inf += int(torch.isinf(w).sum())
        w = w + m[s:e][None]
        p = torch.softmax(w.float(), dim=-1)                    # 모델도 fp32 softmax
        n_nan_rows += int(torch.isnan(p).any(dim=-1).sum())
        del w, p
    return {"layer": layer_idx, "n_inf_logits_fp16": n_inf,
            "n_nan_softmax_rows": n_nan_rows}


@torch.no_grad()
def run_condition(model, name, kw, attention_mask, position_ids, answer_start):
    with LayerTrace(model) as tr, QKCapture() as cap:
        out = model(attention_mask=attention_mask, position_ids=position_ids,
                    use_cache=False, **kw)
    logits = out.logits.float()
    labels = kw["input_ids"][0, answer_start:]
    pred = logits[0, answer_start - 1:-1]
    tok = torch.log_softmax(pred, -1)[torch.arange(len(labels)), labels]
    rec = {
        "condition": name,
        "logits_finite": _finite(out.logits),
        "answer_logp": float(tok.sum()),
        "first_nonfinite_site": tr.first_bad(),
        "hidden_absmax_by_layer": [r.get("hidden_in_absmax") for r in tr.records],
        "layer_finite": [{k: r.get(k) for k in LayerTrace.ORDER} for r in tr.records],
        "qk_logit_map": qk_logit_map(cap.qk),
    }
    overflow_layers = [r["layer"] for r in rec["qk_logit_map"]
                       if r["n_over_fp16_max"] > 0 and "float32" not in r["captured_dtype"]]
    rec["fp16_qk_overflow_layers"] = overflow_layers
    # fp16 softmax NaN 재현: overflow layer + first_bad layer에 대해서만
    suspects = set(overflow_layers)
    if rec["first_nonfinite_site"]:
        suspects.add(rec["first_nonfinite_site"]["layer"])
    if attention_mask is not None and attention_mask.dim() == 4:
        rec["softmax_repro"] = [
            reproduce_fp16_softmax(cap.qk, li, attention_mask) for li in sorted(suspects)]
    del cap
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--doc-id", type=int, default=4733)
    ap.add_argument("--question-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--fp32-layers", default="auto", help="auto | none | 쉼표목록(예 26,27)")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--legacy-seed", action="store_true",
                    help="구 d4 seed 유도 crc32(str(docId)) 사용 — shard0 실패 정확 재현")
    ap.add_argument("--all-questions", action="store_true",
                    help="문서의 K=4 질문 전부에 대해 조건 평가")
    a = ap.parse_args()

    if a.fp32_layers == "auto":
        fp32 = "auto"
    elif a.fp32_layers == "none":
        fp32 = ()
    else:
        fp32 = tuple(int(x) for x in a.fp32_layers.split(","))

    doc = next(d for d in (json.loads(l) for l in open(META)) if d["docId"] == a.doc_id)
    img = Image.open(doc["image"]).convert("RGB")
    q_indices = range(len(doc["questions"][:4])) if a.all_questions else [a.question_idx]

    model, processor = load_qwen25vl(device=a.device, max_pixels=a.max_pixels,
                                     fp32_layers=fp32)
    if a.legacy_seed:
        sample_seed = zlib.crc32(str(doc["docId"]).encode()) & 0x7FFFFFFF
    else:
        sample_seed = zlib.crc32(f"{a.seed}:{doc['docId']}".encode()) & 0x7FFFFFFF

    report = {
        "doc_id": a.doc_id, "budget": a.budget, "sample_seed": sample_seed,
        "legacy_seed": a.legacy_seed, "fp32_layers": a.fp32_layers,
        "max_pixels": a.max_pixels, "questions": [],
    }
    for qi in q_indices:
        q = doc["questions"][qi]
        ins = S.vlm_inputs(processor, img, q["q"], a.device)
        ans_ids = processor.tokenizer(q["answers"][0], add_special_tokens=False,
                                      return_tensors="pt").input_ids.to(a.device)
        full = torch.cat([ins["input_ids"], ans_ids], 1)
        sp = token_spans(full, model.config)
        vis, vis_end, L = sp["visual"], sp["vis_end"], sp["L"]
        P = ins["input_ids"].shape[1]
        attn2d = torch.ones(1, L, dtype=torch.long, device=a.device)
        pos = mrope_position_ids(model, full, ins["image_grid_thw"], attn2d)
        kw = dict(input_ids=full, pixel_values=ins["pixel_values"],
                  image_grid_thw=ins["image_grid_thw"])

        m4 = causal_mask_4d(L, a.device)
        s0 = S.score_s0(vis.shape[0], seed=sample_seed)
        keep = S.topk_keep(s0, a.budget)
        vis_list = vis.tolist()
        evict = torch.tensor([p for o, p in enumerate(vis_list) if o not in keep],
                             device=a.device)
        m_evict = evict_columns(m4, evict, vis_end + 1)

        qrec = {
            "question_idx": qi, "qid": q["qid"], "question": q["q"],
            "n_visual": len(vis_list), "L": int(L), "prompt_len": int(P),
            "mask_rows": {
                "full_4d": mask_rows_check(m4),
                "evict_s0": mask_rows_check(m_evict),
            },
            "conditions": [],
        }
        print(f"[setup] doc={a.doc_id} q{qi} L={L} n_vis={len(vis_list)} "
              f"evicted={len(evict)} seed={sample_seed} fp32={a.fp32_layers}", flush=True)
        print(f"[mask_rows] full_4d ok={qrec['mask_rows']['full_4d']['all_rows_have_key']} "
              f"evict ok={qrec['mask_rows']['evict_s0']['all_rows_have_key']}", flush=True)

        for name, mask, p in (("full_2d", attn2d, None),
                              ("full_4d", m4, pos),
                              ("evict_s0", m_evict, pos)):
            rec = run_condition(model, name, kw, mask, p, P)
            qrec["conditions"].append(rec)
            bad = rec["first_nonfinite_site"]
            print(f"[q{qi} {name}] logits_finite={rec['logits_finite']} "
                  f"logp={rec['answer_logp']:.3f} first_bad={bad} "
                  f"fp16_qk_overflow_layers={rec['fp16_qk_overflow_layers']}", flush=True)
            for r in rec.get("softmax_repro", []):
                print(f"    softmax_repro layer{r['layer']}: "
                      f"inf_logits={r['n_inf_logits_fp16']} "
                      f"nan_rows={r['n_nan_softmax_rows']}", flush=True)
        report["questions"].append(qrec)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "legacy" if a.legacy_seed else str(a.seed)
    out = os.path.join(
        OUT_DIR,
        f"doc{a.doc_id}_seed-{tag}_fp32-{a.fp32_layers.replace(',', '_')}.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
