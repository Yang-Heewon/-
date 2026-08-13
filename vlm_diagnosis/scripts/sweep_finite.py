"""M0-04 검증 sweep — d4_mini 32문서 × 4질문에서 finite 전수 확인.

경로: full_4d, evict(legacy seed), evict(derived seed). NaN이 나오면 기록만 하고
계속 진행한다 (실패 지도를 얻기 위해). 결과: results/smoke/nan_diagnosis/sweep.jsonl

  python -m vlm_diagnosis.scripts.sweep_finite --device cuda:1
"""
import argparse
import json
import os
import time
import zlib

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import (
    causal_mask_4d, evict_columns, mrope_position_ids, answer_logp)
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
META = os.path.join(ROOT, "data", "d4_mini", "meta.jsonl")
OUT = os.path.join(ROOT, "results", "smoke", "nan_diagnosis", "sweep.jsonl")
MAX_PIXELS = 1280 * 28 * 28
BUDGET = 0.2


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32-layers", default="auto")
    a = ap.parse_args()
    fp32 = "auto" if a.fp32_layers == "auto" else (
        () if a.fp32_layers == "none" else
        tuple(int(x) for x in a.fp32_layers.split(",")))

    model, processor = load_qwen25vl(device=a.device, max_pixels=MAX_PIXELS,
                                     fp32_layers=fp32)
    docs = [json.loads(l) for l in open(META)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_ok = n_bad = 0
    with open(OUT, "w") as f:
        for di, doc in enumerate(docs):
            img = Image.open(doc["image"]).convert("RGB")
            t0 = time.time()
            for qi, q in enumerate(doc["questions"][:4]):
                ins = S.vlm_inputs(processor, img, q["q"], a.device)
                ans = processor.tokenizer(q["answers"][0], add_special_tokens=False,
                                          return_tensors="pt").input_ids.to(a.device)
                full = torch.cat([ins["input_ids"], ans], 1)
                sp = token_spans(full, model.config)
                vis, vis_end, L = sp["visual"], sp["vis_end"], sp["L"]
                P = ins["input_ids"].shape[1]
                attn2d = torch.ones(1, L, dtype=torch.long, device=a.device)
                pos = mrope_position_ids(model, full, ins["image_grid_thw"], attn2d)
                kw = dict(input_ids=full, pixel_values=ins["pixel_values"],
                          image_grid_thw=ins["image_grid_thw"], answer_start=P)
                m4 = causal_mask_4d(L, a.device)
                conds = {"full_4d": m4}
                vis_list = vis.tolist()
                for tag, seed in (
                        ("legacy", zlib.crc32(str(doc["docId"]).encode()) & 0x7FFFFFFF),
                        ("derived", zlib.crc32(f"{a.seed}:{doc['docId']}".encode()) & 0x7FFFFFFF)):
                    keep = S.topk_keep(S.score_s0(len(vis_list), seed=seed), BUDGET)
                    ev = torch.tensor(
                        [p for o, p in enumerate(vis_list) if o not in keep],
                        device=a.device)
                    conds[f"evict_{tag}"] = evict_columns(m4, ev, vis_end + 1)
                rec = {"docId": doc["docId"], "q": qi, "L": int(L),
                       "n_vis": len(vis_list), "logp": {}, "finite": {}}
                for name, mask in conds.items():
                    lp, tok = answer_logp(model, attention_mask=mask,
                                          position_ids=pos, **kw)
                    fin = bool(torch.isfinite(tok).all())
                    rec["logp"][name] = lp
                    rec["finite"][name] = fin
                    n_ok, n_bad = n_ok + fin, n_bad + (not fin)
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print(f"[{di+1}/{len(docs)}] doc={doc['docId']} "
                  f"ok={n_ok} bad={n_bad} {time.time()-t0:.0f}s", flush=True)
    print(f"DONE ok={n_ok} bad={n_bad} → {OUT}", flush=True)


if __name__ == "__main__":
    main()
