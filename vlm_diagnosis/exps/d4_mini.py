"""FT-1: D4 mini — K×K 지시 전이 행렬 (PLAN §6.1).

신호 S0(random)/S1_i(SnapKV식, 질문 i)/S3(인코더)/S4(픽셀 분산)/S5(KVzip-VLM) 로
시각 KV를 예산 20%로 압축 → 질문 j를 V2 semantics(질문 도착 전 축출)로 평가.
측정: 정답 teacher-forced logp (full 캐시 대비 Δ).

실행:  python -m vlm_diagnosis.exps.d4_mini --shard 0 --nshards 4 --device cuda:0
집계:  python -m vlm_diagnosis.exps.d4_mini --aggregate
"""
import argparse, glob, json, math, os, time, zlib

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import (
    causal_mask_4d, evict_columns, mrope_position_ids, answer_logp)
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
META = os.path.join(ROOT, "data", "d4_mini", "meta.jsonl")
RESULTS = os.path.join(ROOT, "results", "smoke", "legacy", "d4_mini")
BUDGET = 0.2
K = 4
MAX_PIXELS = 1280 * 28 * 28


@torch.no_grad()
def eval_question(model, processor, img, question, answer, device, keep_sets):
    """질문 1개에 대해 full + 각 keep set의 정답 logp (V2: 시각 끝 직후부터 축출)."""
    ins = S.vlm_inputs(processor, img, question, device)
    ans_ids = processor.tokenizer(answer, add_special_tokens=False,
                                  return_tensors="pt").input_ids.to(device)
    full = torch.cat([ins["input_ids"], ans_ids], 1)
    sp = token_spans(full, model.config)
    vis, vis_end, L = sp["visual"], sp["vis_end"], sp["L"]
    P = ins["input_ids"].shape[1]
    attn2d = torch.ones(1, L, dtype=torch.long, device=device)
    pos = mrope_position_ids(model, full, ins["image_grid_thw"], attn2d)
    kw = dict(input_ids=full, pixel_values=ins["pixel_values"],
              image_grid_thw=ins["image_grid_thw"], answer_start=P)

    m4 = causal_mask_4d(L, device)
    # FULL and keep-set conditions must use the identical 4D attention and
    # explicit mRoPE path. Historical outputs used 2D/no-position FULL and are
    # archived as invalid because that difference confounded every delta.
    lp_full, _ = answer_logp(model, attention_mask=m4, position_ids=pos, **kw)
    out = {"full": lp_full}
    vis_list = vis.tolist()
    for name, keep in keep_sets.items():
        evict = torch.tensor(
            [p for o, p in enumerate(vis_list) if o not in keep], device=device)
        lp, _ = answer_logp(model, attention_mask=evict_columns(m4, evict, vis_end + 1),
                            position_ids=pos, **kw)
        if math.isnan(lp):
            raise RuntimeError(f"NaN logp: {name} — fp32_layers 확장 필요")
        out[name] = lp
    return out, len(vis_list), int(ans_ids.shape[1])


def run(shard, nshards, device, seed):
    os.makedirs(RESULTS, exist_ok=True)
    model, processor = load_qwen25vl(device=device, max_pixels=MAX_PIXELS)
    docs = [json.loads(l) for l in open(META)][shard::nshards]
    outp = os.path.join(RESULTS, f"shard{shard}.jsonl")
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["docId"] for l in open(outp)}
    with open(outp, "a") as f:
        for di, doc in enumerate(docs):
            if doc["docId"] in done:
                continue
            t0 = time.time()
            img = Image.open(doc["image"]).convert("RGB")
            qs = doc["questions"][:K]

            s3 = S.score_s3(model, processor, img, device)
            n_vis = s3.shape[0]
            ins_probe = S.vlm_inputs(processor, img, "x", device)
            s4 = S.score_s4(processor, img, ins_probe["image_grid_thw"].cpu())
            sample_seed = zlib.crc32(f"{seed}:{doc['docId']}".encode()) & 0x7FFFFFFF
            s0 = S.score_s0(n_vis, seed=sample_seed)
            s5 = S.score_s5(model, processor, img, device)
            s1 = [S.score_s1(model, processor, img, q["q"], device) for q in qs]
            for nm, sc in [("S3", s3), ("S4", s4), ("S5", s5)] + [
                    (f"S1_{i}", s1[i]) for i in range(len(qs))]:
                assert sc.shape[0] == n_vis, f"{nm}: {sc.shape[0]} != {n_vis}"

            keeps = {"S0": S.topk_keep(s0, BUDGET), "S3": S.topk_keep(s3, BUDGET),
                     "S4": S.topk_keep(s4, BUDGET), "S5": S.topk_keep(s5, BUDGET)}
            for i in range(len(qs)):
                keeps[f"S1_{i}"] = S.topk_keep(s1[i], BUDGET)

            rows = []
            for j, q in enumerate(qs):
                res, nv, alen = eval_question(
                    model, processor, img, q["q"], q["answers"][0], device, keeps)
                rows.append({"j": j, "qid": q["qid"], "answer": q["answers"][0],
                             "answer_tokens": alen, "logp": res})
            f.write(json.dumps({"schema_version": "legacy-d4-v2", "base_seed": seed,
                                "sample_seed": sample_seed, "docId": doc["docId"],
                                "n_vis": n_vis, "K": len(qs),
                                "rows": rows}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[shard{shard}] {di+1}/{len(docs)} {doc['docId']} "
                  f"n_vis={n_vis} {time.time()-t0:.0f}s", flush=True)


def aggregate():
    files = sorted(glob.glob(os.path.join(RESULTS, "shard*.jsonl")))
    recs = [json.loads(l) for p in files for l in open(p)]
    if not recs:
        print("결과 없음")
        return
    agg = {}  # name -> list of Δlogp(정답 토큰당)
    diag, off = [], []
    for r in recs:
        for row in r["rows"]:
            j, lp, n = row["j"], row["logp"], max(row["answer_tokens"], 1)
            for name, v in lp.items():
                if name == "full":
                    continue
                d = (v - lp["full"]) / n
                if name.startswith("S1_"):
                    (diag if int(name[3:]) == j else off).append(d)
                    agg.setdefault("S1(모든 i)", []).append(d)
                else:
                    agg.setdefault(name, []).append(d)
    print(f"문서 {len(recs)}개, 예산 {BUDGET:.0%}, Δlogp/토큰 (0에 가까울수록 좋음):")
    for name in sorted(agg):
        v = torch.tensor(agg[name])
        print(f"  {name:10s} mean {v.mean():+.3f}  median {v.median():+.3f}  n={len(v)}")
    dg, of = torch.tensor(diag), torch.tensor(off)
    print(f"\n★ S1 대각(질문 일치)  : {dg.mean():+.3f} (n={len(dg)})")
    print(f"★ S1 비대각(재사용)   : {of.mean():+.3f} (n={len(of)})")
    print(f"★ 전이 격차 (대각-비대각): {dg.mean()-of.mean():+.3f}")
    print("LEGACY DIAGNOSTIC ONLY: task metric·T0–T4·M0 gate 없이 발견으로 해석하지 않음")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        run(a.shard, a.nshards, a.device, a.seed)
