"""A안 — read-time 커버리지 감지기의 원천 데이터 수집.

질문: "남긴 부분집합이 새 질문을 못 덮는다는 것을, 답을 생성하기 전에 알 수 있는가?"

이미지마다 held-out/교차 실험(m3_pilot_cross_eval)과 완전히 같은 부분집합
(S_q0..2 / UNION / S5_MATCHED / RANDOM_MATCHED)을 재구성하고, 각 평가 질문의
s1 attention(그 질문이 시각 토큰에 주는 참조 분포)을 확률로 정규화한 뒤
부분집합 위에 떨어지는 비중 = coverage 를 기록한다. 생성은 하지 않는다 —
정답/오답은 기존 실험 로그와 (sample_id, subset, question, budget)으로 join한다.

  python -m vlm_diagnosis.exps.coverage_probe \
    --manifest experiments/manifests/gqa_transfer.jsonl --device cuda:2
"""
import argparse
import json
import os
import time
import zlib

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.kv_baselines import KVShape, dense_storage, max_keep_for_budget
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."
N_LAYERS, N_KV_HEADS, HEAD_DIM = 28, 4, 128


def coverage(s1_scores, keep):
    p = s1_scores.clamp(min=0)
    tot = float(p.sum())
    if tot <= 0:
        return None
    idx = torch.tensor(sorted(keep), dtype=torch.long)
    return round(float(p[idx].sum()) / tot, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budgets", default="0.05,0.2")
    ap.add_argument("--seed", type=int, default=42)   # m3 runner와 동일해야 함
    ap.add_argument("--out", default="results/smoke/coverage_probe.jsonl")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    budgets = [float(x) for x in a.budgets.split(",")]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    rows = rows[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
    if a.nshards > 1:
        a.out = a.out.replace(".jsonl", f".shard{a.shard}.jsonl")
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if a.resume and os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(str(json.loads(line)["sample_id"]))
            except Exception:
                pass
        print(f"[resume] {len(done)}장 건너뜀", flush=True)
    model, processor = load_qwen25vl(device=a.device, max_pixels=MAX_PIXELS)

    with open(out_path, "a" if a.resume else "w") as f:
        for di, row in enumerate(rows):
            if str(row["sample_id"]) in done:
                continue
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs = row["questions"][1:4]           # source (m3와 동일)
            qs_eval = row["questions"][4:6]      # heldout (m3와 동일)
            if len(qs) < 2:
                continue
            s1_src = [S.score_s1(model, processor, img,
                                 q["question"] + BRIEF, a.device).cpu()
                      for q in qs]
            s1_ho = [S.score_s1(model, processor, img,
                                q["question"] + BRIEF, a.device).cpu()
                     for q in qs_eval]
            s5_doc = S.score_s5(model, processor, img, a.device).cpu()
            n_vis = s1_src[0].shape[0]
            shape = KVShape(layers=N_LAYERS, batch=1, kv_heads=N_KV_HEADS,
                            tokens=n_vis, head_dim=HEAD_DIM)
            e = dense_storage(shape)
            full_bytes = e.payload_bytes + e.metadata_bytes + e.position_bytes
            gen = torch.Generator().manual_seed(
                zlib.crc32(f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF)

            for B in budgets:                    # budget 순서도 m3와 동일 (generator 상태)
                k = max_keep_for_budget(shape, int(B * full_bytes), "sparse")
                keeps = [set(torch.topk(s, min(k, n_vis)).indices.tolist())
                         for s in s1_src]
                union = set().union(*keeps)
                m = len(union)
                subsets = {f"S_q{i}": ks for i, ks in enumerate(keeps)}
                subsets["UNION"] = union
                subsets["S5_MATCHED"] = set(
                    torch.topk(s5_doc, min(m, n_vis)).indices.tolist())
                subsets["RANDOM_MATCHED"] = set(
                    torch.randperm(n_vis, generator=gen)[:m].tolist())
                evals = ([("cross", j, q, s1_src[j]) for j, q in enumerate(qs)]
                         + [("heldout", j, q, s1_ho[j])
                            for j, q in enumerate(qs_eval)])
                for src, keep in subsets.items():
                    for mode, j, q, s1e in evals:
                        cov = coverage(s1e, keep)
                        f.write(json.dumps({
                            "sample_id": row["sample_id"], "eval_mode": mode,
                            "subset_from": src, "eval_q_idx": j,
                            "eval_question_id": q["question_id"],
                            "budget_per_question": B,
                            "keep_tokens": len(keep), "n_visual": n_vis,
                            "coverage": cov,
                            "cov_uniform": round(len(keep) / n_vis, 6)},
                            ensure_ascii=False) + "\n")
                f.flush()
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} n_vis={n_vis} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
