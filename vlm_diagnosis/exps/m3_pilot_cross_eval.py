"""M3 파일럿 — s1 부분집합의 질문 간 교차 평가 (T0–T4 라벨 없는 총량 신호).

질문: "한 질문에 최적인 시각 KV 부분집합이 같은 이미지의 다른 질문도 커버하는가?"
  - S_qi = 질문 i의 attention(s1)으로 고른 예산 B 부분집합
  - 교차 평가: S_qi를 질문 j 생성에 사용 (i=j 자기 재현 / i≠j 전이)
  - UNION   = 문서 내 세 질문의 S_qi 합집합 (실제 크기 기록 — '작은 만능 집합' 존재 시험)

주의: T0–T4 라벨 전이므로 결과는 총량 신호로만 해석하고 유형별 주장을 하지 않는다.

실행:
  python -m vlm_diagnosis.exps.m3_pilot_cross_eval --shard 0 --nshards 4 --device cuda:0
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.metrics import anls, exact_match
from vlm_diagnosis.core.kv_baselines import KVShape, dense_storage, max_keep_for_budget
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."
N_LAYERS, N_KV_HEADS, HEAD_DIM = 28, 4, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/m2a_diagnostic.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budgets", default="0.05,0.2")
    ap.add_argument("--questions-per-doc", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="results/smoke/m3_pilot.jsonl")
    a = ap.parse_args()

    budgets = [float(x) for x in a.budgets.split(",")]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    rows = rows[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
    if a.nshards > 1:
        a.out = a.out.replace(".jsonl", f".shard{a.shard}.jsonl")
    model, processor = load_qwen25vl(device=a.device, max_pixels=MAX_PIXELS)
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_id = f"m3pilot-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    with open(out_path, "w") as f:
        f.write(json.dumps({
            "record_type": "run_metadata", "schema_version": "1.1",
            "run_id": run_id, "stage": "M3", "run_kind": "smoke",
            "note": "s1 cross-eval pilot, NO T0-T4 labels — aggregate signal only",
            "budgets_keep": budgets,
            "started_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False) + "\n")
        for di, row in enumerate(rows):
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs = row["questions"][1:1 + a.questions_per_doc]  # track1 평가 질문과 동일
            if len(qs) < 2:
                continue
            # 질문별 s1 점수와 입력 준비
            ins_list, s1_list = [], []
            for q in qs:
                ins = S.vlm_inputs(processor, img, q["question"] + BRIEF, a.device)
                ins_list.append(ins)
                s1_list.append(
                    S.score_s1(model, processor, img,
                               q["question"] + BRIEF, a.device).cpu())
            n_vis = s1_list[0].shape[0]
            shape = KVShape(layers=N_LAYERS, batch=1, kv_heads=N_KV_HEADS,
                            tokens=n_vis, head_dim=HEAD_DIM)
            e = dense_storage(shape)
            full_bytes = e.payload_bytes + e.metadata_bytes + e.position_bytes

            for B in budgets:
                k = max_keep_for_budget(shape, int(B * full_bytes), "sparse")
                keeps = [set(torch.topk(s, min(k, n_vis)).indices.tolist())
                         for s in s1_list]
                union = set().union(*keeps)
                subsets = {f"S_q{i}": ks for i, ks in enumerate(keeps)}
                subsets["UNION"] = union
                # 겹침 기록 (판단 근거가 아니라 관찰값)
                jac = {}
                for i in range(len(keeps)):
                    for j in range(i + 1, len(keeps)):
                        inter = len(keeps[i] & keeps[j])
                        jac[f"q{i}q{j}"] = round(
                            inter / max(len(keeps[i] | keeps[j]), 1), 3)
                for src, keep in subsets.items():
                    for j, q in enumerate(qs):
                        ins = ins_list[j]
                        sp = token_spans(ins["input_ids"], model.config)
                        vis, vis_end = sp["visual"], sp["vis_end"]
                        evict = torch.tensor(
                            [int(vis[o]) for o in range(n_vis) if o not in keep],
                            device=a.device)
                        pred = greedy_generate_masked(
                            model, processor, ins,
                            max_new_tokens=a.max_new_tokens,
                            evict_cols=evict, row_start=vis_end + 1)
                        f.write(json.dumps({
                            "run_id": run_id, "sample_id": row["sample_id"],
                            "subset_from": src, "eval_q_idx": j,
                            "eval_question_id": q["question_id"],
                            "is_self": src == f"S_q{j}",
                            "budget_per_question": B,
                            "keep_tokens": len(keep),
                            "keep_ratio_actual": round(len(keep) / n_vis, 4),
                            "jaccard": jac if src == "UNION" else None,
                            "gold": q["answers"],
                            "prediction": pred,
                            "anls": anls(pred, q["answers"]),
                            "em": exact_match(pred, q["answers"])},
                            ensure_ascii=False) + "\n")
                        f.flush()
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
