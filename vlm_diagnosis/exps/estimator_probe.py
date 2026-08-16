"""2a — 배포형 estimator 신호 수집 (docs/EXP-2A-PROTOCOL.md 단계 1).

각 (이미지, 부분집합 S, 평가 질문 q)에 대해 실전 조건(남긴 토큰만 보이는
4D 마스크)으로 q를 forward 1회 하고, 그 부산물에서 배포 가능한 신호를 뽑는다.

신호 (입력은 '보관 조각 + 질문'뿐 — 전체-KV 정보 사용 금지):
  a1_mass     질문 행들이 보관 시각 토큰에 준 attention 비중 (softmax는 보이는
              열 위에서만 — 실전과 동일). 질문 행 평균.
  a2_entropy  보관 시각 토큰 위 분포의 엔트로피 ÷ log(보관 수) (0~1)
  a3_sink     질문 attention 중 앞쪽 sink 토큰(4개) 비중
  a4_margin   첫 답 토큰의 top1-top2 logit 차 (사후 검증형)
후보 5용: 같은 부분집합에서 에피소드 질문(q0)의 a1~a4 = ref_* (자기 보정 기준점).

부분집합 재구성은 coverage_probe와 동일 (m3 러너와 결정적 일치 검증됨).

  python -m vlm_diagnosis.exps.estimator_probe \
    --manifest experiments/manifests/gqa_transfer.jsonl --device cuda:2 --limit 2
"""
import argparse
import json
import math
import os
import time
import zlib

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import mrope_position_ids, causal_mask_4d, evict_columns
from vlm_diagnosis.core.attnstat import QKCapture
from vlm_diagnosis.core.kv_baselines import KVShape, dense_storage, max_keep_for_budget
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."
N_SINK = 4


@torch.no_grad()
def masked_question_signals(model, processor, ins, keep_vis_pos, evict_cols_t,
                            q_start, device):
    """실전 조건 forward 1회 → (a1,a2,a3,a4). keep_vis_pos: 보관 시각 토큰의
    시퀀스 위치 목록. q_start: 질문 시작 행 (= vis_end+1)."""
    ids = ins["input_ids"]; L = ids.shape[1]
    attn2d = torch.ones(1, L, dtype=torch.long, device=device)
    pos = mrope_position_ids(model, ids, ins["image_grid_thw"], attn2d)
    m4 = causal_mask_4d(L, device, torch.float16)
    if evict_cols_t.numel():
        m4 = evict_columns(m4, evict_cols_t, row_start=q_start)
    with QKCapture() as cap:
        out = model(input_ids=ids, attention_mask=m4, position_ids=pos,
                    pixel_values=ins["pixel_values"],
                    image_grid_thw=ins["image_grid_thw"], use_cache=False)
    logits = out.logits[0, -1].float()
    top2 = torch.topk(logits, 2).values
    a4 = float(top2[0] - top2[1])

    keep_t = torch.tensor(sorted(keep_vis_pos), device=device)
    evset = set(evict_cols_t.tolist())
    cols = torch.arange(L, device=device)
    # 질문 행들이 보는 마스크: causal + 삭제 열 제외 (실전의 softmax 분모와 동일)
    a1 = a2 = a3 = 0.0
    n_layers = 0
    for q, k in cap.qk:
        q = q[0].float(); k = k[0].float()
        H = q.shape[0]
        k = k.repeat_interleave(H // k.shape[0], dim=0)
        qr = q[:, q_start:L]                              # (H, R, d) 질문 행만
        w = qr @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])   # (H, R, L)
        rows = torch.arange(q_start, L, device=device)
        w.masked_fill_(cols[None, None, :] > rows[None, :, None], float("-inf"))
        if evict_cols_t.numel():
            w[:, :, evict_cols_t] = float("-inf")
        p = w.softmax(-1)                                  # 보이는 열 위 분포
        pk = p[:, :, keep_t]                               # 보관 시각 토큰 몫
        a1 += float(pk.sum(-1).mean())
        pn = pk / pk.sum(-1, keepdim=True).clamp(min=1e-9)
        ent = -(pn * pn.clamp(min=1e-9).log()).sum(-1).mean()
        a2 += float(ent) / max(math.log(len(keep_vis_pos)), 1e-9)
        a3 += float(p[:, :, :N_SINK].sum(-1).mean())
        n_layers += 1
        del w, p, pk, pn
    return dict(a1_mass=round(a1 / n_layers, 6),
                a2_entropy=round(a2 / n_layers, 6),
                a3_sink=round(a3 / n_layers, 6),
                a4_margin=round(a4, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budgets", default="0.05,0.2")
    ap.add_argument("--seed", type=int, default=42)      # m3와 동일해야 함
    ap.add_argument("--out", default="results/smoke/estimator_signals.jsonl")
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
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    NL, NKV, HD = kv_dims(model)

    with open(out_path, "a" if a.resume else "w") as f:
        for di, row in enumerate(rows):
            if str(row["sample_id"]) in done:
                continue
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs_src = row["questions"][1:4]
            qs_ho = row["questions"][4:6]
            q0 = row["questions"][0]                      # 에피소드 (자기 보정 기준)
            if len(qs_src) < 2:
                continue
            s1_src = [S.score_s1(model, processor, img, q["question"] + BRIEF,
                                 a.device).cpu() for q in qs_src]
            s5_doc = S.score_s5(model, processor, img, a.device).cpu()
            n_vis = s1_src[0].shape[0]
            shape = KVShape(layers=NL, batch=1, kv_heads=NKV, tokens=n_vis,
                            head_dim=HD)
            e = dense_storage(shape)
            full_bytes = e.payload_bytes + e.metadata_bytes + e.position_bytes
            gen = torch.Generator().manual_seed(
                zlib.crc32(f"{a.seed}:{row['sample_id']}".encode()) & 0x7FFFFFFF)

            for B in budgets:                             # 순서 = m3 (generator 상태)
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
                evals = ([("cross", j, q) for j, q in enumerate(qs_src)]
                         + [("heldout", j, q) for j, q in enumerate(qs_ho)]
                         + [("episode_ref", 0, q0)])      # 자기 보정 기준점
                for src, keep in subsets.items():
                    ins0 = S.vlm_inputs(processor, img,
                                        qs_src[0]["question"] + BRIEF, a.device)
                    sp0 = token_spans(ins0["input_ids"], model.config)
                    vis = sp0["visual"]
                    keep_pos = [int(vis[o]) for o in sorted(keep)]
                    for mode, j, q in evals:
                        ins = S.vlm_inputs(processor, img,
                                           q["question"] + BRIEF, a.device)
                        sp = token_spans(ins["input_ids"], model.config)
                        ev = torch.tensor(
                            [int(sp["visual"][o]) for o in range(n_vis)
                             if o not in keep], device=a.device)
                        sig = masked_question_signals(
                            model, processor, ins, keep_pos, ev,
                            sp["vis_end"] + 1, a.device)
                        f.write(json.dumps({
                            "sample_id": row["sample_id"], "eval_mode": mode,
                            "subset_from": src, "eval_q_idx": j,
                            "eval_question_id": q["question_id"],
                            "budget_per_question": B,
                            "keep_tokens": len(keep), "n_visual": n_vis,
                            **sig}, ensure_ascii=False) + "\n")
                f.flush()
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} n_vis={n_vis} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
