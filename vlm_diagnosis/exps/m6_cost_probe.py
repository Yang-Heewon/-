"""M6 비용 측정 — 하이브리드 정책의 두 경로가 실제로 얼마나 걸리는가.

경로 A (기억 재사용): 저장해둔 시각 KV 부분집합 위에 질문 토큰만 prefill → decode
경로 B (이미지 재읽기): vision 인코더 + 이미지 전체 prefill → decode

출력이 정확할 필요는 없고(정확도는 기존 실험이 담당) 계산 모양이 실전과 같아야 한다.
KV는 실제 full prefill에서 잘라내 사용한다. 저장 바이트(이미지 파일 vs KV)도 기록.

  python -m vlm_diagnosis.exps.m6_cost_probe \
    --manifest experiments/manifests/gqa_transfer.jsonl --device cuda:2 --limit 10
"""
import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers.cache_utils import DynamicCache

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.kv_baselines import KVShape, dense_storage, sparse_storage
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."
N_LAYERS, N_KV_HEADS, HEAD_DIM = 28, 4, 128


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def decode_steps(model, last_logits, past, pos_start, device, steps=16):
    """greedy로 steps 토큰 decode (cache 사용) — 시간 측정용."""
    tok = int(last_logits[0, -1].argmax())
    t0 = sync_time()
    for i in range(steps):
        ids = torch.tensor([[tok]], device=device)
        pos = torch.full((3, 1, 1), pos_start + i, device=device, dtype=torch.long)
        out = model(input_ids=ids, past_key_values=past,
                    position_ids=pos, use_cache=True)
        tok = int(out.logits[0, -1].argmax())
    return sync_time() - t0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--keep-ratios", default="0.05,0.2")
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/smoke/m6_cost.jsonl")
    a = ap.parse_args()

    keeps = [float(x) for x in a.keep_ratios.split(",")]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))][:a.limit]
    model, processor = load_qwen25vl(device=a.device, max_pixels=MAX_PIXELS)
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "a") as f:
        for di, row in enumerate(rows):
            img_path = os.path.join(ROOT, row["image"])
            img = Image.open(img_path).convert("RGB")
            q = row["questions"][1]["question"] + BRIEF
            ins = S.vlm_inputs(processor, img, q, a.device)
            sp = token_spans(ins["input_ids"], model.config)
            vis, L = sp["visual"], sp["L"]
            n_vis = len(vis)
            attn = torch.ones(1, L, dtype=torch.long, device=a.device)
            pos = mrope_position_ids(model, ins["input_ids"],
                                     ins["image_grid_thw"], attn)
            rec = {"sample_id": row["sample_id"], "n_visual": n_vis, "seq_len": L,
                   "image_bytes": os.path.getsize(img_path)}
            shape = KVShape(layers=N_LAYERS, batch=1, kv_heads=N_KV_HEADS,
                            tokens=n_vis, head_dim=HEAD_DIM)
            e = dense_storage(shape)
            rec["kv_full_bytes"] = e.payload_bytes + e.metadata_bytes + e.position_bytes

            for rep in range(a.repeats):   # 마지막 반복만 기록 (워밍업 겸용)
                # 경로 B: vision 인코더 단독
                t0 = sync_time()
                model.model.visual(ins["pixel_values"],
                                   grid_thw=ins["image_grid_thw"])
                t_vision = sync_time() - t0
                # 경로 B: 이미지+질문 전체 prefill (+cache)
                t0 = sync_time()
                out = model(input_ids=ins["input_ids"], attention_mask=attn,
                            position_ids=pos, pixel_values=ins["pixel_values"],
                            image_grid_thw=ins["image_grid_thw"], use_cache=True)
                t_prefill_full = sync_time() - t0
                full_past = out.past_key_values
                t_dec_b = decode_steps(model, out.logits, full_past,
                                       int(pos.max()) + 1, a.device,
                                       a.decode_steps)
            rec.update(t_vision=round(t_vision, 4),
                       t_prefill_full=round(t_prefill_full, 4),
                       t_decode_B=round(t_dec_b, 4))

            # 질문 토큰 구간 (시각 span 뒤 텍스트) — 기억 경로에서 새로 계산할 부분
            q_start = sp["vis_end"] + 1
            q_ids = ins["input_ids"][:, q_start:]
            for K in keeps:
                k = max(1, int(n_vis * K))
                keep_vis = [int(vis[i]) for i in
                            torch.linspace(0, n_vis - 1, k).round().long().tolist()]
                keep_idx = sorted(set(range(int(vis[0]))) | set(keep_vis))
                idx = torch.tensor(keep_idx, device=a.device)
                ek = sparse_storage(KVShape(layers=N_LAYERS, batch=1,
                                            kv_heads=N_KV_HEADS, tokens=n_vis,
                                            head_dim=HEAD_DIM), keep_tokens=k)
                for rep in range(a.repeats):
                    # 저장된 KV 재현: full prefill cache에서 [프리픽스+kept 시각] 슬라이스
                    legacy = tuple(
                        (kk.index_select(2, idx), vv.index_select(2, idx))
                        for kk, vv in full_past.to_legacy_cache())
                    past = DynamicCache.from_legacy_cache(legacy)
                    # 경로 A: 질문 prefill (저장 KV 위, 위치는 원래 위치 유지)
                    t0 = sync_time()
                    qpos = pos[:, :, q_start:]
                    la = len(keep_idx) + q_ids.shape[1]
                    out_a = model(input_ids=q_ids, past_key_values=past,
                                  position_ids=qpos,
                                  attention_mask=torch.ones(1, la, dtype=torch.long,
                                                            device=a.device),
                                  use_cache=True)
                    t_prefill_mem = sync_time() - t0
                    t_dec_a = decode_steps(model, out_a.logits, out_a.past_key_values,
                                           int(pos.max()) + 1, a.device,
                                           a.decode_steps)
                rec[f"keep{int(K*100)}"] = {
                    "keep_tokens": k,
                    "kv_sparse_bytes": ek.payload_bytes + ek.metadata_bytes + ek.position_bytes,
                    "t_prefill_mem": round(t_prefill_mem, 4),
                    "t_decode_A": round(t_dec_a, 4)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} n_vis={n_vis} "
                  f"vision {rec['t_vision']:.2f}s prefill {rec['t_prefill_full']:.2f}s",
                  flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
