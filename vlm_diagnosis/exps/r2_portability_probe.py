"""R2 이식성 + 실제 직렬화 검증 (외부 검토 반영 — gpt.md '복원할 것' 항목).

질문: 남긴 KV 조각을 **실제로 디스크에 직렬화**했다가 **새 세션에서 복원**해도,
그리고 **다른 문맥 위치(offset)**에 놓아도 답이 유지되는가?

각 화면에 대해 4조건으로 held-out 질문에 답하게 한다:
  A in-session-mask : 기존 실험 경로 (4D 마스크 시뮬레이션) — 기준
  B reload          : full prefill→UNION@5% 열 추출→torch.save→**프로세스 상태와
                      무관하게 재로드**→질문 prefill(과거 위치 유지)→생성
  C reload+offset   : B와 같되 질문의 위치 번호를 +512 밀어서 (세션이 길어진 상황)
  D reload+prompt   : B와 같되 질문 앞에 다른 시스템 문장 삽입 (문맥 교체 상황)

측정: 각 조건의 EM + A와의 답 일치율(loyalty). B≈A면 "마스크 시뮬레이션 = 실제
직렬화" 등가가 실증되고, C·D가 버티면 R2(위치·문맥 이식성) gate 통과.

  python -m vlm_diagnosis.exps.r2_portability_probe \
    --manifest experiments/manifests/screenqa_discovery.jsonl --device cuda:2 --limit 40
"""
import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers.cache_utils import DynamicCache

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.kv_baselines import KVShape, dense_storage, max_keep_for_budget
from vlm_diagnosis.core.metrics import exact_match, normalize_text
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."


@torch.no_grad()
def gen_from_loaded(model, processor, loaded_path, keep_pos, q_ids, q_pos,
                    device, max_new_tokens=16):
    """디스크에서 복원한 KV 위에 질문 prefill → greedy 생성 (실전 경로)."""
    legacy = torch.load(loaded_path, map_location="cpu", weights_only=True)
    past = DynamicCache.from_legacy_cache(
        tuple((k.to(device).half(), v.to(device).half()) for k, v in legacy))
    L_kv = past.get_seq_length()
    tok = processor.tokenizer
    ids = q_ids
    out = model(input_ids=ids, past_key_values=past, position_ids=q_pos,
                attention_mask=torch.ones(1, L_kv + ids.shape[1],
                                          dtype=torch.long, device=device),
                use_cache=True)
    toks = []
    cur = int(out.logits[0, -1].argmax())
    pos_next = int(q_pos.max()) + 1
    for i in range(max_new_tokens):
        if cur in (tok.eos_token_id,):
            break
        toks.append(cur)
        step = model(input_ids=torch.tensor([[cur]], device=device),
                     past_key_values=out.past_key_values,
                     position_ids=torch.full((3, 1, 1), pos_next + i,
                                             device=device, dtype=torch.long),
                     use_cache=True)
        out = step
        cur = int(step.logits[0, -1].argmax())
    return tok.decode(toks, skip_special_tokens=True).strip()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--offset", type=int, default=512)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="results/discovery/r2_portability.jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))][:a.limit]
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = "/tmp/r2_kv.pt"
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    NL, NKV, HD = kv_dims(model)
    tok = processor.tokenizer

    with open(out_path, "w") as f:
        for di, row in enumerate(rows):
            t0 = time.time()
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            qs_src = row["questions"][1:4]
            qs_eval = row["questions"][4:6]
            if len(qs_src) < 2 or not qs_eval:
                continue
            s1 = [S.score_s1(model, processor, img, q["question"] + BRIEF,
                             a.device).cpu() for q in qs_src]
            n_vis = s1[0].shape[0]
            shape = KVShape(layers=NL, batch=1, kv_heads=NKV, tokens=n_vis,
                            head_dim=HD)
            e = dense_storage(shape)
            fb = e.payload_bytes + e.metadata_bytes + e.position_bytes
            k = max_keep_for_budget(shape, int(a.budget * fb), "sparse")
            union = sorted(set().union(*[set(torch.topk(s, min(k, n_vis))
                                             .indices.tolist()) for s in s1]))
            # write 세션: full prefill → [프리픽스 + 보관 시각열] 추출 → 디스크
            ins0 = S.vlm_inputs(processor, img, qs_eval[0]["question"] + BRIEF,
                                a.device)
            sp0 = token_spans(ins0["input_ids"], model.config)
            vis = sp0["visual"]
            attn = torch.ones_like(ins0["input_ids"])
            pos0 = mrope_position_ids(model, ins0["input_ids"],
                                      ins0["image_grid_thw"], attn)
            pre = model(input_ids=ins0["input_ids"][:, :sp0["vis_end"] + 1],
                        attention_mask=attn[:, :sp0["vis_end"] + 1],
                        position_ids=pos0[:, :, :sp0["vis_end"] + 1],
                        pixel_values=ins0["pixel_values"],
                        image_grid_thw=ins0["image_grid_thw"], use_cache=True)
            keep_seq = sorted(set(range(int(vis[0])))
                              | {int(vis[o]) for o in union})
            idx = torch.tensor(keep_seq, device=a.device)
            legacy = tuple((kk.index_select(2, idx).cpu(),
                            vv.index_select(2, idx).cpu())
                           for kk, vv in pre.past_key_values.to_legacy_cache())
            torch.save(legacy, tmp)
            nbytes = os.path.getsize(tmp)
            del pre
            torch.cuda.empty_cache()

            for q in qs_eval:
                golds = q["answers"]
                ins = S.vlm_inputs(processor, img, q["question"] + BRIEF, a.device)
                sp = token_spans(ins["input_ids"], model.config)
                # A: 기존 마스크 경로
                ev = torch.tensor([int(sp["visual"][o]) for o in range(n_vis)
                                   if o not in set(union)], device=a.device)
                ansA = greedy_generate_masked(model, processor, ins,
                                              max_new_tokens=16, evict_cols=ev,
                                              row_start=sp["vis_end"] + 1)
                # 질문 토큰 구간과 위치 (write 때와 같은 프롬프트 형식)
                q_ids = ins["input_ids"][:, sp["vis_end"] + 1:]
                attn2 = torch.ones_like(ins["input_ids"])
                posq = mrope_position_ids(model, ins["input_ids"],
                                          ins["image_grid_thw"], attn2)
                qp = posq[:, :, sp["vis_end"] + 1:]
                ansB = gen_from_loaded(model, processor, tmp, keep_seq,
                                       q_ids, qp, a.device)
                ansC = gen_from_loaded(model, processor, tmp, keep_seq,
                                       q_ids, qp + a.offset, a.device)
                # D: 다른 프롬프트 문맥 — 질문 앞에 시스템 문장 추가
                pre_ids = tok(" You are answering from stored memory.",
                              add_special_tokens=False,
                              return_tensors="pt").input_ids.to(a.device)
                q_idsD = torch.cat([pre_ids, q_ids], 1)
                start = int(qp[:, :, 0].max())
                qpD = torch.arange(start, start + q_idsD.shape[1],
                                   device=a.device)
                qpD = qpD[None, None, :].expand(3, 1, -1)
                ansD = gen_from_loaded(model, processor, tmp, keep_seq,
                                       q_idsD, qpD, a.device)
                rec = {"sample_id": row["sample_id"],
                       "question_id": q["question_id"], "gold": golds,
                       "kv_file_bytes": nbytes, "keep_tokens": len(keep_seq)}
                for name, ans in (("A_mask", ansA), ("B_reload", ansB),
                                  ("C_offset", ansC), ("D_prompt", ansD)):
                    rec[f"ans_{name}"] = ans
                    rec[f"em_{name}"] = exact_match(ans, golds)
                    rec[f"agree_{name}"] = float(
                        normalize_text(ans) == normalize_text(ansA))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} {time.time()-t0:.0f}s",
                  flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
