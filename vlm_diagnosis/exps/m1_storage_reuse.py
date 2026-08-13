"""M1 — 저장·재사용 경계 runner (01_M1_STORAGE_REUSE.md §8, 부분 구현).

이 버전이 커버하는 축:
  M1-A  : STORED-FULL(K_p+K_v, 동일 offset) vs IMAGE-REENCODE — serialization/
          load/injection **gate** (연구 결과 아님; DECISIONS §3 규칙)
  M1-B' : write 시 system 문맥 변형 → 동일 read (context conditioning 1차 프로브)
  M1-C' : naive offset shift (+128/512/2048, 재회전 없음 — position 결박 프로브)
  M1-D  : K_v 단독 (prefix 행 제거) vs K_p+K_v

아직 없는 것: mRoPE 재회전 주입, canonical 10 payload 전체(M1-F, T_visual 필요),
block 합성(M1-E), layer별 최초 divergence tracing.

프롬프트 경계 (M1-01 권장 시작점): K_p+K_v = 토큰 [0..vis_end] — 질문과 무관한
prefix이므로 write 시 질문을 몰라도 동일 (M1-02 generic write와 정합).

실행 (smoke):
  python -m vlm_diagnosis.exps.m1_storage_reuse \
    --manifest experiments/manifests/m2a_diagnostic.jsonl --limit 4 --device cuda:1
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch
from PIL import Image
from transformers import DynamicCache

from vlm_diagnosis.core.loader import load_qwen25vl, assert_finite_logits
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.metrics import exact_match, contains_match

BRIEF = " Answer with a single word or phrase."
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
WRITE_SYSTEM_VARIANT = (
    "You are an archival assistant. Carefully memorize everything in the "
    "image for future reference.")


def _mem_path(out_dir, sample_id, tag):
    return os.path.join(out_dir, f"mem_{sample_id}_{tag}.pt")


@torch.no_grad()
def write_memory(model, processor, img, device, out_path, system_text=None):
    """generic write: [system?, image] prefix forward → KV+Z 직렬화.

    system_text가 있으면 write 문맥의 system 메시지를 바꾼다 (M1-B' 프로브).
    이 경우 visual 행의 절대 position도 함께 이동하므로, CTX_SHIFT 조건은
    '문맥 텍스트+position 이동'의 결합 프로브다 (분리는 후속 M1-B/C 본실험).
    """
    msgs = []
    if system_text is not None:
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "x"}]})
    text = processor.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    ins = processor(text=[text], images=[img], return_tensors="pt").to(device)
    full_ids = ins["input_ids"]
    sp = token_spans(full_ids, model.config)
    cut = sp["vis_end"] + 1                      # K_p+K_v 경계 (M1-01 권장)
    prefix = full_ids[:, :cut]
    attn = torch.ones(1, cut, dtype=torch.long, device=device)
    pos = mrope_position_ids(model, prefix, ins["image_grid_thw"], attn)
    out = model(input_ids=prefix, attention_mask=attn, position_ids=pos,
                pixel_values=ins["pixel_values"],
                image_grid_thw=ins["image_grid_thw"], use_cache=True)
    assert_finite_logits(out.logits, "write_memory")
    legacy = out.past_key_values.to_legacy_cache()
    torch.save({"kv": [(k.cpu(), v.cpu()) for k, v in legacy],
                "position_ids": pos.cpu(), "prefix_len": cut,
                "vis_start": int(sp["visual"].min()),
                "vis_end": sp["vis_end"],
                "grid_thw": ins["image_grid_thw"].cpu(),
                "dtype": "float16", "boundary": "system+image (M1-01 proposed)",
                "system_variant": system_text is not None}, out_path)
    return cut


def _load_cache(path, device, kv_only=False, vis_start=None):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    kv = blob["kv"]
    if kv_only:
        s = blob["vis_start"]
        kv = [(k[:, :, s:], v[:, :, s:]) for k, v in kv]
    cache = DynamicCache.from_legacy_cache(
        tuple((k.to(device), v.to(device)) for k, v in kv))
    return cache, blob


@torch.no_grad()
def read_with_memory(model, processor, img_question_ins, mem_path, device,
                     offset=0, kv_only=False, max_new_tokens=24):
    """저장 KV 주입 + 질문 suffix 실행 (이미지 픽셀 없음). greedy 답 반환."""
    cache, blob = _load_cache(mem_path, device, kv_only=kv_only)
    full_ids = img_question_ins["input_ids"]
    # suffix 경계는 read-side 템플릿 기준 (write 문맥 길이와 다를 수 있음 — M1-B')
    P = token_spans(full_ids, model.config)["vis_end"] + 1
    suffix = full_ids[:, P:]
    L = full_ids.shape[1]
    attn2d = torch.ones(1, L, dtype=torch.long, device=device)
    pos_full = mrope_position_ids(model, full_ids,
                                  img_question_ins["image_grid_thw"], attn2d)
    pos_suffix = pos_full[:, :, P:] + offset
    # 2D mask 길이는 항상 cache 길이+suffix (kv_only·ctx 변형 시 P와 다름)
    n_cached = cache.get_seq_length()
    attn = torch.ones(1, n_cached + suffix.shape[1],
                      dtype=torch.long, device=device)
    out = model(input_ids=suffix, attention_mask=attn,
                position_ids=pos_suffix, past_key_values=cache, use_cache=True)
    logits_first = out.logits
    next_id = out.logits[0, -1].argmax()
    next_pos = int(pos_suffix.max()) + 1
    generated = [int(next_id)]
    eos = {model.config.eos_token_id} if isinstance(
        model.config.eos_token_id, int) else set(model.config.eos_token_id or [])
    past = out.past_key_values
    for _ in range(max_new_tokens - 1):
        if int(next_id) in eos:
            break
        attn_step = torch.ones(1, past.get_seq_length() + 1,
                               dtype=torch.long, device=device)
        p = torch.full((3, 1, 1), next_pos, device=device,
                       dtype=pos_suffix.dtype)
        o = model(input_ids=next_id.view(1, 1), attention_mask=attn_step,
                  position_ids=p, past_key_values=past, use_cache=True)
        past = o.past_key_values
        next_id = o.logits[0, -1].argmax()
        next_pos += 1
        generated.append(int(next_id))
    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, logits_first


@torch.no_grad()
def image_reencode(model, processor, ins, device, max_new_tokens=24):
    out = model.generate(**ins, max_new_tokens=max_new_tokens, do_sample=False)
    P = ins["input_ids"].shape[1]
    return processor.tokenizer.decode(out[0, P:], skip_special_tokens=True).strip()


@torch.no_grad()
def no_mem(model, processor, question, device, max_new_tokens=24):
    msgs = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    text = processor.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    ids = processor.tokenizer(text, return_tensors="pt").input_ids.to(device)
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                         do_sample=False)
    return processor.tokenizer.decode(out[0, ids.shape[1]:],
                                      skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/m2a_diagnostic.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--questions-per-doc", type=int, default=2)
    ap.add_argument("--offsets", default="128,512,2048")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", default="results/smoke/m1_storage_reuse.jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    if a.limit:
        rows = rows[:a.limit]
    offsets = [int(x) for x in a.offsets.split(",") if x]
    model, processor = load_qwen25vl(device=a.device, max_pixels=MAX_PIXELS)
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mem_dir = os.path.join(os.path.dirname(out_path), "m1_mem")
    os.makedirs(mem_dir, exist_ok=True)
    run_id = f"m1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    with open(out_path, "w") as f:
        f.write(json.dumps({
            "record_type": "run_metadata", "schema_version": "1.1",
            "run_id": run_id, "stage": "M1", "run_kind": "smoke",
            "manifest_path": a.manifest, "offsets": offsets,
            "boundary": "system+image (M1-01 proposed)",
            "note": "M1-A gate + naive-offset/K_v-only probes; no re-rotation yet",
            "started_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False) + "\n")
        for di, row in enumerate(rows):
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            t0 = time.time()
            mem = _mem_path(mem_dir, row["sample_id"], "std")
            write_memory(model, processor, img, a.device, mem)
            mem_ctx = _mem_path(mem_dir, row["sample_id"], "ctx")
            write_memory(model, processor, img, a.device, mem_ctx,
                         system_text=WRITE_SYSTEM_VARIANT)
            for q in row["questions"][:a.questions_per_doc]:
                q_text = q["question"] + BRIEF
                ins = S.vlm_inputs(processor, img, q_text, a.device)
                acc = q["answers"]
                conds = {}
                conds["NO_MEM"] = no_mem(model, processor, q_text,
                                         a.device, a.max_new_tokens)
                conds["IMAGE"] = image_reencode(model, processor, ins,
                                                a.device, a.max_new_tokens)
                conds["STORED_FULL"], _ = read_with_memory(
                    model, processor, ins, mem, a.device,
                    max_new_tokens=a.max_new_tokens)
                conds["STORED_KV_ONLY"], _ = read_with_memory(
                    model, processor, ins, mem, a.device, kv_only=True,
                    max_new_tokens=a.max_new_tokens)
                conds["STORED_FULL_CTX_SHIFT"], _ = read_with_memory(
                    model, processor, ins, mem_ctx, a.device,
                    max_new_tokens=a.max_new_tokens)
                for off in offsets:
                    conds[f"STORED_FULL_OFFSET_{off}"], _ = read_with_memory(
                        model, processor, ins, mem, a.device, offset=off,
                        max_new_tokens=a.max_new_tokens)
                rec = {"run_id": run_id, "dataset": row["dataset"],
                       "split": "smoke", "sample_id": row["sample_id"],
                       "question_id": q["question_id"],
                       "question": q["question"], "gold": acc,
                       "predictions": conds,
                       "scores": {k: exact_match(v, acc)
                                  for k, v in conds.items()},
                       "contains_scores": {k: contains_match(v, acc)
                                           for k, v in conds.items()}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            os.remove(mem); os.remove(mem_ctx)
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
