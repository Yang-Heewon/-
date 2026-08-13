"""게이트 G-B: oracle 마스킹 sanity check (PLAN.md §4).

합성 이미지(암구호가 이미지에만 존재)로 3가지를 확인한다:
  1. 모델이 이미지를 읽어 정답을 맞히는가 (생성 확인)
  2. 4D causal 마스크가 2D 기준선과 같은 logp를 주는가 (마스크 파이프라인 정합성)
  3. 시각 KV 차단 시 정답 logp가 폭락하는가 (마스크가 실제로 소비되는가)
     - V1: 프롬프트 프리필 후 축출 (질문 토큰은 시각 KV를 이미 봄)
     - V2: 질문 도착 전 축출 (D4 재사용 semantics) ← 이게 폭락해야 PASS

V2가 폭락하지 않으면 마스크가 무시되고 있는 것 → D3/D4 전부 무효 (DIAGNOSIS_DESIGN §3.2).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from PIL import Image, ImageDraw, ImageFont

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import (
    causal_mask_4d, evict_columns, mrope_position_ids, answer_logp)

DEVICE = "cuda:0"
SECRET = "4729"


def make_image():
    img = Image.new("RGB", (672, 672), "white")
    d = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 160)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
    except OSError:
        big = small = ImageFont.load_default()
    d.text((336, 200), "PASSCODE", fill="black", font=small, anchor="mm")
    d.text((336, 360), SECRET, fill="black", font=big, anchor="mm")
    return img


def main():
    model, processor = load_qwen25vl(device=DEVICE)
    img = make_image()
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What is the passcode shown in the image? Answer with the digits only."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    # ── 1. 생성 확인: 모델이 이미지에서 암구호를 읽는가
    gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    gen_text = processor.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[1] greedy 생성: {gen_text!r}  (기대: {SECRET})")

    # ── 정답을 이어붙인 전체 시퀀스 구성
    ans_ids = processor.tokenizer(SECRET, add_special_tokens=False,
                                  return_tensors="pt").input_ids.to(DEVICE)
    P = inputs["input_ids"].shape[1]
    full_ids = torch.cat([inputs["input_ids"], ans_ids], dim=1)
    L = full_ids.shape[1]
    attn2d = torch.ones(1, L, dtype=torch.long, device=DEVICE)
    spans = token_spans(full_ids, model.config)
    vis, vis_end = spans["visual"], spans["vis_end"]
    print(f"    프롬프트 {P}토큰 (시각 {len(vis)}), 정답 {L-P}토큰")

    pos = mrope_position_ids(model, full_ids, inputs["image_grid_thw"], attn2d)
    kw = dict(input_ids=full_ids, pixel_values=inputs["pixel_values"],
              image_grid_thw=inputs["image_grid_thw"], answer_start=P)

    # ── 2. 기준선: 2D vs 4D-causal (파이프라인 정합성)
    lp_2d, _ = answer_logp(model, attention_mask=attn2d, **kw)
    m4 = causal_mask_4d(L, DEVICE)
    lp_4d, _ = answer_logp(model, attention_mask=m4, position_ids=pos, **kw)
    print(f"[2] 정답 logp — 2D 기준선: {lp_2d:.3f} | 4D causal: {lp_4d:.3f} "
          f"(|Δ|={abs(lp_2d-lp_4d):.3f}, 작아야 정상)")

    # ── 3. 시각 KV 축출
    lp_v1, _ = answer_logp(model, attention_mask=evict_columns(m4, vis, row_start=P),
                           position_ids=pos, **kw)
    lp_v2, _ = answer_logp(model, attention_mask=evict_columns(m4, vis, row_start=vis_end + 1),
                           position_ids=pos, **kw)
    print(f"[3] 시각 KV 전체 차단 — V1(디코딩 시점): {lp_v1:.3f} | V2(질문 전, D4 semantics): {lp_v2:.3f}")

    # ── 4. 무이미지 prior 바닥: 이미지가 주는 정보 이득의 기준점
    msg_t = [{"role": "user", "content": [{"type": "text",
              "text": "What is the passcode shown in the image? Answer with the digits only."}]}]
    text_t = processor.apply_chat_template(msg_t, tokenize=False, add_generation_prompt=True)
    ids_t = processor.tokenizer(text_t, return_tensors="pt").input_ids.to(DEVICE)
    full_t = torch.cat([ids_t, ans_ids], dim=1)
    out_t = model(input_ids=full_t, attention_mask=torch.ones_like(full_t), use_cache=False)
    lg = out_t.logits.float()
    labels = full_t[0, ids_t.shape[1]:]
    lp_prior = torch.log_softmax(lg[0, ids_t.shape[1]-1:-1], -1)[
        torch.arange(len(labels)), labels].sum().item()
    print(f"[4] 무이미지 prior: {lp_prior:.3f}  (4자리 균등 추측 이론값 ≈ -9.21)")

    # ── 판정: 축출이 이미지의 정보 이득을 대부분 파괴하는가
    ok_pipe = abs(lp_2d - lp_4d) < 1.0
    info_gain = lp_2d - lp_prior
    destroyed = (lp_2d - lp_v2) / info_gain if info_gain > 0 else float("nan")
    ok_gain = info_gain >= 2.0
    ok_evict = destroyed >= 0.7
    print(f"\n판정: 파이프라인 정합 {'PASS' if ok_pipe else 'FAIL'} (|Δ|={abs(lp_2d-lp_4d):.3f}) | "
          f"정보 이득 {info_gain:.1f} nats {'PASS' if ok_gain else 'FAIL(이미지 판독 불충분)'} | "
          f"V2 파괴율 {destroyed:.0%} {'PASS' if ok_evict else 'FAIL'}")
    print("게이트 G-B:", "PASS ✅" if (ok_pipe and ok_gain and ok_evict)
          else "FAIL ❌ — 위 항목 점검")


if __name__ == "__main__":
    main()
