"""NO-MEM 기준선 — 이미지 없이 질문 텍스트만 주고 정답률을 잰다 (언어 편향 점검).

random 부분집합의 유지율이 높게 나온 것이 "장면 중복성" 때문인지, 아니면
애초에 이미지 없이도 질문만 보고 답을 찍을 수 있어서인지(언어 편향)를 가른다.
같은 manifest·같은 프롬프트 형식(BRIEF)에서 이미지 항목만 뺀다.

  python -m vlm_diagnosis.scripts.no_mem_accuracy \
    --manifest experiments/manifests/gqa_transfer.jsonl --device cuda:2
"""
import argparse
import json
import os
import time

import torch

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.metrics import anls, exact_match

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BRIEF = " Answer with a single word or phrase."


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", default="results/smoke/no_mem.jsonl")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    if a.limit:
        rows = rows[:a.limit]
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if a.resume and os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
                done.add((str(r["sample_id"]), r["question_id"]))
            except Exception:
                pass
        print(f"[resume] {len(done)}문항 건너뜀", flush=True)
    model, processor = load_qwen25vl(device=a.device, max_pixels=1280 * 28 * 28)
    tok = processor.tokenizer

    n = ok = 0
    with open(out, "a" if a.resume else "w") as f:
        for di, r in enumerate(rows):
            t0 = time.time()
            for q in r["questions"]:
                if (str(r["sample_id"]), q["question_id"]) in done:
                    continue
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": q["question"] + BRIEF}]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                ins = tok([text], return_tensors="pt").to(a.device)
                gen = model.generate(**ins, max_new_tokens=a.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                pred = tok.decode(gen[0, ins["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
                em = exact_match(pred, q["answers"])
                n += 1
                ok += em
                f.write(json.dumps({
                    "dataset": r.get("dataset"), "sample_id": r["sample_id"],
                    "question_id": q["question_id"], "role": q.get("role"),
                    "question": q["question"], "gold": q["answers"],
                    "prediction": pred, "em": em,
                    "anls": anls(pred, q["answers"])},
                    ensure_ascii=False) + "\n")
            f.flush()
            if (di + 1) % 20 == 0:
                print(f"[{di+1}/{len(rows)}] EM {ok/max(n,1):.3f} ({n}문항) "
                      f"{time.time()-t0:.1f}s/장", flush=True)
    print(f"DONE: NO-MEM EM {ok/max(n,1):.3f} over {n} → {out}")


if __name__ == "__main__":
    main()
