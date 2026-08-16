"""새 도메인의 IMAGE 기준선 측정 (M0-A gate) — 압축 없이 이미지만 주고 정답률을 잰다.

기준선이 너무 낮으면 "압축 때문에 틀렸다"를 판정할 수 없고, 천장에 붙으면 효과가
안 보인다. 본실험 전에 반드시 확인한다.

  python -m vlm_diagnosis.scripts.base_accuracy \
    --manifest experiments/manifests/screenqa_transfer.jsonl --shard 0 --nshards 4
"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.metrics import anls, exact_match
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BRIEF = " Answer with a single word or phrase."
MAX_PIXELS = 1280 * 28 * 28


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", default="results/smoke/base_accuracy.jsonl")
    ap.add_argument("--resume", action="store_true",
                    help="이미 기록된 sample_id는 건너뛰고 이어서 실행")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    rows = rows[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
    out = os.path.join(ROOT, a.out.replace(
        ".jsonl", f".shard{a.shard}.jsonl" if a.nshards > 1 else ".jsonl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model, processor = load_qwen25vl(device=a.device, max_pixels=MAX_PIXELS)

    done = set()
    if a.resume and os.path.exists(out):
        for line in open(out):
            try:
                done.add(str(json.loads(line)["sample_id"]))
            except Exception:
                pass
        print(f"[resume] 이미 완료한 화면 {len(done)}개 건너뜀", flush=True)
    n = ok = 0
    with open(out, "a" if a.resume else "w") as f:
        for di, r in enumerate(rows):
            if str(r["sample_id"]) in done:
                continue
            img = Image.open(os.path.join(ROOT, r["image"])).convert("RGB")
            t0 = time.time()
            for q in r["questions"]:
                ins = S.vlm_inputs(processor, img, q["question"] + BRIEF, a.device)
                pred = greedy_generate_masked(model, processor, ins,
                                              max_new_tokens=a.max_new_tokens)
                em = exact_match(pred, q["answers"])
                n += 1
                ok += em
                f.write(json.dumps({
                    "dataset": r["dataset"], "sample_id": r["sample_id"],
                    "question_id": q["question_id"], "role": q.get("role"),
                    "question": q["question"], "gold": q["answers"],
                    "prediction": pred, "em": em,
                    "anls": anls(pred, q["answers"]),
                    "n_visual_tokens": int(ins["input_ids"].shape[1]),
                }, ensure_ascii=False) + "\n")
            f.flush()
            if (di + 1) % 10 == 0:
                print(f"[{di+1}/{len(rows)}] EM {ok/max(n,1):.3f} "
                      f"({n}문항) {time.time()-t0:.0f}s/screen", flush=True)
    print(f"DONE shard{a.shard}: EM {ok/max(n,1):.3f} over {n} questions → {out}")


if __name__ == "__main__":
    main()
