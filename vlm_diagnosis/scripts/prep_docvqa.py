"""D4 mini 데이터 준비: DocVQA validation에서 질문 ≥K개인 문서 N개 수집.

2-pass 스트리밍: ① 이미지 디코드 없이 docId별 질문 수 집계 → ② 선정된 docId의
이미지·QA 저장. 산출: data/d4_mini/{docId}.png + meta.jsonl
"""
import argparse, json, os
from datasets import load_dataset

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "data", "d4_mini")


def main(n_docs=32, k_min=4, scan_rows=4000):
    os.makedirs(OUT, exist_ok=True)
    # pass 1: 질문 수 집계 (image 컬럼 제외 → 디코드 회피)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
    meta_ds = ds.select_columns(["questionId", "question", "answers", "docId"])
    counts, order = {}, []
    for i, ex in enumerate(meta_ds):
        if i >= scan_rows:
            break
        d = ex["docId"]
        if d not in counts:
            counts[d] = 0
            order.append(d)
        counts[d] += 1
    chosen = [d for d in order if counts[d] >= k_min][:n_docs]
    print(f"스캔 {min(scan_rows, i+1)}행 → 문서 {len(counts)}개, ≥{k_min}문항 {sum(1 for d in counts if counts[d]>=k_min)}개, 선정 {len(chosen)}개")
    chosen_set = set(chosen)

    # pass 2: 이미지 + QA 저장
    docs = {d: {"docId": d, "questions": []} for d in chosen}
    saved = set()
    for i, ex in enumerate(ds):
        if i >= scan_rows:
            break
        d = ex["docId"]
        if d not in chosen_set:
            continue
        if d not in saved:
            path = os.path.join(OUT, f"{d}.png")
            ex["image"].convert("RGB").save(path)
            docs[d]["image"] = path
            saved.add(d)
        docs[d]["questions"].append(
            {"qid": ex["questionId"], "q": ex["question"], "answers": ex["answers"]})

    with open(os.path.join(OUT, "meta.jsonl"), "w") as f:
        for d in chosen:
            f.write(json.dumps(docs[d], ensure_ascii=False) + "\n")
    print(f"저장 완료: 문서 {len(saved)}개 → {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=32)
    ap.add_argument("--k_min", type=int, default=4)
    ap.add_argument("--scan_rows", type=int, default=4000)
    a = ap.parse_args()
    main(a.n_docs, a.k_min, a.scan_rows)
