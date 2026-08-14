"""T0–T4 라벨링 초안 생성기 — 사람 검수용 Excel + 기계용 JSONL.

실험에 실제 사용된 질문 쌍(문서당 최대 12쌍)을 나열하고, 규칙 기반 초안 라벨을
붙인다. 초안은 질문·답 텍스트만 보고 만든 것이므로(이미지의 근거 위치는 못 봄)
T1/T2/T4와 애매한 T3에는 uncertain=true를 달아 검수자가 이미지를 보고 확정한다.

쌍 구성 (실험에서 실제 쓰인 조합):
  - q0(write)–q1..q3        : h2o 해석에 쓰인 쌍
  - q1..q3 상호             : 교차 평가에 쓰인 쌍
  - q1..q3 × q4..q5         : held-out 평가에 쓰인 쌍

초안 규칙 (M3-01 가이드의 판정 트리를 텍스트 정보로 근사):
  T0: 정규화 후 질문 동일
  T1: 답 동일 + 질문 토큰 겹침 ≥ 0.6 (바꿔 묻기 후보)
  T2: 답 동일 + 질문 겹침 < 0.6 (같은 근거를 다르게 묻기 후보)
  T4: 한쪽만 공간형 질문(where/side/top/position/page number 등) (유형 교차 후보)
  T3: 그 외 (다른 근거 후보 — 기본값)

실행:
  python -m vlm_diagnosis.scripts.gen_t_label_draft
"""
import json
import os
import re
from itertools import combinations

import pandas as pd

from vlm_diagnosis.core.metrics import normalize_text

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MANIFEST = os.path.join(ROOT, "experiments", "manifests", "m2a_diagnostic.jsonl")
OUT_X = os.path.join(ROOT, "experiments", "manifests", "m3_pairs_draft.xlsx")
OUT_J = os.path.join(ROOT, "experiments", "manifests", "m3_pairs_draft.jsonl")

SPATIAL = re.compile(
    r"\b(where|side|top|bottom|left|right|position|located|corner|page number|"
    r"figure number|x axis|y axis|axis)\b", re.I)


def token_overlap(a, b):
    ta, tb = set(normalize_text(a).split()), set(normalize_text(b).split())
    return len(ta & tb) / max(len(ta | tb), 1)


def answers_equal(a, b):
    na = {normalize_text(x) for x in a}
    nb = {normalize_text(x) for x in b}
    return bool(na & nb)


def draft_label(qa, qb):
    if normalize_text(qa["question"]) == normalize_text(qb["question"]):
        return "T0", "질문 동일", False
    sa, sb = bool(SPATIAL.search(qa["question"])), bool(SPATIAL.search(qb["question"]))
    if sa != sb:
        return "T4", "한쪽만 공간형 질문 (유형 교차 후보)", True
    if answers_equal(qa["answers"], qb["answers"]):
        ov = token_overlap(qa["question"], qb["question"])
        if ov >= 0.6:
            return "T1", f"답 동일·질문 겹침 {ov:.2f} (바꿔 묻기 후보)", True
        return "T2", f"답 동일·질문 겹침 {ov:.2f} (같은 근거 다른 질문 후보)", True
    return "T3", "답 다름 (다른 근거 후보 — 이미지에서 근거 위치 확인 필요)", True


def main():
    rows = []
    for doc in map(json.loads, open(MANIFEST)):
        qs = doc["questions"][:6]
        idx_pairs = [p for p in combinations(range(min(4, len(qs))), 2)]
        idx_pairs += [(i, j) for i in (1, 2, 3) for j in (4, 5)
                      if i < len(qs) and j < len(qs)]
        for i, j in idx_pairs:
            qa, qb = qs[i], qs[j]
            label, why, unc = draft_label(qa, qb)
            rows.append({
                "pair_id": f"{doc['sample_id']}_{qa['question_id']}_{qb['question_id']}",
                "sample_id": doc["sample_id"],
                "image": doc["image"],
                "role": f"q{i}-q{j}" + (" (held-out쌍)" if j >= 4 else ""),
                "qA_id": qa["question_id"], "qA": qa["question"],
                "qA_answers": " | ".join(qa["answers"]),
                "qB_id": qb["question_id"], "qB": qb["question"],
                "qB_answers": " | ".join(qb["answers"]),
                "draft_label": label, "draft_rationale": why,
                "uncertain": unc,
                "final_label": "",       # ← 검수자 기입
                "reviewer_notes": "",    # ← 검수자 기입
            })
    df = pd.DataFrame(rows)

    guide = pd.DataFrame([
        ["T0", "같은 질문 반복 (표현까지 동일)"],
        ["T1", "바꿔 묻기 — 답과 근거 위치가 같고 말만 다름"],
        ["T2", "다른 질문이지만 같은 근거 위치를 봄 (답도 대개 같음)"],
        ["T3", "같은 이미지의 다른 근거 위치를 봄 (답 다름)"],
        ["T4", "정보 유형이 교차 (내용 질문 ↔ 위치/공간 질문)"],
        ["", ""],
        ["검수 방법", "image 열의 사진을 열고, 두 질문 각각 '어디를 봐야 답할 수 있나'를"],
        ["", "확인 → 그 위치가 같으면 T1/T2, 다르면 T3, 유형이 갈리면 T4."],
        ["", "final_label에 확정 라벨 기입. 애매하면 notes에 이유."],
    ], columns=["라벨", "정의"])

    with pd.ExcelWriter(OUT_X, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="pairs", index=False)
        guide.to_excel(w, sheet_name="라벨_정의", index=False)
        ws = w.sheets["pairs"]
        for col, width in {"E": 46, "H": 46, "F": 24, "I": 24, "K": 40,
                           "C": 30, "D": 14}.items():
            ws.column_dimensions[col].width = width
    with open(OUT_J, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_unc = sum(r["uncertain"] for r in rows)
    by = df["draft_label"].value_counts().to_dict()
    print(f"쌍 {len(rows)}개 (문서 {df['sample_id'].nunique()}개), "
          f"검수 필요 {n_unc}개, 초안 분포 {by}")
    print(f"→ {OUT_X}\n→ {OUT_J}")


if __name__ == "__main__":
    main()
