"""T0–T4 라벨링 검수표 생성기 v2 — 2026-08-14 검수 반영판.

v1 대비 변경 (검수 지적사항):
  1. T4 휴리스틱 교체: 질문의 단어(where/located)가 아니라 **답이 위치·방향인 경우만**
     T4 후보 (지리적 where의 답=지명 → semantic). 위치가 단서일 뿐인 OCR은 T4 아님.
  2. T1/T2 규칙을 동결된 가이드 트리에 정합: T1=바꿔 말하기(같은 답·같은 블록 추정),
     T2=다른 질문·같은 블록(사람 확인 필수 — 텍스트만으로 추정 불가한 경우 T3 기본).
  3. 검수에서 확정된 교정 13건을 override로 내장.
  4. T0 자기쌍(q1–q1 등, 파일럿 self 평가에 실제 사용됨) 추가.
  5. 2인 검수 구조: label_A/notes_A, label_B/notes_B, adjudicated_label.
  6. evidence 블록·overlap(same/partial/different)·질문별 task type 초안 필드 추가.

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
VERSION = "t-label-draft-v2"

# 답 자체가 위치/방향일 때만 layout (동결된 T4 규칙)
POSITIONAL_ANSWERS = {"top", "bottom", "left", "right", "center", "middle",
                      "first", "last", "top left", "top right",
                      "bottom left", "bottom right"}

# 2026-08-14 검수에서 이미지 대조로 확정된 교정.
# key: (sample_id, qA_id or None, qB_id or None) — None은 와일드카드.
REVIEW_OVERRIDES = [
    (("1936", "4525", "4527"), "T3", False,
     "검수 확정: 같은 값 14,500이지만 Winston/Camel의 다른 표 셀 — 근거 블록 다름"),
    (("7691", None, "61371"), "T3", False,
     "검수 확정: 'Where is the coffee mill?'의 답 Kona는 지명(semantic) — layout 아님"),
    (("14533", None, "50209"), "T3", True,
     "검수: 'located'가 지리적 의미 — 같은 블록이면 T2로 adjudication"),
    (("14179", None, "49081"), "T3", False,
     "검수 확정: Nashville은 지리 정보(semantic) — layout 아님"),
    (("14332", "49530", "49536"), "T3", False,
     "검수 확정: figure number는 단순 OCR — 유형 교차 아님"),
    (("14332", "49530", "49542"), "T3", False,
     "검수 확정: figure number는 단순 OCR — 유형 교차 아님"),
]


def find_override(sample_id, qa_id, qb_id):
    for (sid, a, b), label, unc, note in REVIEW_OVERRIDES:
        if sid != str(sample_id):
            continue
        ids = {qa_id, qb_id}
        if (a is None or a in ids) and (b is None or b in ids):
            return label, unc, note
    return None


def token_overlap(a, b):
    ta, tb = set(normalize_text(a).split()), set(normalize_text(b).split())
    return len(ta & tb) / max(len(ta | tb), 1)


def answers_equal(a, b):
    return bool({normalize_text(x) for x in a} & {normalize_text(x) for x in b})


def task_type_draft(q):
    """primary_task_type 초안 — 동결 규칙: 답에 필요한 능력 기준."""
    ans = [normalize_text(x) for x in q["answers"]]
    if any(x in POSITIONAL_ANSWERS for x in ans):
        return "layout"
    if re.match(r"\s*how many\b", q["question"], re.I):
        return "count"
    return "OCR/semantic"


def draft_label(qa, qb):
    """(label, uncertain, rationale). 텍스트 정보만으로의 초안 — 근거 블록은 검수자 몫."""
    if normalize_text(qa["question"]) == normalize_text(qb["question"]):
        return "T0", False, "질문 동일"
    ta, tb = task_type_draft(qa), task_type_draft(qb)
    if (ta == "layout") != (tb == "layout"):
        return "T4", True, f"답 유형 교차 후보 ({ta} vs {tb}) — 동결 규칙으로 확인"
    if answers_equal(qa["answers"], qb["answers"]):
        ov = token_overlap(qa["question"], qb["question"])
        if ov >= 0.6:
            return "T1", True, f"같은 답·질문 겹침 {ov:.2f} — 바꿔 말하기인지 확인"
        return "T2", True, (f"같은 답·다른 질문(겹침 {ov:.2f}) — 같은 블록이면 T2, "
                            "다른 블록이면 T3")
    return "T3", True, "답 다름 — 근거 블록이 같으면 T2로 정정 (이미지 확인 필요)"


def main():
    rows = []
    for doc in map(json.loads, open(MANIFEST)):
        qs = doc["questions"][:6]

        def add(i, j, used_in):
            qa, qb = qs[i], qs[j]
            if i == j:
                label, unc, why = "T0", False, "자기쌍 (파일럿 self 평가 행)"
            else:
                ov = find_override(doc["sample_id"], qa["question_id"],
                                   qb["question_id"])
                if ov:
                    label, unc, why = ov
                else:
                    label, unc, why = draft_label(qa, qb)
            rows.append({
                "pair_id": f"{doc['sample_id']}_{qa['question_id']}_{qb['question_id']}",
                "sample_id": doc["sample_id"], "image": doc["image"],
                "used_in": used_in,
                "qA_id": qa["question_id"], "qA": qa["question"],
                "qA_answers": " | ".join(qa["answers"]),
                "qA_type_draft": task_type_draft(qa),
                "qA_evidence_block": "",          # ← 검수자: 문단/표 셀/제목/범례/그림
                "qB_id": qb["question_id"], "qB": qb["question"],
                "qB_answers": " | ".join(qb["answers"]),
                "qB_type_draft": task_type_draft(qb),
                "qB_evidence_block": "",
                "evidence_overlap": "",           # ← 검수자: same/partial/different
                "draft_label": label, "draft_rationale": why, "uncertain": unc,
                "label_A": "", "notes_A": "",     # ← 검수자 A
                "label_B": "", "notes_B": "",     # ← 검수자 B
                "adjudicated_label": "", "adjudication_note": "",
            })

        for i in (1, 2, 3):                       # T0 자기쌍 (self 평가에 사용)
            if i < len(qs):
                add(i, i, "self(T0)")
        for i, j in combinations(range(min(4, len(qs))), 2):
            add(i, j, "episode(q0)" if i == 0 else "cross")
        for i in (1, 2, 3):
            for j in (4, 5):
                if i < len(qs) and j < len(qs):
                    add(i, j, "heldout")

    df = pd.DataFrame(rows)
    guide = pd.DataFrame([
        ["T0", "같은 질문 반복 (자기쌍 포함)"],
        ["T1", "바꿔 말하기 — 같은 답·같은 evidence 블록, wording만 다름"],
        ["T2", "바꿔 말하기가 아닌 다른 질문인데 evidence 블록이 same (답은 같을 수도 다를 수도)"],
        ["T3", "evidence 블록이 다름"],
        ["T4", "primary_task_type 교차: {OCR/semantic/count} ↔ {layout/grounding/icon}"],
        ["", ""],
        ["T4 동결 규칙", "답에 필요한 능력 기준. 답이 지명이면 semantic (where라는 단어 무관)."],
        ["", "위치가 단서일 뿐 답이 글자면 OCR. 답 자체가 위치/방향일 때만 layout."],
        ["evidence 단위", "의미 블록: 문단 / 표 셀 / 제목·머리글 / 범례 / 그림 영역"],
        ["overlap", "same / partial / different (partial이면 uncertain 표시)"],
        ["검수 절차", "A·B가 독립적으로 label_A/label_B 기입 → 불일치·uncertain만"],
        ["", "adjudicated_label로 확정. 완료 후 Claude에게 알리면 κ 계산·변환 실행."],
    ], columns=["항목", "내용"])

    with pd.ExcelWriter(OUT_X, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="pairs", index=False)
        guide.to_excel(w, sheet_name="라벨_정의", index=False)
        ws = w.sheets["pairs"]
        widths = {"C": 28, "D": 12, "F": 44, "G": 22, "H": 12, "I": 18,
                  "K": 44, "L": 22, "M": 12, "N": 18, "O": 14, "P": 8,
                  "Q": 40}
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
    with open(OUT_J, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by = df[df.used_in != "self(T0)"]["draft_label"].value_counts().to_dict()
    n_ovr = sum(1 for r in rows if "검수" in r["draft_rationale"])
    print(f"[{VERSION}] 쌍 {len(rows)}개 (자기쌍 {sum(df.used_in=='self(T0)')} 포함), "
          f"override {n_ovr}건 적용, 비자기쌍 초안 분포 {by}")
    print(f"→ {OUT_X}\n→ {OUT_J}")


if __name__ == "__main__":
    main()
