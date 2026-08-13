# M3-01 — T0–T4 질문 쌍 라벨링 가이드 (PROPOSED)

> **상태: PROPOSED.** M3-01의 최종 결정(가이드 채택, 검수자 2인 지정)은 사용자 몫이다.
> 이 가이드는 결과를 보기 전에 동결해야 하며, 채택 시 DECISIONS.md §14에 기록한다.

같은 이미지의 질문 쌍 `(q_w, q_r)`에 T0–T4 라벨을 붙인다. 라벨은 "무엇이 바뀌었는가"를
**wording → answer → evidence → 정보 유형** 순서로 판정한다. T5는 존재하지 않는다
(G10 UNVERIFIED — 임의로 만들지 않는다).

## 1. 조작적 정의

**Evidence region**: 사람이 그 질문에 답하기 위해 반드시 봐야 하는 이미지 영역(bbox 또는
UI element 집합). 라벨러는 각 질문에 대해 evidence region을 먼저 그리고(또는 기존
annotation을 사용), 쌍 비교는 그 다음에 한다.

**Evidence overlap**: `IoU(region_w, region_r)` 또는 element 집합의 Jaccard.
경계값: overlap ≥ 0.5 → "같은 근거", < 0.5 → "다른 근거". 0.3–0.7 구간이면
`uncertain=true`를 함께 기록한다.

**정보 유형**: M4와 동일한 6분류 — OCR / semantic / layout / grounding / icon / count.
질문이 요구하는 유형을 각 질문에 하나 이상 부여한다.

## 2. 판정 트리

```text
1. q_w == q_r (문자열 동일 또는 공백/대소문자 차이뿐)?          → T0
2. 답과 evidence region이 같고 wording만 다른가?                → T1
3. 답은 다르지만 evidence overlap ≥ 0.5 인가?                   → T2
4. evidence overlap < 0.5 이고 정보 유형은 같은 그룹인가?       → T3
5. 정보 유형이 {semantic, OCR} ↔ {layout, grounding, icon} 를
   가로지르는가? (evidence 이동 여부와 무관)                    → T4
```

- 4·5가 동시에 성립하면 **T4가 우선**한다 (유형 교차가 더 강한 조건).
- 어느 단계에서도 판단이 갈리면 `uncertain=true` + adjudication으로 보낸다.

## 3. 예시 (d4_mini 실측 질문 사용)

doc 4733 (ITC 광고 문서):

| 쌍 | 질문 | 라벨 | 근거 |
|---|---|---|---|
| (57349, 57349) | "What is the name of the company?" ×2 | T0 | 동일 질문 |
| (57349, "Which company published this ad?") | 합성 paraphrase | T1 | 같은 답(itc limited)·같은 로고 영역 |
| (57357, 57364) | Atta 브랜드(aashirvaad) ↔ choco fills 이름(dark fantasy) | T3 | 둘 다 OCR이지만 서로 다른 제품 영역 |
| (57349, "Where in the page is the ITC logo, top or bottom?") | 합성 | T4 | OCR/semantic → layout 교차 |

T2 예시(합성): "What is the passcode?" ↔ "How many characters does the code have?" —
같은 코드 영역(overlap ≥ 0.5), 다른 답. 단 두 번째는 count 유형이므로 유형이 갈리면
T4 규칙을 먼저 확인할 것 (이 예는 OCR↔count로 같은 텍스트 그룹 → T2 유지).

## 4. 엣지 케이스 규칙

- **부분 겹침**: overlap 0.3–0.7 → 계산값 기록 + `uncertain=true`.
- **다영역 evidence**: region 합집합으로 IoU 계산. 서로 소인 영역이 하나라도 겹치면
  overlap에 반영된다.
- **표/차트 내부 이동**: 같은 표 안의 다른 셀은 "다른 근거"로 본다 (T3), 같은 셀의
  다른 속성(값 vs 단위)은 같은 근거 (T2).
- **답이 이미지 밖 상식으로도 가능**: `answerable_without_image=true`를 기록하고
  M3 분석에서 별도 층으로 둔다.
- **grounding 질문의 evidence**: 클릭 대상 element 자체가 region이다.

## 5. 2인 검수 워크플로

```text
1. annotator A, B가 독립적으로 전 쌍 라벨링 (서로의 라벨 비공개)
2. 자동 비교 → 불일치·uncertain 쌍만 adjudication 회의로
3. adjudicator(제3자 또는 합의)가 최종 라벨 확정, 사유 한 줄 기록
4. inter-annotator agreement (Cohen's κ, 라벨 5종)를 report에 기록
5. κ < 0.6이면 가이드 규칙을 보강하고 해당 배치 재라벨 (결과를 본 뒤 규칙을 바꾸지 않는다)
```

## 6. 출력 schema (m3_pairs.jsonl)

```json
{"dataset": "DocVQA", "dataset_revision": "<commit>", "split": "discovery",
 "sample_id": "4733", "image": "data/docvqa_manifest/4733.png",
 "pair_id": "4733_57349_57357", "question_ids": ["57349", "57357"],
 "pair_label": "T3", "evidence_overlap": 0.12,
 "task_types": ["semantic", "OCR"], "uncertain": false,
 "adjudicated": false, "answerable_without_image": false,
 "annotators": ["A", "B"], "selection_seed": 42}
```

`experiments/manifests/README.md`의 공통 필드 규칙을 따른다. adjudication을 거친 쌍은
`adjudicated=true`와 최종 사유를 `.meta.json`에 남긴다.

## 7. 사용자가 확정할 것

- [ ] 가이드 채택 여부와 수정 사항 (M3-01)
- [ ] annotator 2인과 adjudicator 지정
- [ ] evidence overlap 경계값 0.5 (권장 시작점) 유지 여부
- [ ] 이미지당 쌍 수 (M3-03과 연동; 권장: 유형별 최소 1쌍)
