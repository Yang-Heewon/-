# ScreenQA 접근 smoke 리포트

- 작성일: 2026-08-14
- 성격: **접근 smoke 리포트** (다운로드·라이선스·schema 실측 기록). M3 T4 데이터 선정에 대한
  **결정 문서가 아니다** — 결정은 DECISIONS.md에 별도 기록한다.
- 목적: T4 질문 쌍(같은 화면에 대해 내용 질문 ↔ 위치/grounding 질문)의 소스로 ScreenQA가
  쓸 수 있는지, 특히 UI 요소 bounding box(화면 위 요소의 사각형 좌표)가 실제로 있는지 확인.

## 1. 출처와 라이선스

| 항목 | 내용 |
|---|---|
| 공식 저장소 | https://github.com/google-research-datasets/screen_qa (Google) |
| 데이터 라이선스 | **CC BY 4.0** (저장소 LICENSE 파일 실물 확인) |
| 평가 코드 | `code/metrics.py`, Apache 2.0 (SQuAD 방식 Exact Match/F1) |
| 스크린샷 | 공식 저장소에는 **annotation(JSON)만** 있고 이미지는 없음. 이미지는 RICO(공개 모바일 UI 스크린샷 데이터셋)에서 `image_id`로 별도 취득 |
| 이미지 포함 미러 | HuggingFace `bevaya/RICO-ScreenQA` / `-Short` / `-Complex` (커뮤니티 미러, CC BY 4.0 표기, parquet에 RICO 이미지 동봉) — 스트리밍으로 실물 확인함 |

공식 HuggingFace 릴리즈는 없다. Google의 공식 배포는 GitHub JSON이고, 이미지가 필요한
실험은 RICO 원본(약 6GB) 또는 위 미러를 쓰게 된다.

## 2. 받은 것 (`data/screenqa_probe/`, 총 29MB, gitignored)

- `official/answers_and_bboxes/{validation,train}.json` — 본 변형(full answer + bbox), 8,614 / 68,951건
- `official/short_answers/validation.json` — ScreenQA-Short 변형, 8,614건
- `official/complex_qa/data.json` — ComplexQA 변형, 11,781건
- `official/LICENSE`, `official/README.md`, `official/code/metrics.py`
- `hf_mirror_samples/` — HF 미러에서 스트리밍한 5건의 메타 + 이미지 2장(PNG) + bbox crop 검증 이미지

## 3. Schema 실측 (변형 3종)

세 변형은 역할이 다르다.

| 변형 | 파일 | 답 형식 | bbox |
|---|---|---|---|
| ScreenQA (본) | `answers_and_bboxes/` | 완전한 문장(`full_answer`) + 근거 UI 요소 리스트 | **있음** (answer별 `ui_elements[].bounds`) |
| ScreenQA-Short | `short_answers/` | 짧은 구(모델 생성 후 검수) | 없음 |
| ComplexQA | `complex_qa/` | 짧은 답(세기/비교 중심) | 없음 |

핵심: **bbox는 "answer별 근거 UI 요소 리스트"로 붙는다.** 화면 전체 요소 목록이 아니라,
각 annotator의 답변을 뒷받침하는 요소들(`text` + `bounds`)이다. validation 기준 답변의
95.8%가 비어있지 않은 `ui_elements`를 가지며, 질문 단위로는 91.8%가 bbox 있는 답을 최소
하나 가진다. `vh_index`는 RICO view hierarchy(앱 UI 트리) 내 요소 index이고, -1이면 트리에
매칭되지 않아 OCR 기반으로 그린 박스다(요소의 약 44%가 -1).

full과 Short는 같은 8,614개 (image_id, question) 쌍이 같은 순서로 정렬되어 1:1 대응함을
확인했다(불일치 0건). 즉 Short의 짧은 답 + 본 변형의 bbox를 병합해 쓸 수 있다.

### 샘플 레코드 (실제 데이터에서 발췌)

본 변형 (`answers_and_bboxes/validation.json[0]`, annotator 3명분 중 1명분만 표시):

```json
{
  "image_id": 31,
  "image_width": 1080,
  "image_height": 1920,
  "question": "From whom are you protected?",
  "ground_truth": [
    {
      "full_answer": "You are protected from unauthorized transactions.",
      "ui_elements": [
        {"text": "unauthorized transactions",
         "bounds": [424, 1078, 852, 1117],
         "vh_index": -1}
      ]
    }
  ]
}
```

ScreenQA-Short (같은 질문):

```json
{
  "image_id": 31,
  "question": "From whom are you protected?",
  "ground_truth": ["unauthorized transactions"]
}
```

ComplexQA:

```json
{
  "image_id": 3,
  "question": "How many exercises are there in the workout?",
  "ground_truth": ["12"]
}
```

HF 미러(`bevaya/RICO-ScreenQA`) 레코드는 위 본 변형 schema에
`file_name: "images/rico/31.jpg"`와 `image`(PIL, 1080×1920 RGB)가 추가된 형태이며,
`screen_id`가 공식 `image_id`와 일치함을 확인했다.

### 좌표계 검증

`bounds`는 `[left, top, right, bottom]` **픽셀 좌표**이며, 기준은 레코드의
`image_width`/`image_height`(= 실제 이미지 해상도)다. 검증: image_id 31 이미지(1080×1920)에서
bounds [424,1078,852,1117]를 crop → 정확히 "unauthorized transactions" 텍스트가 나옴.
validation 27,788개 bbox 전수 검사에서 이미지 범위를 벗어난 박스 0건. 해상도는 1080×1920이
79%, 540×960이 21%로 혼재하며 각자 자기 픽셀 공간을 쓴다.

## 4. 이미지당 질문 수 분포 (공식 JSON 전수 집계)

| split | QA 수 | 이미지 수 | 평균 q/img | ≥2문항 | ≥3문항 | 최대 |
|---|---|---|---|---|---|---|
| validation (본) | 8,614 | 3,485 | 2.47 | 67% | **38%** | 9 |
| train (본) | 68,951 | 28,378 | 2.43 | 65% | 38% | 9 |
| ComplexQA | 11,781 | 10,286 | 1.15 | 15% | 0% | 2 |

validation 분포: 1문항 1,157 / 2문항 994 / 3문항 585 / 4문항 350 / 5문항 204 / 6+ 195.
ComplexQA 이미지의 60%(6,190/10,286)가 본 변형 train/val 이미지와 겹치므로, 합치면
validation 이미지의 **43%가 3문항 이상**이 된다. 교차 질문 전이 실험(같은 이미지에 여러
질문)에 충분한 밀도다.

## 5. 위치 질문 자동 생성 가능성: **가능**

answer별 bbox + 이미지 해상도가 있으므로, 내용 질문(원본 QA)과 위치 질문(자동 생성)을
같은 근거 요소에 대해 쌍으로 만들 수 있다. 구체 템플릿:

- **T-A (반구 위치)**: "Is the text '<text>' located in the top half or the bottom half of
  the screen?" — 답은 bbox 중심 `(top+bottom)/2 < image_height/2` 여부로 결정.
  좌/우 반구 버전도 동일. 중앙 밴드(중심이 45–55% 구간)는 모호하므로 제외.
- **T-B (3×3 grid)**: "In which region of the screen is '<text>' — e.g. top-left, center,
  bottom-right?" — bbox 중심을 3×3 칸에 사상. 칸 경계 ±5% 이내 중심은 제외.
- **T-C (상대 순서)**: 같은 화면에서 bbox가 둘 이상인 답 또는 서로 다른 질문의 근거 요소
  두 개를 골라 "Which appears higher on the screen: '<text1>' or '<text2>'?" — 세로 중심
  좌표 비교, 차이가 이미지 높이의 5% 미만이면 제외.

생성 규칙 주의점(함정):

1. **좌표 공간**: bounds는 원본 픽셀 공간. 모델 입력용으로 이미지를 resize하면 bbox도 같은
   비율로 변환해야 한다. 상대 좌표(0–1 정규화)로 먼저 변환해 두는 것이 안전하다.
2. **annotator 간 bbox 편차**: 같은 요소라도 annotator마다 수 픽셀씩 다르다(예: 424 vs 426
   vs 416). 위치 판정은 여러 ground_truth의 bbox 합집합 또는 중앙값 중심으로 하고,
   annotator 간 반구 판정이 갈리면 그 문항은 버린다.
3. **`<text>` 중복**: 같은 텍스트가 화면에 여러 번 나오면(예: "OK" 버튼 다수) 위치 질문이
   모호해진다. 화면 내 텍스트 유일성 검사가 필요한데, 공식 데이터에는 화면 전체 요소
   목록이 없으므로 유일성은 OCR 또는 RICO view hierarchy로 별도 확인해야 한다. 1차로는
   답 텍스트가 길거나(≥2단어) 숫자+단위인 경우만 쓰는 보수적 필터로 회피 가능.
4. **status bar 영역**: 화면 상단 시계·배터리 영역(약 상단 3%)의 요소는 앱 내용이 아니므로
   제외.
5. ComplexQA·Short 단독에는 bbox가 없다 — 위치 질문 생성은 본 변형에서만 한다.

## 6. 남은 리스크

- HF 미러는 커뮤니티 업로드다. 본 실험에서 쓰려면 공식 JSON을 정본으로 삼고, 미러
  이미지가 RICO 원본과 같은지 몇 십 장 표본으로 hash/육안 대조하거나 RICO 원본을 직접
  받는 것이 안전하다 (RICO 자체 약관도 확인 필요 — 이번 smoke 범위 밖).
- bbox는 "답의 근거"이지 화면 전체 레이아웃이 아니다. 요소 유일성·distractor가 필요한
  위치 질문 설계는 추가 신호(OCR/view hierarchy)가 필요할 수 있다.
- 위치 질문의 정답은 규칙으로 생성되므로, 사람이 100문항 정도 표본 검수하는 절차를
  T4 manifest 생성 시 넣어야 한다.
- 540×960 저해상도 화면(21%)은 VLM 입력 전처리에서 1080×1920과 다르게 다뤄질 수 있어
  해상도별 층화가 필요할 수 있다.
