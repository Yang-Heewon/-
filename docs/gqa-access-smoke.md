# GQA 접근 smoke 리포트

- 작성일: 2026-08-16
- 성격: **접근 smoke 리포트** (다운로드·라이선스·schema 실측 기록). 전이 실험 데이터 선정에
  대한 **결정 문서가 아니다** — 결정은 DECISIONS.md에 별도 기록한다.
- 목적: DocVQA(문서)·ScreenQA(모바일 UI)에 이은 세 번째 도메인(자연 이미지)으로 GQA가
  쓸 수 있는지 확인. 핵심 요건 두 가지: (1) 이미지당 질문이 충분히 많은가, (2) 각 질문의
  답이 이미지 위 사각형 좌표(bounding box, bbox)로 근거 지어지는가 — 이것이 있어야
  ScreenQA처럼 질문 쌍의 같은/다른 근거 라벨을 사람 없이 자동 계산할 수 있다.

## 1. 출처와 라이선스

| 항목 | 내용 |
|---|---|
| 공식 배포 | https://downloads.cs.stanford.edu/nlp/data/gqa/ (Stanford, gqadataset.org) |
| 데이터 라이선스 | **CC BY 4.0** — 공식 download 페이지의 CC BY 4.0 배지(creativecommons.org/licenses/by/4.0/ 링크) 실물 확인. 이미지는 Visual Genome/COCO/Flickr 유래 |
| 질문 | `questions1.2.zip` (1.4GB, Last-Modified 2019-03-26) — 이 중 `val_balanced_questions.json`(114MB)만 HTTP Range 요청으로 부분 추출 (전송량 15.3MB) |
| 장면그래프 | `sceneGraphs.zip` (45MB, Last-Modified 2019-02-03) — `val_sceneGraphs.json`만 추출 |
| 이미지 | 공식 `images.zip`은 **21.8GB라 받지 않음**. 선택된 150장만 HF 미러에서 취득 |
| 이미지 미러 | HuggingFace `lmms-lab/GQA` (→ `lmms-lab-encoder/GQA`로 리다이렉트), 커밋 `a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8` 고정, config `val_balanced_images` (parquet 3샤드, 10,234장). 카드의 라이선스 표기는 "mit"인데 이는 부정확 — 정본 라이선스는 공식 CC BY 4.0을 따른다 |

## 2. 받은 것 (`data/gqa_probe/` 152MB + `data/gqa_pilot/` 21MB, gitignored)

- `gqa_probe/val_balanced_questions.json` — val balanced 질문 132,062건 (dict: qid → record)
- `gqa_probe/val_sceneGraphs.json` — 이미지 10,696장의 장면그래프(scene graph: 이미지 속
  객체·속성·관계 주석)
- `gqa_probe/readme.txt` — sceneGraphs.zip 동봉 readme
- `gqa_pilot/*.jpg` — manifest에 선택된 150장 (row-group 단위 parquet 부분 다운로드,
  장당 sha256 기록, 전수 디코딩 + 장면그래프 width/height 일치 검증 통과)

## 3. Schema 실측

질문 레코드 (val_balanced_questions.json, 실제 데이터 발췌):

```json
{
  "question": "What is this bird called?",
  "imageId": "2405722",
  "answer": "parrot",
  "fullAnswer": "This is a parrot.",
  "annotations": {
    "answer":     {"0": "329774"},
    "question":   {"3": "329774"},
    "fullAnswer": {"3": "329774"}
  },
  "types": {"structural": "query", "semantic": "cat", "detailed": "categoryThis"},
  "isBalanced": true, "entailed": ["05515937", "..."], "semantic": ["..."]
}
```

장면그래프 레코드 (val_sceneGraphs.json["2405722"]):

```json
{
  "width": 500, "height": 375, "location": "outdoors",
  "objects": {
    "329774": {"name": "parrot", "x": ..., "y": ..., "w": ..., "h": 320,
               "attributes": [...], "relations": [{"object": "329800", "name": "to the right of"}, ...]}
  }
}
```

확인된 사실:

- `annotations.answer`는 **답 텍스트의 토큰 → 장면그래프 객체 id** 매핑이다. val balanced
  132,062건 중 **49,474건(37.5%)이 answer 주석을 가지며, 있으면 항상 객체 1개**다
  (2개 이상인 경우 0건 — union bbox 분기는 방어적으로만 존재).
- 나머지 82,588건은 answer 주석이 없다: yes/no(verify 27,413 + logical 대부분) 질문은
  **전부** answer 주석이 없고(46,750건 중 0건), fullAnswer 주석만 있는 경우가 74,253건.
  과제 규칙("답 객체 주석이 있는 질문만")대로 이들은 전량 제외했다.
- bbox는 `x, y, w, h` **픽셀 좌표**, 기준은 장면그래프의 `width`/`height`(= 실제 이미지
  해상도, 150장 전수 디코딩 대조로 확인). 주석된 answer 객체는 장면그래프에 **전부
  존재**하고 퇴화 bbox(w≤0/h≤0) 0건. 다만 49,474건 중 684건(1.4%)은 bbox가 이미지
  경계를 약간 벗어나 경계로 클리핑했고, 1건은 클리핑 후 퇴화해 제외했다.
- 해상도는 대부분 긴 변 500px (최빈 500×375). ScreenQA(1080×1920)보다 훨씬 작다.
- HF 미러 parquet은 `id`(공식 imageId와 일치 확인) + `image`(JPEG bytes) 구조.

## 4. 이미지당 질문 수 분포 (val balanced 전수 집계)

- 전체: 132,062질문 / 10,234이미지 (평균 12.9 q/img) — 단 answer 주석 필터 후가 기준.
- **근거 bbox 확보 질문 49,473건 / 8,215이미지 (평균 6.0 q/img, 최대 59)**.
- 문항수 분포(근거 확보 기준): ≥3문항 5,777장 / **≥5문항 3,957장** / ≥8문항 2,238장.
- ScreenQA(≥5문항 322장)의 12배 밀도 — 교차 질문 전이 실험에 여유가 크다.

## 5. 자동 쌍 라벨: **가능** — manifest 생성 완료

생성기 `vlm_diagnosis/scripts/prep_gqa_transfer.py` (seed 42, 재실행 시 바이트 동일 확인):

- ≥5문항 이미지 3,957장 → 150장 선택, 이미지당 최대 8문항 캡 → **질문 1,050개**
  (역할: episode 150 / source 450 / heldout 450; 구조 유형은 query 1,012 · choose 35 ·
  compare 3 — verify/logical은 answer 주석이 없어 자연 배제됨).
- 순서 있는 쌍 6,516개, ScreenQA와 동일 임계값(T2 = IoU≥0.5 또는 중심거리<0.03,
  T3 = 거리≥0.10, 그 사이 = partial):

| 라벨 | 쌍 수 | 비율 | (참고) ScreenQA |
|---|---|---|---|
| T2 (같은 근거) | 1,802 | 27.7% | 2.5% |
| partial | 602 | 9.2% | 13.8% |
| T3 (다른 근거) | 4,112 | 63.1% | 83.8% |

- source→heldout 방향 쌍만 보면 1,350개 (T2 373 / partial 120 / T3 857).
- T2가 ScreenQA보다 훨씬 많은 이유: GQA는 같은 객체에 대한 entailed(함의)·동치 질문을
  체계적으로 생성하므로 같은 answer 객체를 공유하는 질문이 흔하다. T2 쌍의 87%는 짧은 답도
  동일하다(같은 객체를 다른 표현으로 묻는 준-패러프레이즈).
- 채점: `answers`에는 짧은 답(`answer`) 1개만 넣었다(EM 채점용). `fullAnswer`는 문장이라
  EM에 부적합해 `full_answer` 필드로 별도 보존.

산출물: `experiments/manifests/gqa_transfer.jsonl` · `gqa_transfer_pairs.jsonl` ·
`gqa_transfer.meta.json` (ScreenQA manifest와 같은 레코드 schema).

## 6. 남은 리스크

- **T2 쌍의 답 중복**: T2 쌍의 87%가 짧은 답까지 같다. 전이 성공이 "근거 재사용"이 아니라
  "답 복사"일 가능성을 가르려면, 답이 다른 T2 쌍(234개)과 답이 같은 T2 쌍을 나눠 분석하는
  층화가 필요하다.
- 이미지 미러는 커뮤니티 업로드(카드 라이선스 표기도 부정확)다. 150장이 장면그래프
  해상도와 전부 일치함은 확인했지만, confirmation 단계 전에 공식 `images.zip`(21.8GB) 또는
  Visual Genome 원본과 표본 hash 대조를 권장한다.
- bbox는 Visual Genome 사람 주석 유래라 느슨한 경우가 있다(객체보다 크게 그려진 박스,
  경계 밖 684건 클리핑). 자동 라벨의 표본 검증(시각 렌더 12쌍 1차 통과)을 사람이 한 번 더
  하는 절차는 유지해야 한다.
- 장면그래프의 `location`("bedroom" 등) 답 질문은 근거 bbox가 이미지 대부분을 덮는 큰
  박스일 수 있다(예: "Where is the man?" → 화면의 66%). 큰 bbox 질문은 위치 구분력이
  낮으므로 bbox 면적 비율을 공변량으로 기록해 분석 시 통제할 수 있다(면적은 좌표에서
  계산 가능, manifest에 이미 좌표 있음).
- GQA 질문은 템플릿 생성이라 문체가 단조롭다 — 자연어 다양성이 필요한 해석에는 한계.
