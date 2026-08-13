# G02 — T_visual 생성 사양 (PROPOSED)

> **상태: PROPOSED.** 최종 도구 선택·prompt 동결·토큰 예산은 사용자 결정(G02)이며,
> 채택 시 DECISIONS.md §14에 기록한다. 이 baseline이 약하면 modality 비교 전체가
> 무효가 되므로(§0.1 thesis), 아래 anti-strawman 체크리스트를 통과해야 한다.

`T_visual = T_o + T_d + T_u` (SHARED_PROTOCOL §1). write-time에 생성하고 read 문맥에
텍스트로 주입한다. write-time 신호 규칙상 이미지 자체와 그 시점의 모델 출력은 모두
합법이다. **미래 질문은 어떤 구성요소에도 들어가지 않는다.**

## 1. T_o — OCR + bbox

| 후보 | 장점 | 단점 | V100/CPU |
|---|---|---|---|
| **PaddleOCR (권장)** | 문서·장면 문자 강함, line/word bbox, 다국어 | 별도 의존성 | GPU/CPU 모두 가능 |
| docTR | HF 생태계, 좋은 문서 성능 | 속도 | GPU 권장 |
| Tesseract (fallback) | 설치 쉬움, CPU | 저해상도·장면 문자 약함 | CPU |
| Qwen2.5-VL 자체 OCR prompt | 추가 의존성 없음 | **순환성**: 평가 대상 모델이 baseline 생성 — 모델의 OCR 한계가 양쪽에 같이 들어가 비교가 물러짐 | GPU |

권장: **PaddleOCR 주 + Tesseract fallback**, Qwen 자체 OCR은 보조 비교군으로만.
버전·모델 가중치 revision을 DECISIONS에 고정한다.

출력 형식 (word-level, 원본 픽셀 좌표):

```json
{"words": [{"text": "ITC", "bbox": [x0, y0, x1, y1], "conf": 0.98}, ...]}
```

직렬화: line 단위로 묶고 각 line에 대표 bbox를 붙인 layout-preserving 텍스트:

```text
[OCR]
(120,88,340,132) ITC Limited
(96,210,610,258) Aashirvaad Atta — Whole Wheat
...
```

- 좌표는 **정수 픽셀, 원본 이미지 기준**. 소수점·스케일 변환 금지 (grounding 공정성).
- confidence < 0.5 단어도 버리지 않고 `?` 표시로 포함 (정보 손실 방지).

## 2. T_d — dense description

생성기: **Qwen2.5-VL 자체** (write-time 합법 — 과거 이미지를 그 시점에 서술).
결정론: greedy(do_sample=false), max_new_tokens 512.

권장 prompt (동결 대상):

```text
Describe this image in exhaustive detail. Include: every visible text string
and number; the type of document or screen; all objects, icons, and buttons;
their colors and states; the spatial layout (what is where, reading order);
and any tables, charts, or figures with their contents.
```

주의: **T_d 생성기의 품질이 TEXT baseline의 상한을 결정한다.** report에는 항상
"T_visual (Qwen2.5-VL-generated T_d)"처럼 생성기를 명기한다.

## 3. T_u — UI tree

- ScreenQA류 화면: dataset이 제공하는 UI element annotation(type/label/bounds)을
  그대로 직렬화 — 모델 생성보다 우선.
- annotation이 없는 화면: Qwen2.5-VL로 element 열거 prompt (보조, 라벨에 `model_generated`).
- 문서 이미지: `T_u = N/A`로 기록 (빈 문자열, 필드는 유지).

```text
[UI]
button "Submit" (840,1180,1010,1240) enabled
input "Search…" (120,80,700,140) focused
```

## 4. 직렬화와 read-문맥 주입

순서 고정: `[OCR] → [LAYOUT-SUMMARY(선택)] → [UI] → [DESCRIPTION]`.
system/question과의 결합은 M1-02(write/read 순서) 결정을 따른다.

기록 의무 (SHARED_PROTOCOL §3): 각 표본의 `T_visual` **토큰 수와 serialized bytes**를
record에 남긴다. byte 예산 비교축(B grid)에는 넣지 않지만 M6 비용축에 필요하다.

## 5. Anti-strawman 체크리스트 (전부 만족해야 공정한 TEXT baseline)

- [ ] OCR이 word/line bbox를 **포함**한다 (bbox 없는 순수 텍스트 금지)
- [ ] OCR 결과를 토큰 예산 때문에 임의 truncation하지 않는다 (긴 문서는 전체 포함,
      context 한계 초과 시 초과 사실을 record에 표시하고 해당 표본을 별도 층으로)
- [ ] T_d가 generic caption 1–2문장이 아니라 exhaustive description이다
- [ ] UI annotation이 있으면 반드시 사용한다 (모델 생성으로 대체하지 않는다)
- [ ] 세 구성요소를 임의로 빼지 않는다 — ablation(T_o만, T_d만)은 별도 조건으로 실행
- [ ] 생성기·버전·prompt가 DECISIONS에 동결돼 있다

## 6. 사용자가 확정할 것

- [ ] T_o 도구 최종 선택 (권장: PaddleOCR) + 버전 고정
- [ ] T_d prompt 문안 동결 + max_new_tokens (권장 512)
- [ ] T_u의 model-generated fallback 허용 여부
- [ ] context 한계 초과 표본 처리 규칙
