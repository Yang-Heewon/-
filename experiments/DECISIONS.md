# 사용자 결정 로그

이 파일에는 코드가 대신 결정하면 안 되는 값과 과거 설계의 채택·조건부·descope 상태를 기록한다.
권장안은 결정값이 아니다. 결정을 내리면 날짜, 근거, 당시 볼 수 있었던 evidence를 남긴다.

## 1. 전역 결정

| ID | 결정할 것 | 상태·현재값 | 권장 시작점 | 결정 시점 |
|---|---|---|---|---|
| G00 | source revision 관리 | DECIDED: root Git+baseline submodule pin | config/manifest hash도 결과에 복사 | 2026-08-13 |
| G01 | published baseline 목록과 압축 시점 | TBD | write-time 1–2개, read-time 1–2개를 분리 | M2-A 구현 전 |
| G02 | `T_visual` 생성 모델·revision·prompt | TBD | OCR+bbox, dense caption, UI tree를 분리 | M1-F 전 |
| G03 | discovery 표본 수와 ID | TBD | pilot variance 후 manifest 동결 | 본실험 전 |
| G04 | M7 모델 독립성 기준과 후보 | TBD | position/fusion/LLM 중 주장 대상부터 고정 | M7 freeze 전 |
| G05 | 실용적 최소 효과 | TBD | task 또는 SLO 의미로 결정 | M7 freeze 전 |
| G06 | model·processor·dataset revision | TBD | 최초 해당 stage smoke 전에 hash 고정 | 각 단계 전 |
| G07 | text-only modality control | CONDITIONAL | visual-specific 주장일 때 M2-A 소표본 포함 | M2-A 전 |
| G08 | loyalty 정의 | CONDITIONAL | QA normalized agreement, action target agreement | metric 구현 전 |
| G09 | D2/D3 신호 진단 | CONDITIONAL | M2-A/M3 selector failure 뒤에만 실행 | M3 판정 후 |
| G10 | 과거 T5 정의의 처리 | UNVERIFIED | 원 정의를 찾은 뒤 absorb 또는 descope | M3 manifest 전 |

### G04 결정 규칙

후보를 “Qwen/비-Qwen” 한 축으로만 나누지 않는다.

| 발견된 현상 | M7에서 달라야 하는 것 |
|---|---|
| M1-C position portability | position encoding; 같은 mRoPE 계보만 사용 금지 |
| visual token selection/정보 유형 | vision encoder·tokenization·fusion |
| GUI tuning 특이성 | 같은 base 구조의 일반 VLM과 GUI-tuned VLM |
| architecture-general 현상 | language backbone까지 non-Qwen인 후보 최소 1개 |

확인된 후보 특성:

- Qwen3-VL: Qwen 계보
- UI-TARS-1.5-7B: Qwen2.5-VL, mRoPE
- OpenCUA-7B: Qwen2.5-VL 기반이지만 공식 구현은 1D RoPE
- InternVL3-8B: InternViT+MLP 구조지만 language model은 Qwen2 계열
- LLaVA-OneVision-7B: 다른 visual integration이지만 language model은 Qwen2 계열
- strict non-Qwen 후보: TBD, cache-resume와 V100 실행 가능성을 먼저 smoke

## 2. M0

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M0-01 | strict·operational 허용오차 | TBD | one-shot vs resume와 chunk/batch perturbation을 분리 |
| M0-02 | non-text sanity task와 표본 | TBD | icon/layout/grounding 각 최소 10개 |
| M0-03 | IMAGE base 최소 성능 | TBD | 공식 metric으로 task별 결정 |
| M0-04 | fp16 NaN 대응 | BLOCKED | mask·position·layer finite를 먼저 localize |

## 3. M1

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M1-01 | 저장 prefix의 정확한 token 경계 | TBD | system+image boundary, question KV 제외 |
| M1-02 | write/read token 순서 | TBD | generic image write → future question read |
| M1-03 | offset sweep | PROPOSED, 미확정 | 0/128/512/2048 |
| M1-04 | M1-F 2차 interaction 승격 기준 | TBD | 절대 task 차이+CI |

M1-A는 research finding이 아니라 serialization/load/injection gate로 고정한다. context binding 해석은
M1-B부터 허용한다.

## 4. M2-A

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M2A-01 | sparse baseline 목록과 timing | TBD | random/uniform + write-time SOTA + read-time S1/SOTA |
| M2A-02 | diagnostic sample ID와 task별 수 | TBD | task별 5–10장 사전 고정 |
| M2A-03 | answer-aware grouping | PROPOSED | 2×2 patch, ≤5장 region sensitivity |
| M2A-04 | search 계산 상한 | TBD | sample·budget별 forward 횟수 |
| M2A-05 | grounding selection score | TBD | click success와 정렬 audit |
| M2A-06 | extreme budget 발동 규칙 | DECIDED | 20% 무손실 시 diagnostic에 5%·10% 추가 |

S1을 storage compression으로 보고하려면 미래 질문 없이 write-time에 실행 가능해야 한다. 그렇지
않으면 read-time comparator로 라벨링한다.

## 5. M3

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M3-01 | T0–T4 label guide와 검수자 | TBD | 2인 검수+adjudication |
| M3-02 | source self-fidelity 통과 기준 | TBD | task equivalence와 M0 operational 범위 분리 |
| M3-03 | 질문 쌍 수 | TBD | 이미지당 유형별 최소 1쌍 |

## 6. M4

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M4-01 | PCTD 이미지·질문 수 | TBD | 4 domain×50 image, image당 최소 4 type |
| M4-02 | evidence annotation 단위 | TBD | bbox/UI element+허용 답 목록 |
| M4-03 | external confirmation dataset | TBD | 발견 유형별 2개, M7 전 고정 |

## 7. M2-B

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M2B-01 | family별 published baseline | TBD | SPARSE/QUANT/TRANSFORMED/HYBRID 각 최소 1 |
| M2B-02 | payload bit·group size·planner | TBD | payload bits 2/4/8, target 이하 최대 quality |
| M2B-03 | physical backend와 GPU | HQQ/V100 smoke only | 본측정 backend·GPU 별도 고정 |

`payload_bits`, smoke의 `budget_anchor_bits`, hybrid bit를 같은 `reference_bits` 어휘로 부르지 않는다.

## 8. M5

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M5-01 | 첫 trajectory dataset | TBD | 접근 smoke 후 Mind2Web/AndroidControl/GUIOdyssey 비교 |
| M5-02 | 시간 거리와 block 수 | TBD | 1/2/4 block+고정 offset control |
| M5-03 | `T_episode` 저장 범위 | TBD | q/action/outcome/trajectory를 분리 |

AndroidControl·GUIOdyssey는 공식 경로 존재만으로 READY로 보지 않는다. 실제 다운로드, 라이선스,
parser, evaluation 재현을 확인한다.

## 9. M6

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M6-01 | workload·회상 빈도·보존 기간 | TBD | low/medium/high 세 시나리오 |
| M6-02 | TTFT/throughput SLO | TBD | 시스템 선택을 바꾸는 최소 개선 |
| M6-03 | GPU·storage 가격과 기준일 | TBD | 측정량과 가격 가정을 분리 기록 |

## 10. M7

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M7-01 | 확인할 현상 하나 | TBD | 효과·귀속·실용 귀결이 가장 명확한 것 |
| M7-02 | confirmation sample size | TBD | discovery variance 기반 power |
| M7-03 | frozen exclusion·analysis revision | TBD | 결과를 보기 전 commit/hash 고정 |

## 11. 과거 설계 처리 기록

| 항목 | 상태 | 이유·복구 조건 |
|---|---|---|
| T5 | UNVERIFIED | 현재 파일에 정의가 없음; 원문 확보 전 임의 복원 금지 |
| text-only QA | CONDITIONAL | visual-specific 귀속 주장일 때 M2-A에 복구 |
| loyalty | ADOPTED_SECONDARY | correctness가 아닌 behavior drift canary |
| D2 signal-confound | CONDITIONAL | selector failure 발견 뒤 원인 진단 |
| D3 signal alignment | CONDITIONAL | answer-aware capacity가 있고 실제 selector만 실패할 때 |
| legacy d4 result | INVALID_ARCHIVED | full/compressed attention·position 경로 confound+NaN |

## 12. 결정 기록 형식

```text
ID:
date:
decision:
rationale:
evidence_available_at_decision:
changes_previous_decision: yes/no
impact_on_existing_results:
source_code_revision:
```
