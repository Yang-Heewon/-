# 사용자 결정 로그

이 파일에는 **코드가 대신 결정하면 안 되는 값**만 기록한다. 결정을 내리면 `TBD`를 실제 값으로
바꾸고 날짜·근거를 남긴다. 결과를 본 뒤 기준을 바꾼 경우 반드시 변경 이력을 보존한다.

## 전역 결정

| ID | 결정할 것 | 현재값 | 권장 시작점 | 결정 시점 | 바꾸면 달라지는 해석 |
|---|---|---|---|---|---|
| G01 | published SOTA 정확한 목록 | TBD | sparse 2개+family별 1개 | M2-A 구현 전 | baseline 강도와 계산량 |
| G02 | `T_visual` OCR/parser/caption 모델·prompt | TBD | OCR+bbox, generic dense caption, UI tree | M1-F 전 | TEXT baseline 공정성 |
| G03 | discovery 이미지 수와 ID | TBD | pilot variance 후 확정 | 본실험 전 | CI와 계산량 |
| G04 | confirmation 모델 2개 | Qwen3-VL-8B + GUI model TBD | Qwen3-VL-8B, UI-TARS/OpenCUA 중 1 | M7 freeze 전 | 일반화 범위 |
| G05 | 실용적 최소 효과 | TBD | task 의미로 결정 | M7 freeze 전 | E/S/N 판정 |
| G06 | 모든 모델·processor의 정확한 revision | TBD | 최초 smoke 전에 commit/hash 고정 | 각 단계 실행 전 | 재현성과 수치 차이 |

## M0

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M0-01 | CACHE-IDENTITY 수치 허용오차 | TBD | 반복 run noise floor 측정 후 고정 |
| M0-02 | non-text sanity task와 표본 | TBD | icon/layout/grounding 각 ≥10 |
| M0-03 | IMAGE base 성능 최소 기준 | TBD | 데이터셋 공식 metric 기준으로 별도 결정 |

## M1

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M1-01 | 저장할 prefix의 정확한 token 경계 | TBD | system+image boundary, question KV 제외 |
| M1-02 | write/read token 순서 | TBD | generic image write → future question read |
| M1-03 | offset sweep | TBD | 0/128/512/2048 token |
| M1-04 | M1-F 2차 interaction 승격 기준 | TBD | 절대 task 차이+CI로 결정 |

## M2-A

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M2A-01 | 전체 sparse SOTA | TBD | Random, spatial-uniform, S1, upstream adaptation 1–2개 |
| M2A-02 | diagnostic sample ID | TBD | task type별 5–10장 사전 고정 |
| M2A-03 | grouping | 2×2 patch | patch/region은 ≤5장 sensitivity |
| M2A-04 | search 계산 상한 | TBD | 표본·budget별 forward 횟수 고정 |
| M2A-05 | grounding selection score | TBD | click metric과 정렬 audit 후 결정 |

## M3

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M3-01 | T0–T4 label 규칙·검수자 | TBD | 2인 검수+불일치 adjudication |
| M3-02 | source self-fidelity 통과 기준 | TBD | IMAGE 대비 2%p equivalence 또는 task별 기준 |
| M3-03 | 질문 쌍 수 | TBD | 이미지당 유형별 최소 1쌍 |

## M4

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M4-01 | PCTD 이미지와 질문 수 | TBD | 4 domain×50 image, image당 ≥4 type |
| M4-02 | evidence annotation 단위 | TBD | bbox/UI element+허용 답 목록 |
| M4-03 | 외부 confirmation dataset routing | TBD | 현상별 2개 dataset을 M7 전 고정 |

## M2-B

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M2B-01 | family별 published baseline | TBD | SPARSE/QUANT/TRANSFORMED/HYBRID 각 ≥1 |
| M2B-02 | quant bit 후보 | TBD | 2/4/8 bit 중 B에 맞는 조합 |
| M2B-03 | physical 측정 backend/GPU | HQQ/V100 smoke only | 본측정 GPU를 별도 고정 |

## M5–M7

| ID | 결정할 것 | 현재값 | 권장 시작점 |
|---|---|---|---|
| M5-01 | trajectory dataset 우선순위 | TBD | Mind2Web → AndroidControl → OSWorld |
| M5-02 | 시간 거리와 block 수 | TBD | 1/2/4 block+고정 offset control |
| M6-01 | workload·회상 빈도 | TBD | low/medium/high 세 시나리오 |
| M6-02 | SLO와 가격 가정 | TBD | 측정량과 가격을 분리 기록 |
| M7-01 | 확인할 현상 하나 | TBD | M0–M6 후 선택 |
| M7-02 | confirmation sample size | TBD | pilot variance 기반 power |

## 결정 기록 형식

```text
ID:
date:
decision:
rationale:
evidence_available_at_decision:
changes_previous_decision: yes/no
impact_on_existing_results:
```
