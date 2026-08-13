# 시각 경험 재사용 실패를 발견하기 위한 연구 계획 (v6.0)

이 문서는 프로젝트의 **최상위 의사결정 문서**다. 세부 실행법은
[`experiments/`](experiments/README.md)에 단계별로 분리한다.

Phase 1의 목표는 새 압축 방법을 만드는 것이 아니다.

> VLM 에이전트가 과거의 시각 경험을 미래 태스크에 재사용할 때, 성능 손실이 처음 발생하는
> 단계와 그 조건을 통제 실험으로 찾는다.

실험 결과가 문제를 확정한 뒤에만 Phase 2에서 해결 방법을 선택한다.

---

## 1. 연구에서 고정한 것

### 1.1 비교 대상

```yaml
전체 평가: FULL-KV + random + spatial-uniform + published SOTA
소규모 원인 진단: answer-aware/query-aware probe
시각 표현 비교: T_visual
실제 에이전트 기록 비교: T_episode
```

- `IMAGE`: 원본 이미지를 저장하고 read 시 다시 인코딩한다.
- `FULL-KV`: visual KV를 압축하지 않고 저장·주입한다.
- `T_visual`: OCR, dense description, UI 구조와 좌표를 저장한다.
- `T_episode`: 과거 지시, 답·행동, 결과, trajectory를 저장한다.
- `SPARSE-KV`: 일부 visual token만 저장한다.
- `QUANT-KV`: visual K/V의 precision을 낮춘다.
- `TRANSFORMED-KV`: merge·pooling 등으로 새로운 KV 표현을 저장한다.
- `HYBRID-KV`: sparsification과 quantization/transform을 결합한다.

`T_visual`과 `T_episode`는 하나의 `TEXT` 점수로 합치지 않는다. `T_episode`에 과거 정답이
포함된 T0 결과는 `answer-carryover`로 표시한다.

### 1.2 압축 예산

모든 압축 실험의 예산은 serialized visual-KV bytes를 기준으로 한다.

```text
B = bytes(compressed visual KV) / bytes(FULL-KV)
```

주 압축 설정은 다음 네 개다.

```text
20% keep / 40% keep / 60% keep / 80% keep
```

`100%`는 다섯 번째 압축 설정이 아니라 무압축 `FULL-KV` reference다. sparse index,
quantization scale·zero-point, mRoPE position metadata도 byte 계산에 포함한다.

### 1.3 두 종류의 동등성

- `CACHE-IDENTITY`: 같은 token sequence를 일반 prefill한 결과와 cache-resume한 결과가
  수치적으로 같은가? 이는 M0의 **구현 gate**다.
- `STORED-FULL FIDELITY`: write-time full KV를 read 문맥에서 재사용했을 때
  `IMAGE-REENCODE`를 재현하는가? 이는 M1의 **연구 결과**다.

`CACHE-IDENTITY` 실패 시 후속 결과를 해석하지 않는다. `STORED-FULL FIDELITY` 실패는
중단 사유가 아니라 “KV가 독립적인 기억 payload가 아닐 수 있다”는 문제 후보다.

### 1.4 Answer-aware probe

고정 예산 `B`에서 answer-aware probe는 다음 두 단계를 수행한다.

1. 미래 질문과 gold answer의 loss를 이용해 `B` 안에 남길 KV subset을 선택한다.
2. 선택된 subset만 주입해 답을 새로 생성하고 공식 task metric으로 평가한다.

gold answer는 모델 prompt에 넣지 않는다. 이 probe는 배포 가능한 baseline도, 조합 최적해를
보장하는 oracle도 아니다. 전체 데이터가 아니라 task별 5–10장에만 사용해 다음을 구분한다.

- 같은 예산의 좋은 subset 자체가 없는가?
- 좋은 subset은 있지만 기존 selector가 찾지 못하는가?

---

## 2. 발견할 수 있는 병목

| 병목 | 질문 | 핵심 대조 |
|---|---|---|
| 측정 오류 | 마스크와 cache 주입이 의도대로 작동하는가? | CACHE-IDENTITY, full mask, 100% keep |
| 표현 | 저장 full KV가 이미지를 재현하는가? | IMAGE-REENCODE vs STORED-FULL |
| 이식 | 문맥·offset이 바뀌면 full KV가 깨지는가? | 동일 위치 vs context/offset shift |
| sparse capacity | 고정 예산 안에 좋은 subset이 존재하는가? | SOTA vs answer-aware probe |
| 미래 relevance | 과거에 중요한 subset이 미래에도 유효한가? | source/target subset 교차평가 |
| 신호 추정 | subset은 유효하지만 실제 selector가 못 찾는가? | source probe vs published SOTA |
| 정보 유형 | 특정 정보만 선택적으로 사라지는가? | OCR/semantic/layout/grounding/icon/count |
| 시간·합성 | 궤적과 다중 기억에서만 실패하는가? | single vs multi-frame/block |
| 경제성 | 정확해도 저장·지연 비용상 쓸 이유가 없는가? | IMAGE/T_visual/T_episode/KV 파레토 |

항상 앞선 병목부터 검사한다. 예를 들어 full KV가 offset 이동에서 이미 깨지면, 그 조건의
sparse 성능을 selector 실패로 해석하지 않는다.

---

## 3. 실행 순서와 gate

| 순서 | 단계 | 핵심 산출물 | 다음 단계 gate | 문서 |
|---:|---|---|---|---|
| 0 | M0 측정 계약 | measurement report | CACHE-IDENTITY·finite·mask sanity 통과 | [`00_M0_MEASUREMENT.md`](experiments/00_M0_MEASUREMENT.md) |
| 1 | M1 저장·재사용 경계 | payload/position loss map | 해석 가능한 canonical KV 조건 확보 | [`01_M1_STORAGE_REUSE.md`](experiments/01_M1_STORAGE_REUSE.md) |
| 2 | M2-A 고정 예산 sparse | FULL/random/uniform/SOTA curve + 소표본 원인 진단 | task metric과 probe 안정성 확인 | [`02_M2A_FIXED_BUDGET.md`](experiments/02_M2A_FIXED_BUDGET.md) |
| 3 | M3 미래 질문 전이 | T0–T4 transfer attribution | relevance와 estimator 원인 분리 | [`03_M3_TRANSFER.md`](experiments/03_M3_TRANSFER.md) |
| 4 | M4 정보 유형 | information-loss map | 같은 이미지 통제에서 정보 유형 확인 | [`04_M4_INFORMATION_TYPES.md`](experiments/04_M4_INFORMATION_TYPES.md) |
| 5 | M2-B family stress | byte-matched family damage map | quality 결과와 physical 주장을 분리 | [`05_M2B_FAMILY_STRESS.md`](experiments/05_M2B_FAMILY_STRESS.md) |
| 6 | M5 궤적·다중 기억 | temporal/composition map | 단일 이미지 재사용이 성립할 때만 실행 | [`06_M5_TRAJECTORY.md`](experiments/06_M5_TRAJECTORY.md) |
| 7 | M6 시스템 성립 영역 | accuracy×cost Pareto | 현실 workload에서 비지배 영역 확인 | [`07_M6_SYSTEM.md`](experiments/07_M6_SYSTEM.md) |
| 8 | M7 확인 실험 | confirmation report | 새 표본과 2개 모델 계열에서 재현 | [`08_M7_CONFIRMATION.md`](experiments/08_M7_CONFIRMATION.md) |

`M2-B`는 이름상 M2의 일부지만 실행은 M4 뒤다. 먼저 실제 sparse 실패와 정보 유형을 확인한
뒤 family별 손상 방식을 비교하기 위해서다.

---

## 4. 공통 실험 규칙

- 주 평가는 데이터셋 공식 task metric이다.
- teacher-forced log-prob는 selection·기제 진단에만 사용한다.
- IMAGE가 원래 틀린 표본과 IMAGE가 맞힌 표본의 retention을 분리한다.
- 질문 여러 개가 있는 이미지는 이미지 하나를 통계 단위로 사용한다.
- 탐색 split과 confirmation split은 이미지 단위로 분리한다.
- NaN/Inf가 하나라도 있는 run은 부분 평균을 내지 않는다.
- quality simulation으로 실제 GPU memory·latency 절감을 주장하지 않는다.
- 모든 결과에 model revision, prompt, processor, dtype, seed, token span, position ID,
  keep budget, 실제 bytes를 기록한다.

공통 schema와 상태 규칙은 [`experiments/SHARED_PROTOCOL.md`](experiments/SHARED_PROTOCOL.md)를
따른다.

---

## 5. 데이터 사용 원칙

### 원인 분리

- PCTD: 같은 이미지에 OCR/semantic/layout/grounding/icon/count 질문을 함께 부착할 신규
  진단 세트
- ScreenQA/ComplexQA: 같은 모바일 화면의 여러 질문과 UI element bounds
- DocVQA: 같은 문서의 여러 질문. evidence bbox 부재는 OCR matching proxy로 처리

### 외적 타당성

- 문서: DocVQA, InfographicVQA, MP-DocVQA
- 차트: ChartQA/ChartQAPro
- 장면 문자: TextVQA, OCRBench v2
- 자연 이미지: GQA, RefCOCO/RefCOCO+/RefCOCOg
- GUI·웹: ScreenQA, ScreenSpot-v2/Pro, VisualWebBench

### 시간·행동

- Multimodal-Mind2Web
- AndroidControl
- GUIOdyssey
- OSWorld/OSWorld-Verified

모든 데이터셋을 모든 단계에 돌리지 않는다. discovery는 PCTD+ScreenQA+DocVQA를 중심으로
하고, 발견된 정보 유형에 맞는 외부 데이터셋을 M7 전에 고정한다.

---

## 6. 모델과 하드웨어

- Discovery: Qwen2.5-VL-7B
- Confirmation 1: Qwen3-VL-8B
- Confirmation 2: UI-TARS-1.5-7B 또는 OpenCUA-7B
- 기본 하드웨어: V100 4×32GB, fp16
- A100/bf16: fp16 수치 문제가 효과 크기를 바꿀 가능성이 있을 때만 확인

모델 전체 weight quantization은 연구 대상이 아니다. KV cache의 sparse/quant/transform/
hybrid 표현만 비교한다.

---

## 7. 사용자가 결정해야 하는 항목

세부 목록과 결정 시점은 [`experiments/DECISIONS.md`](experiments/DECISIONS.md)에 기록한다.
현재 핵심 미결정은 다음이다.

1. 전체 평가에 포함할 published SOTA의 정확한 목록
2. `T_visual`을 생성할 OCR·caption·UI parser와 prompt
3. discovery/diagnostic/confirmation의 정확한 이미지 ID와 표본 수
4. M1에서 저장할 prefix 경계와 write/read token 순서
5. grounding answer-aware selection score
6. M6 workload, SLO, 회상 빈도와 가격 가정

결정값을 코드에 하드코딩하지 않고 [`experiments/configs/`](experiments/configs/README.md)에
기록한다. `null` 또는 `TBD`가 남은 필수 항목은 본실험을 시작하지 않는다.

```bash
python -m vlm_diagnosis.scripts.validate_experiment_configs --allow-unresolved
```

이 명령으로 단계별 미결정 경로를 확인한다. 본실험 직전에는 `--allow-unresolved` 없이
실행해 종료 코드 0을 확인한다.

---

## 8. 현재 구현 상태

| 단계 | 상태 | 현재 가능한 것 | 아직 없는 것 |
|---|---|---|---|
| M0 | PARTIAL | 합성 OCR 4D mask/V2 sanity | non-text sanity, CACHE-IDENTITY 본검사 |
| M1 | PLANNED | condition registry | full-KV extraction/resume canonical runner |
| M2-A | PARTIAL | random/S1 logical masking 구성요소 | 20/40/60/80 grid, task metric, answer-aware audit |
| M3 | PARTIAL | legacy D4 K×K logp runner | T0–T4 labels, source/target 교차평가 |
| M4 | PLANNED | 일부 데이터 후보 | PCTD annotation과 통합 runner |
| M2-B | PARTIAL | 1문서 native 4-bit-ratio quality smoke | 20/40/60/80 byte planner와 task metric |
| M5 | PLANNED (upstream pinned) | HERMES source 고정 | trajectory adapter와 평가 통합 |
| M6 | PARTIAL | HQQ physical cache smoke | 실제 memory/latency/workload benchmark |
| M7 | PLANNED | 없음 | hypothesis freeze와 confirmation runner |

현재 smoke 결과는 실행 경로 확인일 뿐 발견이나 논문 결과가 아니다.

---

## 9. Phase 1 종료 조건

다음 문장을 데이터로 채울 수 있을 때만 문제의식이 확립된다.

> **[재사용 조건]에서 [기준 표현/기존 정책]은 [정보·태스크 그룹]의 성능을
> [효과 크기]만큼 잃는다. [대조 개입]으로 이 손실을 [병목]에 귀속했고,
> [확인 모델·데이터]에서 재현했다. 그 결과 [정확도·지연·비용상의 실제 귀결]이 생긴다.**

다음이면 Phase 2로 넘어가지 않는다.

- M0 측정 계약이 불완전하다.
- 탐색 효과가 confirmation에서 사라진다.
- log-prob만 변하고 실제 task metric은 변하지 않는다.
- 어느 병목에도 귀속할 수 없다.
- 문제는 보이지만 시스템 선택을 바꾸지 않는다.

---

## 10. 문서·파일 트리

```text
VLM/
├── PLAN.md                         # 이 문서: 목표, 순서, gate
├── BASELINES.md                    # 외부 baseline 구현·라이선스·주의점
├── experiments/
│   ├── README.md                   # 전체 실행 인덱스
│   ├── SHARED_PROTOCOL.md          # 공통 metric·schema·상태 규칙
│   ├── DECISIONS.md                # 사용자가 확정할 값과 결정 로그
│   ├── 00_M0_MEASUREMENT.md
│   ├── 01_M1_STORAGE_REUSE.md
│   ├── 02_M2A_FIXED_BUDGET.md
│   ├── 03_M3_TRANSFER.md
│   ├── 04_M4_INFORMATION_TYPES.md
│   ├── 05_M2B_FAMILY_STRESS.md
│   ├── 06_M5_TRAJECTORY.md
│   ├── 07_M6_SYSTEM.md
│   ├── 08_M7_CONFIRMATION.md
│   ├── configs/                    # 단계별 사람이 읽는 실행 계약
│   └── manifests/                  # 실제 image/question ID 목록
├── vlm_diagnosis/
│   ├── core/                       # 공통 cache/mask/storage 로직
│   ├── exps/                       # 본실험 runner
│   └── scripts/                    # smoke·준비·검증 도구
├── results/
│   ├── smoke/                      # 실행 경로 확인
│   ├── discovery/                  # 가설 탐색
│   └── confirmation/               # 동결된 확인 실험
└── third_party/                    # 고정한 외부 baseline source
```

기존 `DIAGNOSIS_DESIGN.md`, `PRIOR_WORK.md`, `MOTIVATION_ANALYSIS.md`는 역사적 설계와
선행연구 기록이다. 실행 순서와 판정이 충돌하면 이 문서와 `experiments/` 문서가 우선한다.
