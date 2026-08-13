# 시각 경험 재사용의 실패 지점을 찾는 연구 계획 (v6.1)

이 문서는 프로젝트의 최상위 연구·실행 계약이다. 단계별 명령과 결과 해석은
[`experiments/`](experiments/README.md)에 둔다.

## 0. 연구 목표

Phase 1의 목표는 새로운 압축 방법을 만드는 것이 아니다.

> VLM 에이전트가 과거의 시각 경험을 저장했다가 미래 질문·행동에 재사용할 때, 성능 손실이
> 처음 발생하는 단계와 조건을 통제 실험으로 찾는다.

먼저 다음 병목을 순서대로 분리한다.

1. 측정·cache-resume 구현이 틀렸는가?
2. 압축하지 않은 visual KV도 저장·이동·합성 과정에서 깨지는가?
3. 고정 byte 예산 안에 정답을 보존하는 subset 자체가 없는가?
4. 좋은 subset은 있지만 미래 질문을 모르는 selector가 찾지 못하는가?
5. 특정 정보 유형·시간 거리·trajectory에서만 실패하는가?
6. 정확도를 유지해도 실제 저장·지연·비용 이점이 없는가?

문제가 확인되고 독립 조건에서 재현된 뒤에만 Phase 2에서 해결 방법을 선택한다. 모델 weight
quantization은 연구 대상이 아니다. visual KV cache의 sparse, quantized, transformed, hybrid
표현만 다룬다.

---

## 1. 저장·재사용 조건

### 1.1 저장 payload

- `IMAGE`: 원본 이미지를 저장하고 read 시 다시 인코딩
- `FULL-KV`: 해석 가능한 canonical visual/prefix KV를 100% 저장·주입
- `T_visual`: OCR+bbox, dense description, UI tree·label·bbox
- `T_episode`: 과거 지시, 답·행동, 결과, trajectory
- `SPARSE-KV`: 일부 visual token만 저장
- `QUANT-KV`: visual K/V precision을 낮춰 저장
- `TRANSFORMED-KV`: merge·pooling 등으로 새로운 KV를 저장
- `HYBRID-KV`: sparse와 quantization/transform을 결합

`T_visual`과 `T_episode`를 하나의 TEXT 점수로 합치지 않는다. 과거 정답 `T_a`가 포함된 결과는
`answer_carryover=true`로 표시하고 미래 정보 보존의 증거로 사용하지 않는다.

### 1.2 압축 시점

selector는 미래 정보를 언제 사용하는지에 따라 분리한다.

| 분류 | 선택 시점 | 미래 질문 사용 | 저장 압축 baseline인가? |
|---|---|---:|---:|
| write-time/query-agnostic | memory 저장 전 | 불가 | 예 |
| write-time/source-aware | 과거 질문·행동 후 | 과거 정보만 | 예 |
| read-time/query-aware | 미래 질문 도착 후 | 가능 | 아니오: full 저장이 필요할 수 있음 |
| answer-aware probe | 진단 시 | 미래 질문+gold answer | 아니오: diagnostic only |

S1/SnapKV-style처럼 미래 질문을 이용하는 방법은 전체 split에서 비교할 수 있지만, persistent
storage 압축의 직접 baseline으로 해석하지 않는다. answer-aware probe는 항상 소규모 진단이다.

### 1.3 byte 예산

주 결과의 예산은 serialized visual-KV bytes 기준이다.

```text
B = bytes(compressed visual KV) / bytes(FULL-KV)
primary grid = 20%, 40%, 60%, 80% keep
FULL-KV = 100% uncompressed reference
```

sparse index, quantization scale·zero-point, position metadata를 포함한다. family별로 target 이하의
가장 유용한 구성을 선택하고 실제 사용 비율 `actual_bytes / target_bytes`도 보고한다. 순수 quant
family처럼 정확한 target을 채울 수 없는 경우 억지로 맞추지 않고 budget slack을 기록한다.

20%에서도 손실이 없으면 task별 5–10장 diagnostic subset에만 5%와 10%를 추가한다. 이를 실행하지
않고 최소 성공 예산 `B*`를 보고할 때는 `B* < 0.2 (left-censored)`라고 쓴다.

---

## 2. 측정 계약과 구현 gate

### 2.1 CACHE-IDENTITY

동일한 token sequence를 다음 두 경로로 계산했을 때 cache와 logits가 같은지 검사한다.

```text
one-shot prefill
vs
prefix prefill → cache serialize/load → 동일 suffix resume
```

이는 연구 결과가 아니라 M0의 구현 gate다. 실패하면 후속 결과를 해석하지 않는다.

허용오차는 두 층으로 분리한다.

- `strict identity`: 동일 backend·batch에서 one-shot과 cache-resume 비교
- `operational equivalence`: 합법적인 chunked prefill·batch 구성 변화에서의 수치 범위

동일 입력 반복 실행만으로 noise floor를 정하지 않는다. threshold는 후속 효과를 보기 전에
config에 고정한다.

### 2.2 STORED-FULL FIDELITY

- `M1-A`: 같은 sequence·offset에서 IMAGE-REENCODE와 STORED-FULL 비교
- `M1-B`: write/read 주변 문맥 변경
- `M1-C`: offset·position 변경

M1-A는 serialization→disk→load→injection 전체 경로의 gate다. 동일 prefix와 offset에서 실패하면
먼저 payload 누락·dtype·position·주입 구현을 의심한다. “KV가 write 문맥에 결박됐다”는 연구
해석은 CACHE-IDENTITY와 M1-A가 통과하고 실제 문맥을 바꾼 M1-B부터 허용한다.

### 2.3 finite gate

NaN/Inf가 하나라도 있는 run은 부분 평균을 내지 않는다. 현재 Qwen2.5-VL fp16 masked path에서
layer-27 fp32 패치 후에도 NaN이 재현됐으므로, M0 시작 전 다음을 분리한다.

- 2D vs 4D attention 경로
- mask된 행의 유효 key 존재 여부
- layer별 Q/K/V·attention probability·hidden-state finite
- fp16 overflow와 position/mask artifact

원인을 확인하기 전에는 fp32 layer를 임의로 늘려 결과만 통과시키지 않는다.

---

## 3. 실행 순서

| 순서 | 단계 | 핵심 질문 | 다음 단계 gate | 문서 |
|---:|---|---|---|---|
| P0 | 저장소·계약 | revision과 결과 분류를 재현할 수 있는가? | Git/config/manifest validator 준비 | [`experiments/README.md`](experiments/README.md) |
| 0 | M0 측정 | mask/cache/finite가 맞는가? | CACHE-IDENTITY·100% keep·finite 통과 | [`00_M0_MEASUREMENT.md`](experiments/00_M0_MEASUREMENT.md) |
| 1 | M1 저장·재사용 | 무압축 payload가 어디서 깨지는가? | canonical STORED-FULL 조건 확보 | [`01_M1_STORAGE_REUSE.md`](experiments/01_M1_STORAGE_REUSE.md) |
| 2 | M2-A sparse | 고정 예산에서 capacity와 selection 중 무엇이 문제인가? | task metric과 probe 안정성 확보 | [`02_M2A_FIXED_BUDGET.md`](experiments/02_M2A_FIXED_BUDGET.md) |
| 3 | M3 전이 | 과거 relevance가 미래 질문으로 전이되는가? | relevance와 estimator 원인 분리 | [`03_M3_TRANSFER.md`](experiments/03_M3_TRANSFER.md) |
| 4 | M4 정보 유형 | 어떤 시각 정보가 선택적으로 사라지는가? | 동일 이미지 내 유형 효과 확인 | [`04_M4_INFORMATION_TYPES.md`](experiments/04_M4_INFORMATION_TYPES.md) |
| 5 | M2-B family | sparse/quant/transform/hybrid가 무엇을 다르게 잃는가? | quality와 physical 주장 분리 | [`05_M2B_FAMILY_STRESS.md`](experiments/05_M2B_FAMILY_STRESS.md) |
| 6 | M5 trajectory | 시간·상태 변화·다중 memory에서만 실패하는가? | action/episode metric 확보 | [`06_M5_TRAJECTORY.md`](experiments/06_M5_TRAJECTORY.md) |
| 7 | M6 system | 정확한 표현이 비용상 쓸 이유가 있는가? | accuracy×cost 비지배 영역 확인 | [`07_M6_SYSTEM.md`](experiments/07_M6_SYSTEM.md) |
| 8 | M7 confirmation | 동결한 문제가 새 데이터·모델에서도 재현되는가? | 사전 동결 기준 충족 | [`08_M7_CONFIRMATION.md`](experiments/08_M7_CONFIRMATION.md) |

앞선 병목이 실패하면 뒤 실험을 실행해도 원인으로 해석하지 않는다.

---

## 4. M2-A의 두 평가 track

### Track 1: 전체 discovery split

- FULL-KV
- random
- spatial-uniform
- published write-time/query-agnostic baseline
- S1을 포함한 published/adapted read-time/query-aware comparator

모든 방법은 공식 task metric으로 평가한다. 방법 이름과 함께 압축 시점, 미래 질문 사용 여부,
upstream-runtime인지 VLM adaptation인지 기록한다.

### Track 2: 소규모 원인 진단

- target query-aware probe
- target answer-aware probe
- search audit: leave-group-out, backward elimination, pair removal, random/beam
- 필요할 때만 5%·10% extreme budget

answer-aware probe는 고정 B에서 gold-answer `Δlogp`로 subset을 선택한 뒤 gold answer 없이 새 답을
생성하여 공식 metric으로 평가한다. 다음만 구분한다.

- 같은 예산에 좋은 subset 자체가 없는가?
- 좋은 subset은 있지만 selector가 찾지 못하는가?

---

## 5. 질문 전이와 조건부 신호 진단

M3의 질문 쌍은 T0–T4로 층화한다.

| 유형 | 변화 |
|---|---|
| T0 | 같은 질문 반복 |
| T1 | paraphrase, 같은 답·근거 |
| T2 | 다른 질문, 같은 근거 |
| T3 | 같은 이미지의 다른 근거 |
| T4 | semantic/OCR ↔ layout/grounding |

현재 보존된 문서에는 T5 정의가 없으므로 임의로 복원하지 않는다. 과거 정의를 찾으면 G10에서
T0–T4 흡수 여부 또는 descope를 기록한다.

과거 D2/D3 신호 분석은 기본 선행 실험으로 되돌리지 않는다. M2-A/M3에서 selector failure가
관측된 경우에만 다음 조건부 진단을 실행한다.

- D2-style audit: 길이·sink·position·token 수 통제 후 신호 분포 재측정
- D3-style alignment: 값싼 selector signal과 answer-aware importance의 정렬 측정

이렇게 해야 특정 방법을 먼저 정해 놓지 않고 실패를 발견한 뒤 필요한 원인 분석만 수행할 수 있다.

---

## 6. 공통 평가

- 주 평가는 데이터셋 공식 task metric이다.
- teacher-forced log-prob는 selection·기제 진단에만 사용한다.
- IMAGE가 맞힌 표본에서의 conditional retention을 전체 점수와 함께 보고한다.
- 여러 질문이 있는 이미지는 이미지 하나를 통계 단위로 사용한다.
- discovery와 confirmation은 이미지/episode 단위로 분리한다.
- 이미지 단위 bootstrap 95% CI와 worst-group을 보고한다.
- full-output과 compressed-output의 normalized agreement를 `loyalty` 보조 metric으로 기록한다.
- action/trajectory에서는 답 문자열보다 action type·target과 최초 divergence step을 우선한다.

text-only QA는 “visual KV가 text KV보다 특별히 취약하다”는 모달리티 귀속 주장을 할 때 M2-A에
작은 통제군으로 포함한다. visual memory 내부의 실패만 주장할 때는 필수 대조군으로 확대하지 않는다.

공통 schema와 상태 규칙은 [`SHARED_PROTOCOL.md`](experiments/SHARED_PROTOCOL.md)를 따른다.

---

## 7. 데이터

### 원인 분리 중심

- PCTD: 같은 이미지에 OCR/semantic/layout/grounding/icon/count 질문을 부착할 진단 세트
- ScreenQA/ComplexQA: 같은 모바일 화면의 여러 질문과 UI bounds
- DocVQA: 같은 문서의 여러 질문

### 외적 타당성

- 문서: DocVQA, InfographicVQA, MP-DocVQA
- 차트: ChartQA/ChartQAPro
- 장면 문자: TextVQA, OCRBench v2
- 자연 이미지: GQA, RefCOCO 계열
- GUI·웹: ScreenQA, ScreenSpot-v2/Pro, VisualWebBench

### 시간·행동

- Multimodal-Mind2Web
- AndroidControl
- GUIOdyssey
- OSWorld/OSWorld-Verified

AndroidControl과 GUIOdyssey는 공식 공개 경로가 생겼지만, M5 확정 전 로컬 환경에서 실제 다운로드,
라이선스, screenshot/action schema, evaluation 재현 가능성을 다시 smoke한다. 접근 확인 전에는
우선순위만으로 dataset을 확정하지 않는다.

---

## 8. 모델과 일반화 범위

### Discovery

- Qwen2.5-VL-7B
- V100 4×32GB, fp16/eager

### Confirmation 후보의 정확한 의미

| 후보 | Qwen 계보 여부 | 위치·멀티모달 차이 | 사용할 수 있는 일반화 주장 |
|---|---:|---|---|
| Qwen3-VL-8B | 예 | 세대·visual integration 변화 | Qwen 세대 간 재현 |
| UI-TARS-1.5-7B | 예, Qwen2.5-VL | GUI fine-tuning 차이, mRoPE 유지 | 같은 구조의 domain tuning 재현 |
| OpenCUA-7B | 예, Qwen2.5-VL 기반 | 공식 구현은 1D RoPE로 변경 | position encoding 변화에 대한 재현 |
| InternVL3-8B | LLM은 Qwen2 계열 | InternViT+MLP, 다른 visual tokenization | multimodal integration 변화 |
| LLaVA-OneVision-7B | LLM은 Qwen2 계열 | SigLIP/projector 계열 | multimodal integration 변화 |
| strict non-Qwen 후보 | TBD | LLM·position·fusion까지 독립 | 전체 architecture 일반화 |

따라서 InternVL/LLaVA를 단순히 “비-Qwen”이라고 부르지 않는다. G04에서는 발견된 현상에 맞춰
독립성 기준을 먼저 고정한다.

- M1-C position portability가 핵심이면 position encoding이 다른 모델을 포함한다.
- visual integration이 핵심이면 vision encoder/fusion이 다른 모델을 포함한다.
- architecture-general 주장을 하려면 language backbone까지 non-Qwen인 후보를 최소 1개 포함한다.

M7 모델은 최소 두 confirmation 조건을 사용하되, 같은 Qwen/mRoPE 계보 두 개만으로 전체
architecture 일반화를 주장하지 않는다. 세부 routing은
[`MODEL_MATRIX.md`](experiments/MODEL_MATRIX.md)에 둔다.

---

## 9. 결과와 revision 관리

```text
results/smoke/         실행 경로·측정 gate
results/discovery/     가설 탐색
results/confirmation/  동결한 확인 실험
archive/results/       무효화된 legacy 결과와 실패 증거
```

`archive/results/`는 네 번째 run kind가 아니다. 분석 입력에서 제외된 역사적 artifact다.

모든 본실험은 다음을 고정한다.

- source code Git commit
- config hash와 복사본
- manifest hash와 복사본
- model·processor revision
- dataset revision
- seed와 표본 선택 규칙
- analysis code revision

---

## 10. 현재 blocker와 구현 상태

| 단계 | 상태 | 현재 가능한 것 | 본실험 전 필요한 것 |
|---|---|---|---|
| P0 | PARTIAL | 루트 Git과 baseline source pin | config/resource validator 강화 |
| M0 | PARTIAL | NaN 원인 확정·수정(QK pre-scale), m0_measurement runner 8검사, 합성 non-text sanity 40표본 | M0-01 허용오차 고정, 판정 report |
| M1 | PLANNED | condition registry | cache serialization/resume runner |
| M2-A | PARTIAL | random/S1와 spatial-uniform primitive | full runner 연결, task metric, budget grid, probes |
| M3 | PARTIAL | 무효화된 legacy K×K 코드 | T0–T4 manifest와 교차평가 runner |
| M4 | PLANNED | 데이터 후보 | PCTD schema·annotation·runner |
| M2-B | PARTIAL | 한 문서 quality smoke | target-byte planner와 task metric |
| M5 | PLANNED | HERMES source pin | dataset access smoke·trajectory adapter |
| M6 | PARTIAL | HQQ physical path smoke | workload benchmark와 실제 serializer |
| M7 | PLANNED | 없음 | hypothesis/model/data/analysis freeze |

필요한 manifest 9종은 아직 없다. 빈 파일로 validator만 통과시키지 않고, 앞 단계 gate가 열린
순서대로 실제 ID와 revision을 담아 생성한다.

---

## 11. Phase 1 종료 조건

다음 문장을 데이터로 채울 수 있을 때만 문제의식이 확립된다.

> **[재사용 조건]에서 [기준 표현/기존 정책]은 [정보·태스크 그룹]의 성능을 [효과 크기]만큼
> 잃는다. [대조 개입]으로 이 손실을 [병목]에 귀속했고, [독립 모델·데이터]에서 재현했다.
> 그 결과 [정확도·지연·비용상의 실제 선택]이 달라진다.**

다음이면 Phase 2로 넘어가지 않는다.

- M0 측정 계약이 불완전함
- discovery 효과가 confirmation에서 사라짐
- log-prob만 변하고 공식 task metric은 유지됨
- 어느 병목에도 귀속할 수 없음
- 시스템 선택을 바꾸지 않는 미미한 효과임

---

## 12. 파일 트리

```text
VLM/
├── PLAN.md
├── BASELINES.md
├── docs/
│   └── HARDWARE_CONSTRAINTS.md
├── experiments/
│   ├── README.md
│   ├── SHARED_PROTOCOL.md
│   ├── DECISIONS.md
│   ├── MODEL_MATRIX.md
│   ├── 00_M0_MEASUREMENT.md ... 08_M7_CONFIRMATION.md
│   ├── configs/
│   └── manifests/
├── vlm_diagnosis/
│   ├── core/
│   ├── exps/
│   └── scripts/
├── results/
│   ├── smoke/
│   ├── discovery/
│   └── confirmation/
├── archive/
│   ├── legacy-design/
│   └── results/
└── third_party/
```

실행 순서와 판정이 과거 문서와 충돌하면 이 문서와 `experiments/`가 우선한다. 과거 설계에서
제외한 항목은 삭제로 숨기지 않고 [`DECISIONS.md`](experiments/DECISIONS.md)에 채택·조건부·descope
상태를 기록한다.
