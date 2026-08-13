# M2-B — KV compression family별 손상 방식

**상태:** `PARTIAL`  
**질문:** 같은 serialized bytes에서 SPARSE/QUANT/TRANSFORMED/HYBRID는 무엇을 다르게 잃는가?
**실행 계약:** [configs/m2b.yaml](configs/m2b.yaml)

## 1. 실행 시점

M2-A와 M4 뒤에 실행한다. 먼저 실제 sparse 실패와 정보 유형을 알아야 family 순위를
“방법 경연”이 아니라 손상 방식으로 해석할 수 있다.

## 2. 비교 family

| family | 최소 baseline | 현재 상태 |
|---|---|---|
| FULL | FP16 visual KV | logical reference |
| SPARSE | random, S1/SnapKV-style, published adapter | 일부 구현 |
| QUANT | KIVI-style K/V quantization | fake-quant 구현 |
| TRANSFORMED | merge-on-evict | kvpress adaptation 구현 |
| HYBRID | sparse+quant | 로컬 조합 구현 |

모델 weight quantization은 포함하지 않는다.

## 3. 예산

모든 family를 다음 target byte에서 비교한다.

```text
20% keep / 40% keep / 60% keep / 80% keep
FULL = 100%
```

각 B에서 payload bit 수, sparse token 수, group metadata를 포함해 target bytes 이하가 되는
구성을 planner가 계산한다. exact target을 채우지 못하면 `budget_utilization`과 slack을 기록한다.
all-token 4-bit의 자연 비율은 grid와 별도의 calibration point다.

## 4. 전체 평가와 physical 측정 분리

- M2-B: logical mask/fake quant의 **task quality**
- M6: 실제 cache 객체의 **memory/latency/throughput**

fake quant가 좋은 task score를 냈다고 GPU memory가 줄었다고 주장하지 않는다.

## 5. 사용자가 수정·결정할 부분

| 결정 ID | 결정 | 판단 기준 |
|---|---|---|
| M2B-01 | family별 published baseline | 각 mechanism을 대표하고 라이선스·revision 고정 |
| M2B-02 | payload bit·group size·planner | target 이하에서 품질을 최대화하는 규칙 |
| G03 | 본평가 표본 수 | M2-A/M4 현상 범위와 계산량 |
| 분석 선택 | 핵심 task/budget | 결과 전에는 전체 grid, 헤드라인은 사전 기준으로 선택 |

## 6. 현재 실행 가능한 smoke

```bash
python -m vlm_diagnosis.exps.m2_family_baselines \
  --limit 1 \
  --reference-bits 4 \
  --hybrid-bits 8 \
  --device cuda:0
```

현재 runner는 all-token 4-bit가 차지하는 약 28% bytes를 기준으로 1개 질문의 teacher-forced
logp를 측정한다. 20/40/60/80 grid와 task metric이 없으므로 smoke다.

어휘를 다음처럼 분리한다.

- `payload_bits`: quantized K/V 자체의 2/4/8-bit precision
- `budget_anchor_bits`: 현재 smoke가 target bytes를 만드는 2/4-bit 값
- `hybrid_payload_bits`: sparse subset에 적용할 bit 값

현재 smoke CLI는 `budget_anchor_bits` 역할의 `--reference-bits`에 2/4만 허용하고 8-bit는 hybrid에
허용한다. 이 제한을 QUANT family 전체의 8-bit 미지원으로 해석하지 않는다.

## 7. 목표 runner

```bash
python -m vlm_diagnosis.exps.m2b_family_stress \
  --config experiments/configs/m2b.yaml \
  --manifest experiments/manifests/m2a_full.jsonl
```

필요 구현:

- arbitrary target ratio byte planner
- B별 feasible bit/token 조합 기록
- target bytes, actual bytes, utilization, slack 기록
- task metric generation
- family×task×budget interaction report

## 8. 판정

| 결과 | 이 단계에서 말할 수 있는 것 | 아직 말할 수 없는 것 |
|---|---|---|
| QUANT>SPARSE | 값 perturbation이 token deletion보다 덜 해로움 | 미래 질문에는 항상 quant가 우월 |
| SPARSE>QUANT | 일부 정밀 보존이 전체 저정밀보다 유리 | 특정 selector가 relevance 해결 |
| HYBRID>각 단일 family | 두 손상 방식이 상보적일 가능성 | 새 hybrid 방법 확정 |
| MERGE>SPARSE | 삭제 value 결합이 유효할 가능성 | 실제 latency도 유리 |
| 차이가 T3/T4에만 발생 | family×정보/질문 상호작용 | 원인 확인 없이 relevance 문제 단정 |

## 9. 완료 조건

- 모든 family의 estimated bytes와 overhead 기록
- 20/40/60/80에서 feasible하지 않은 구성은 억지로 실행하지 않고 이유 기록
- task metric과 logp 분리
- quality simulation과 physical result 분리
- family 우승자를 Phase 1의 방법으로 선택하지 않음
