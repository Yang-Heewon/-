# 공통 실험 프로토콜

모든 단계는 이 문서의 정의를 공유한다. 단계 문서가 다르게 정의하면 이유와 적용 범위를 해당
문서와 config에 모두 적는다.

## 1. 기준 표현

- `NO-MEM`: memory 없이 미래 질문 수행
- `IMAGE-REENCODE`: 원본 이미지를 read 시 다시 인코딩
- `FULL-KV`: M1에서 확정한 canonical visual/prefix KV 100% 저장·주입
- `T_visual = T_o + T_d + T_u`
  - `T_o`: OCR+bbox
  - `T_d`: dense visual description
  - `T_u`: UI element/tree+label+bbox
- `T_episode = T_q + T_a + T_out + T_traj`
  - `T_q`: 과거 지시
  - `T_a`: 과거 답 또는 행동
  - `T_out`: 행동 결과·이후 상태
  - `T_traj`: 중간 행동과 관찰 순서

`T_a`가 포함된 결과에는 `answer_carryover=true`를 기록한다.

## 2. selector 시점과 정보 접근

모든 compression condition은 다음 필드를 가진다.

```yaml
selection_timing: write_time|read_time|diagnostic
future_question_available: bool
gold_answer_available_for_selection: bool
full_kv_must_be_stored_until_read: bool
implementation_scope: upstream_runtime|vlm_adaptation|quality_simulation
```

- persistent storage baseline은 `selection_timing=write_time`이어야 한다.
- 미래 질문으로 full KV를 다시 고르는 read-time 방법은 성능 comparator일 수 있지만 저장 압축으로
  세지 않는다.
- gold answer를 사용하는 방법은 항상 `diagnostic`이다.

## 3. 예산과 byte matching

```text
keep_ratio = estimated_serialized_bytes(memory) / estimated_serialized_bytes(FULL-KV)
primary budgets = 0.2, 0.4, 0.6, 0.8
FULL reference = 1.0
```

- `0.2`는 20% 제거가 아니라 20% 보존이다.
- payload, sparse index, scale, zero-point, position metadata를 포함한다.
- token 수와 bytes를 모두 기록하되 family 비교는 bytes 기준이다.
- target 이하에서 가능한 최선의 구성을 사용하고 `budget_utilization=actual/target`을 기록한다.
- exact target이 불가능한 family는 budget을 초과하지 않고 slack과 이유를 보고한다.
- 20%에서 손실이 없으면 diagnostic manifest에만 `0.05, 0.10`을 추가한다.
- extreme-budget을 실행하지 않은 `B*`는 `<0.2 (left-censored)`로 기록한다.

`estimated_serialized_bytes`는 계산식 결과다. 실제 serializer 산출물 크기와 GPU allocated bytes는
M6에서 별도로 측정한다.

## 4. M0 동등성

### Strict CACHE-IDENTITY

동일 backend·batch·token sequence에서 one-shot prefill과 prefix-cache resume를 비교한다.

- cache tensor max/mean absolute difference
- suffix logits max/mean absolute difference
- greedy prediction equality

### Operational equivalence

chunked prefill, 합법적인 batch 구성 차이 등 실제 실행 변형의 수치 범위를 별도로 측정한다.
동일 입력 반복 run만으로 noise floor를 정하지 않는다. strict threshold와 operational threshold를
합치지 않는다.

## 5. 평가 metric

| 태스크 | 주 metric | 보조 metric |
|---|---|---|
| QA/OCR | 공식 EM/ANLS/F1 | gold-answer logp, normalized loyalty |
| grounding | click success | coordinate error, target agreement |
| action | action type+target accuracy | full-condition action agreement |
| trajectory | episode/task success | step accuracy, first divergence, recovery count |

teacher-forced logp는 selection과 기제 분석에만 쓴다. task metric 대신 결론을 내리지 않는다.
`loyalty`는 full-condition과의 출력·행동 일치도를 보는 보조 canary이며 primary correctness metric을
대체하지 않는다.

## 6. 보존율과 모달리티 통제

```text
absolute_score = 전체 표본의 공식 metric
conditional_retention = memory가 맞힌 IMAGE-REENCODE 정답 표본 / IMAGE-REENCODE 정답 표본
```

“visual KV가 text KV보다 특별히 취약하다”는 주장을 할 때는 같은 모델의 작은 text-only QA
통제군을 M2-A에 포함한다. visual memory 내부의 실패만 주장할 때는 필수 대조가 아니다.

## 7. 표본과 통계

- split은 이미지/episode 단위로 나눈다.
- 같은 이미지의 여러 질문을 독립 표본으로 부풀리지 않는다.
- 이미지 단위 bootstrap 95% CI를 사용한다.
- 평균, 이미지별 분포, worst-group을 함께 보고한다.
- sample count와 selection seed는 config에, 실제 ID는 manifest에 둔다.
- exploration에서 선택한 효과 방향·metric·제외 기준은 M7 전에 동결한다.

## 8. 결과 상태

| 상태 | 의미 | 다음 행동 |
|---|---|---|
| `E` | 탐색 표본에서 효과와 귀속이 명확 | M7 후보 |
| `S` | 특정 모델·태스크·예산에서만 성립 | 범위를 고정해 M7 |
| `N` | 실용적 차이 없음 | 해당 문제 가설 폐기 |
| `I` | 측정 오류, 낮은 base, 넓은 CI, 상충 통제 | 결론 금지·재측정 |

M0 실패, NaN/Inf, 낮은 IMAGE base, 불완전 run은 항상 `I`다.

## 9. run 종류와 경로

```text
results/smoke/         # 코드 경로·측정 gate
results/discovery/     # 가설 탐색
results/confirmation/  # 동결한 가설 확인
archive/results/       # 분석에서 제외한 과거 artifact
```

`archive/results/`는 run kind가 아니다. legacy artifact를 discovery와 합치지 않기 위한 보존소다.
config의 `run_kind`와 `output` 경로는 일치해야 한다.

## 10. 재현성

config에는 `seed`를 둔다. sample별 random selector seed는 다음처럼 재현 가능하게 파생한다.

```text
sample_seed = stable_hash(base_seed, stage, sample_id, question_id, condition_id, keep_ratio)
```

Python 내장 `hash()`는 process마다 달라질 수 있으므로 사용하지 않는다. CRC32 또는 명시한 안정
hash를 사용하고 알고리즘을 run metadata에 기록한다.

모든 run은 source code, config, manifest, model, processor, dataset, analysis revision을 남긴다.

## 11. 결과 schema

중복을 줄이기 위해 run-level metadata와 sample-level record를 분리한다.

### Run metadata

```yaml
schema_version: string
run_id: string
stage: M0|M1|M2A|M3|M4|M2B|M5|M6|M7
run_kind: smoke|discovery|confirmation
source_code_revision: string
config_path: string
config_sha256: string
manifest_path: string
manifest_sha256: string
model_id: string
model_revision: string
processor_mode: string
dataset_revision: string
dtype: string
device: string
base_seed: int
seed_derivation: string
started_at: string
completed: bool
```

### Sample-condition record

```yaml
run_id: string
dataset: string
split: string
sample_id: string
question_id: string|null
sample_seed: int
condition_id: string
selection_timing: write_time|read_time|diagnostic|none
keep_ratio_target: float|null
keep_ratio_actual: float|null
estimated_serialized_bytes: int|null
physical_allocated_bytes: int|null
task_metric_name: string
task_score: float|null
loyalty_score: float|null
answer_logp: float|null
finite: bool
status: E|S|N|I|null
```

smoke runner가 여러 condition을 한 record에 묶는 기존 형식은 legacy schema로 표시한다. READY
runner는 condition별 record를 출력해야 한다.

## 12. manifest gate

- config가 참조하는 manifest는 READY/COMPLETE 전에 존재하고 비어 있지 않아야 한다.
- 최소 `dataset`, `split`, `sample_id`를 포함한다.
- stage별 필수 field는 각 단계 문서가 추가한다.
- discovery와 confirmation ID 중복을 validator가 검사한다.
- PLANNED 단계의 미존재 manifest는 허용하되 unresolved resource로 보고한다.
- 빈 placeholder manifest를 만들지 않는다.

## 13. 전역 중단 조건

- 원인 미확인 NaN/Inf
- strict CACHE-IDENTITY 실패
- 100% keep와 동일 4D/full-position baseline 불일치
- full과 compressed 조건이 서로 다른 attention/position 경로를 사용
- task metric runner 없이 logp만으로 본실험 주장
- config 필수 결정값이 `null` 또는 `TBD`
- READY run의 manifest·runner·revision 부재
