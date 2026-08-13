# 공통 실험 프로토콜

모든 단계는 이 문서의 정의를 공유한다. 단계 문서가 이 문서와 다르면 단계 문서에 이유를
명시해야 한다.

## 1. 기준 표현

- `NO-MEM`: memory 없이 미래 질문 수행
- `IMAGE-REENCODE`: 원본 이미지를 read 시 다시 인코딩
- `FULL-KV`: visual KV 100% 저장·주입
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

## 2. 예산과 byte matching

```text
keep_ratio = serialized_bytes(memory) / serialized_bytes(FULL-KV)
```

- 압축 조건: `0.2, 0.4, 0.6, 0.8`
- 무압축 reference: `1.0`
- `0.2`는 20% 제거가 아니라 20% 보존이다.
- metadata, index, scale, zero-point를 포함한다.
- token 수와 bytes를 모두 기록하되 family 비교는 bytes로 맞춘다.

## 3. 평가 metric

| 태스크 | 주 metric | 보조 metric |
|---|---|---|
| QA/OCR | 공식 EM/ANLS/F1 | normalized gold-answer logp |
| grounding | click success | normalized coordinate error |
| action | action type+target accuracy | step success |
| trajectory | episode/task success | step accuracy, recovery count |

teacher-forced logp는 selection과 기제 분석에만 쓴다. task metric 대신 결론을 내리지 않는다.

## 4. 보존율

두 결과를 같이 보고한다.

```text
absolute_score = 전체 표본의 공식 metric
conditional_retention = memory가 맞힌 IMAGE-REENCODE 정답 표본 / IMAGE-REENCODE 정답 표본
```

원 모델이 틀린 표본을 memory 실패로 세지 않기 위해서다.

## 5. 표본과 통계

- split은 이미지/episode 단위로 나눈다.
- 같은 이미지의 여러 질문을 독립 표본으로 부풀리지 않는다.
- 이미지 단위 bootstrap 95% CI를 사용한다.
- 평균, 이미지별 분포, worst-group을 함께 보고한다.
- exploration에서 선택한 효과 방향·metric·제외 기준은 M7 전에 동결한다.

## 6. 결과 상태

| 상태 | 의미 | 다음 행동 |
|---|---|---|
| `E` | 탐색 표본에서 효과와 귀속이 명확 | M7 후보 |
| `S` | 특정 모델·태스크·예산에서만 성립 | 범위를 고정해 M7 |
| `N` | 실용적 차이 없음 | 해당 문제 가설 폐기 |
| `I` | 측정 오류, 낮은 base, 넓은 CI, 상충 통제 | 결론 금지·재측정 |

M0 실패, NaN/Inf, 낮은 IMAGE base, 불완전 run은 항상 `I`다.

## 7. run 종류

결과 경로와 해석 범위를 구분한다.

```text
results/smoke/         # 코드 경로만 확인
results/discovery/     # 가설 탐색
results/confirmation/  # 동결한 가설 확인
```

smoke 결과를 discovery 결과와 합치지 않는다.

## 8. 최소 결과 schema

모든 JSONL record 또는 run metadata에 다음 필드를 둔다.

```yaml
run_id: string
stage: M0|M1|M2A|M3|M4|M2B|M5|M6|M7
run_kind: smoke|discovery|confirmation
model_id: string
model_revision: string
processor_mode: string
dtype: string
device: string
dataset: string
split: string
sample_id: string
question_id: string|null
seed: int
condition_id: string
keep_ratio: float|null
estimated_serialized_bytes: int|null
physical_allocated_bytes: int|null
task_metric_name: string
task_score: float|null
answer_logp: float|null
finite: bool
status: E|S|N|I|null
```

단계별 추가 필드는 각 MD가 정의한다.

## 9. 전역 중단 조건

- CACHE-IDENTITY 실패
- 100% keep와 no-mask 불일치
- 원인 미확인 NaN/Inf
- task metric runner 부재 상태에서 logp만으로 본실험 주장
- config의 필수 결정값이 `null` 또는 `TBD`

