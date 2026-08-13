# M2-A — 고정 byte 예산에서 sparse 실패를 측정한다

> **목적:** 같은 byte 예산에서 sparse KV의 실패가 저장 용량 부족인지 token 선택 실패인지 구분한다.

**상태:** `PARTIAL`  
**질문:** 기존 sparse 방법은 FULL을 얼마나 보존하며, 실패는 capacity인가 selection인가?
**실행 계약:** [configs/m2a.yaml](configs/m2a.yaml)

## 1. 선행 gate

- M0 전체 통과
- M1에서 해석 가능한 canonical KV 조건 확보
- 공식 task metric runner 준비
- [m2a.yaml](configs/m2a.yaml)의 SOTA 목록과 manifest 고정

## 2. 예산

```text
20% keep / 40% keep / 60% keep / 80% keep
FULL-KV = 100% reference
```

모든 조건은 serialized bytes로 맞춘다. sparse index와 position metadata를 포함한다. 20%에서
손실이 없으면 diagnostic subset에만 5%·10%를 추가한다. 추가하지 않은 상태에서 최소 성공 예산을
보고하면 `B* < 0.2 (left-censored)`로 쓴다.

## 3. Track 1 — 전체 planned split

모든 표본에서 다음을 평가한다.

- `FULL-KV`
- `random`
- `spatial-uniform`
- `published write-time/query-agnostic SOTA` — persistent storage의 주 baseline
- `S1` 및 published read-time/query-aware comparator — full KV 저장 필요 여부 표시

각 방법이 budget 안의 subset을 선택한 뒤, subset만으로 답을 새로 생성하고 공식 metric을
측정한다. 결과는 budget–retention curve와 이미지별 `B*`로 보고한다. read-time 방법은 전체
split에서 평가할 수 있지만 write-time 저장 압축의 직접 비교로 해석하지 않는다.

“visual KV가 text KV보다 특별히 취약하다”는 주장을 채택하면 같은 모델의 작은 text-only QA
통제군을 추가한다. 이 결정은 G07에 기록한다.

## 4. Track 2 — 소규모 원인 진단

task type별 사전 고정한 5–10장에서만 추가한다.

- `target answer-aware A_r(B)`
- `target query-aware Q_r(B)`

answer-aware 절차:

```text
고정 B
→ gold-answer Δlogp로 candidate group 순위 계산
→ B 안의 subset 선택
→ gold answer 없이 실제 답 생성
→ 공식 task metric으로 정답 판정
```

소표본 probe는 전체 성능 baseline이 아니다.

## 5. search audit

- 기본 group: 2×2 spatial patch
- primary: leave-group-out ranking
- audit: backward elimination
- interaction: 상·하위 group pair 제거
- random/beam audit: task별 최대 5장
- 표본·budget별 최대 forward 수는 config에서 고정

search 방식에 따라 `B*` 결론이 바뀌면 `I`다.

## 6. 사용자가 수정·결정할 부분

| 결정 ID | 결정 | 판단 기준 |
|---|---|---|
| M2A-01 | published sparse SOTA 목록 | 현재 패러다임을 충분히 대표하되 VLM adaptation 표시 |
| M2A-02 | diagnostic sample ID | 결과를 보기 전에 task별 5–10장 고정 |
| M2A-04 | 최대 search 계산량 | GPU 시간과 probe 안정성의 균형 |
| M2A-05 | grounding selection score | 실제 click success와 정렬되는지 pilot |
| M2A-06 | 5%·10% 발동 규칙 | 20%에서 손실이 없을 때 diagnostic에만 추가 |
| G03 | 전체 discovery 표본 수 | pilot variance 기반 |
| G07 | text-only 통제군 | visual-specific 귀속을 주장할 때 포함 |

## 7. 현재 실행 가능한 legacy 구성요소

`d4_mini.py`에는 random과 S1 logical mask가 있지만 과거 출력은 FULL과 keep-set의
attention/position 경로가 달랐고 한 shard에서 NaN이 발생했다. 과거 결과는 참고치로도 사용하지
않는다. 코드는 legacy 재현·회귀 검사에만 둔다.

```bash
python -m vlm_diagnosis.exps.d4_mini --shard 0 --nshards 1 --device cuda:0
```

## 8. 목표 runner

```bash
python -m vlm_diagnosis.exps.m2a_fixed_budget \
  --config experiments/configs/m2a.yaml \
  --manifest experiments/manifests/m2a_full.jsonl

python -m vlm_diagnosis.exps.m2a_answer_probe \
  --config experiments/configs/m2a.yaml \
  --manifest experiments/manifests/m2a_diagnostic.jsonl
```

구현해야 할 것:

- 20/40/60/80 byte planner
- 조건부 5/10 diagnostic planner와 censored B* 표기
- spatial-uniform primitive를 full runner에 연결
- published SOTA adapter
- 생성 기반 dataset metric
- answer-aware/query-aware search와 random/beam audit
- budget별 bootstrap와 `B*`
- normalized loyalty 보조 metric

DocVQA manifest 생성 목표 interface:

```bash
python -m vlm_diagnosis.scripts.prep_docvqa \
  --dataset-revision <commit> \
  --manifest experiments/manifests/m2a_diagnostic.jsonl \
  --seed 42
```

## 9. 판정

| 결과 | 허용되는 해석 |
|---|---|
| SOTA<FULL, A_r≈FULL at same B | subset은 존재 가능, 실제 selection이 문제 |
| SOTA와 A_r 모두 실패 | sparse capacity 또는 probe search 한계 |
| random/uniform≈SOTA | 복잡한 importance의 추가 가치가 약함 |
| random≈A_r≈FULL | 시각 KV 중복이 커 selector 연구 동기가 약함 |
| 80%에서만 보존 | 강한 압축은 어렵고 낮은 제거율에서만 가능 |
| 20%에서 모두 FULL, 5/10 미실행 | 실패 없음이 아니라 `B*<0.2` censored |
| text-only 유지, visual만 하락 | visual modality/workload 귀속 강화 |
| task score 유지, loyalty만 하락 | correctness 유지 속 behavior drift canary |

## 10. 완료 조건

- Track 1은 전체 manifest에서 task metric과 CI가 있음
- Track 2는 sample ID와 search budget이 고정됨
- logp와 task metric을 혼합하지 않음
- 각 keep ratio의 실제 bytes가 기록됨
- capacity/selection 귀속 불가 표본은 `I`로 분리
