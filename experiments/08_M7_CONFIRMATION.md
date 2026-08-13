# M7 — 발견한 문제 자체를 확인한다

> **목적:** 앞 단계에서 발견한 문제 하나가 새로운 데이터와 독립적인 모델 조건에서도 재현되는지 확인한다.
>
> **왜 필요한가:** 발견한 현상이 특정 표본이나 모델 계보의 우연이 아닌 일반적인 문제인지 확인해야 연구 주장으로 확정할 수 있다.

**상태:** `PLANNED`  
**질문:** M0–M6에서 선택한 현상이 새 표본과 다른 모델에서도 재현되는가?
**실행 계약:** [configs/m7.yaml](configs/m7.yaml)

## 1. 실행 전 동결할 것

M7은 새로운 현상을 탐색하는 단계가 아니다. 다음을 결과 보기 전에 동결한다.

- 확인할 현상 하나
- 문제 문장
- primary metric
- 효과 방향
- practical minimum effect
- 핵심 대조군
- 제외 기준
- 분석 코드와 그림
- confirmation sample ID
- 발견된 현상에 필요한 독립성 기준과 최소 두 confirmation 모델

## 2. 사용자가 수정·결정할 부분

| 결정 ID | 결정 | 판단 기준 |
|---|---|---|
| M7-01 | 확인할 현상 하나 | 효과·귀속·실용 귀결이 가장 명확한 것 |
| G04 | 모델 독립성 기준과 후보 | position/fusion/LLM 중 무엇을 일반화하는지 먼저 고정 |
| G05 | 최소 효과 | 관측 효과가 아니라 task/SLO 의미로 결정 |
| M7-02 | sample size | discovery variance 기반 power |
| M7-03 | exclusion·analysis revision | 결과 전 Git commit과 규칙 동결 |
| M4-03 | 외부 dataset 2개 | 현상 유형에 맞춰 사전에 routing |

### 모델 선택 규칙

- Qwen3-VL과 UI-TARS만으로는 Qwen/mRoPE 계보 밖 일반화를 주장하지 않는다.
- OpenCUA-7B는 Qwen2.5-VL 기반이지만 1D RoPE를 사용하므로 position finding의 대조 후보가 된다.
- InternVL3와 LLaVA-OneVision은 visual integration이 다르지만 language backbone은 Qwen2 계열이다.
- architecture-general 주장을 할 때는 language backbone까지 non-Qwen인 후보를 최소 1개 포함한다.
- 후보 이름보다 해당 revision의 cache-resume 구현 가능성과 discovery task base 성능을 먼저 smoke한다.

## 3. 금지 사항

- confirmation 결과를 보고 metric 변경
- 효과가 큰 subgroup만 새 primary로 변경
- 실패 모델을 근거 없이 제외
- discovery와 같은 이미지 재사용
- M7에서 새 방법 후보를 추가해 문제 확인과 방법 검증을 혼합

## 4. 목표 runner

```bash
python -m vlm_diagnosis.exps.m7_confirmation \
  --config experiments/configs/m7.yaml \
  --manifest experiments/manifests/m7_confirmation.jsonl
```

M7 runner는 선택한 현상에 따라 M1/M2/M3/M4/M5/M6 runner를 호출하되 config와 분석 코드를
동결해야 한다.

## 5. 판정

| 결과 | Phase 1 결론 |
|---|---|
| 방향·효과가 새 표본과 사전 정의한 독립 모델 조건에서 재현 | 범위를 명시한 문제의식 확립 |
| 방향은 같지만 최소 효과보다 작음 | 실용 문제 약함, Phase 2 보류/범위 축소 |
| 한 architecture에서만 재현 | architecture-specific `S` |
| 한 정보 유형에서만 재현 | task-specific `S` |
| 효과 소실·방향 반전 | 원 가설 `N`, Phase 2 진입 금지 |
| CI가 넓음 | `I`, 사전 power 규칙에 따라 표본 추가 |

## 6. 최종 산출물

```text
results/confirmation/m7_results.jsonl
results/confirmation/m7_report.md
results/confirmation/frozen_config.yaml
results/confirmation/frozen_manifest.jsonl
```

최종 report는 다음 문장을 채워야 한다.

> [재사용 조건]에서 [기준 표현/정책]은 [정보·태스크] 성능을 [효과]만큼 잃는다.
> [대조 개입]으로 [병목]에 귀속했고 [모델·데이터]에서 재현했다.
> 이 때문에 [실제 시스템 선택]이 달라진다.
