# M0 — 측정 계약

> **목적:** 이후 실험의 성능 차이를 믿을 수 있도록 mask·cache·수치 계산이 정확한지 검증한다.
>
> **왜 필요한가:** 구현 오류도 정보 손실처럼 보일 수 있으므로, 이를 먼저 배제하지 않으면 이후 결과를 해석할 수 없다.

**상태:** `PARTIAL` — NaN 원인 확정·해결(M0-04, DECISIONS §14: QK pre-scale 패치,
384/384 finite sweep). runner `exps/m0_measurement.py` 가동, 허용오차(M0-01) 고정 대기
**질문:** 이후의 성능 차이가 실제 memory 개입 때문인가, 구현 artifact 때문인가?
**실행 계약:** [configs/m0.yaml](configs/m0.yaml)

## 1. 왜 먼저 하는가

이 프로젝트는 4D mask, cache-resume, KV 주입처럼 표준 inference 경로를 바꾼다. 이 경로가
틀리면 압축·위치·전이 실패처럼 보이는 가짜 현상이 생긴다. M0는 연구 결과가 아니라 모든
후속 결과의 측정 자격을 정한다.

## 2. 선행 조건

- Qwen2.5-VL-7B의 full·4D-mask 경로가 모두 finite
- 합성 이미지와 최소 비텍스트 sanity manifest 준비
- [m0.yaml](configs/m0.yaml)의 필수 결정 완료

## 3. 고정 검사

| 코드 | 검사 | 성공 의미 |
|---|---|---|
| M0-A | IMAGE base task metric | 모델이 해당 표본에서 이미지를 실제 사용할 능력이 있음 |
| M0-B | 2D attention vs 4D causal | 4D baseline이 2D 기준을 재현함 |
| M0-C | full visual mask | 시각 정보를 차단하면 image-dependent task가 하락함 |
| M0-D | 100% keep vs no-mask | keep path가 원 계산과 같음 |
| M0-E1 | strict CACHE-IDENTITY | 동일 backend에서 one-shot과 cache-resume가 같음 |
| M0-E2 | operational equivalence | chunk·batch 변화의 수치 범위를 별도로 측정 |
| M0-F | V1 vs V2 | 질문 KV smuggling을 발견하고 V2를 기본으로 고정 |
| M0-G | dtype/processor/seed finite | 효과가 특정 수치 경로 artifact가 아님 |

`STORED-FULL`과 `IMAGE-REENCODE`의 end-to-end 경로 검사는 M1-A gate다. 실제 context/position
portability 차이는 M1-B/C의 연구 결과다.

## 4. 사용자가 수정·결정할 부분

| 결정 ID | 결정 | 현재 상태 | 판단 기준 |
|---|---|---|---|
| M0-01 | strict·operational 허용오차 | TBD | one-shot/resume와 chunk·batch perturbation을 분리 |
| M0-02 | non-text sanity 표본 | TBD | icon/layout/grounding을 각각 포함 |
| M0-03 | IMAGE base 최소 성능 | TBD | 모델 실패와 memory 실패를 구분할 수 있는 수준 |
| M0-04 | NaN 대응 범위 | BLOCKED | mask 행·position·layer별 finite를 먼저 localize |
| 실행값 | GPU device | 현재 script는 `cuda:0` 하드코딩 | 사용할 GPU에 맞춤 |

동일 입력 반복은 bit-identical일 수 있으므로 noise floor로 충분하지 않다. strict identity와
operational perturbation 분포를 따로 측정하고, 후속 효과를 보기 전에 threshold를 고정한다.

## 5. 현재 실행 가능한 명령

```bash
cd /root/research/heewon/VLM
python -m vlm_diagnosis.scripts.smoke_sanity
```

현재 이 명령이 확인하는 것은 합성 OCR의 2D/4D 정합, V1/V2, full visual eviction이다.
non-text sanity와 독립적인 CACHE-IDENTITY 본검사는 포함하지 않는다. fp16 NaN이 발생하면
smoke 실패이며 일부 출력으로 gate를 통과시키지 않는다.

공통 코드 단위 검증:

```bash
python -m unittest discover -s tests -v
```

## 6. 아직 구현할 목표 runner

목표 CLI:

```bash
python -m vlm_diagnosis.exps.m0_measurement \
  --config experiments/configs/m0.yaml \
  --manifest experiments/manifests/m0_sanity.jsonl
```

구현 순서:

- full 2D, full 4D, 100% keep, failing mask의 layer별 finite trace
- mask된 모든 query 행에 유효 key가 남는지 검사
- device/threshold/seed를 CLI·config로 이동
- cache extraction→serialize/load→resume strict identity 비교
- chunked prefill·batch 구성 operational equivalence 비교
- icon/layout/grounding task metric
- JSONL 결과와 PASS/FAIL 요약

## 7. 출력

```text
results/smoke/m0_synthetic_*.json
results/smoke/m0_measurement.jsonl
results/smoke/m0_report.md
```

추가 필드:

```yaml
check: image_base|mask_2d_4d|full_mask|keep100|cache_identity_strict|cache_equivalence_operational|v1_v2|finite
max_abs_logit_diff: float|null
mean_abs_logit_diff: float|null
prediction_equal: bool|null
threshold_used: float|null
```

## 8. 판정

- strict CACHE-IDENTITY, 100% keep, finite 실패: `I`, 후속 결과 해석 금지
- strict는 성공하고 operational만 차이: 구현 오류로 단정하지 않고 실행 환경별 equivalence band 기록
- full mask에도 유지: 모델이 이미지를 안 썼거나 mask가 무효. 둘을 교란 실험으로 분리
- OCR만 민감하고 icon/layout/grounding이 무감: 비텍스트 M0 미완료
- V1만 좋고 V2 하락: V1은 smuggling, V2를 이후 semantics로 사용

## 9. 완료 조건

- NaN 원인과 strict CACHE-IDENTITY를 해결한 뒤 M1 runner 구현을 시작한다.
- M1 결과를 해석·보고하려면 M0-A–G 전체가 통과해야 한다.
- 완료 report에 사용한 허용오차와 실패 표본을 남긴다.
