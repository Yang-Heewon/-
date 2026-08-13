# M6 — 실제 시스템에서 쓸 이유가 있는가

> **목적:** 정확도를 유지한 memory 표현이 실제 저장 공간·지연·처리량에서도 실용적 이점을 가지는지 확인한다.

**상태:** `PARTIAL`  
**질문:** 정확도를 보존하는 memory 표현이 저장·로드·지연·동시성에서도 비지배 영역을 가지는가?
**실행 계약:** [configs/m6.yaml](configs/m6.yaml)

## 1. 비교 표현

- IMAGE
- T_visual
- T_episode
- FULL-KV
- M2-A/M2-B에서 관측된 SPARSE/QUANT/TRANSFORMED/HYBRID 조건

정확도 없이 비용만 비교하지 않고, 같은 task score 수준의 Pareto frontier를 본다.
압축 조건은 20/40/60/80% keep를 모두 측정하고 FULL은 100% reference로 둔다.

## 2. 측정 항목

- write-time 생성·압축 비용
- serialized bytes/episode
- disk→RAM→GPU cold/warm load
- image re-prefill 포함 TTFT p50/p95
- GPU memory allocated/reserved
- throughput과 동시 session 수
- 보존 기간과 월 회상 빈도별 총비용

## 3. 사용자가 수정·결정할 부분

| 결정 ID | 결정 | 판단 기준 |
|---|---|---|
| M6-01 | workload와 회상 빈도 | 실제 사용 시나리오의 low/medium/high |
| M6-02 | TTFT/throughput SLO | 어떤 지연 개선이 시스템 선택을 바꾸는가 |
| M6-03 | GPU·storage 가격 | 측정값과 가격 가정을 분리 |
| M2B-03 | physical backend와 GPU | V100 smoke를 본측정으로 오인하지 않기 |

## 4. 현재 실행 가능한 physical smoke

```bash
python -m pip install -r requirements-baselines.txt
python -m vlm_diagnosis.scripts.smoke_quantized_cache \
  --backend hqq \
  --nbits 4 \
  --device cuda:0
```

이 명령은 Qwen2.5-VL에서 actual quantized-cache 경로가 실행되는지만 확인한다. latency,
peak memory, quality benchmark가 아니다. Quanto는 V100 sm_70에서 지원되지 않는다.

## 5. 목표 runner

```bash
python -m vlm_diagnosis.exps.m6_system \
  --config experiments/configs/m6.yaml \
  --manifest experiments/manifests/m2a_full.jsonl
```

필요 구현:

- warmup과 반복 측정
- CUDA synchronize와 peak-memory reset
- cold/warm load 분리
- image/text write cost 포함
- workload sensitivity 계산

## 6. 판정

| 결과 | 해석 |
|---|---|
| compressed KV가 현실 workload에서 비지배 | 조건부 시스템 가치 있음 |
| IMAGE가 정확도·총비용에서 지배 | persistent KV 동기 약함 |
| KV는 latency만 우세 | latency/SLO 특화 문제로 범위 축소 |
| warm에서만 우세 | hit-rate 높은 serving에 한정 |
| T_visual은 semantic 우세, grounding 실패 | tiered memory 가능성 |
| 손익분기 회상 빈도가 현실 밖 | 수학적 교차점만 있고 실용 문제는 아님 |

## 7. 완료 조건

- 동일 hardware/software revision
- p50/p95와 반복 횟수 보고
- task quality와 시스템 metric을 같은 condition ID로 연결
- 실제 allocated bytes와 estimated serialized bytes를 구분
- 가격 없는 물리 측정 surface와 가격 기반 해석을 분리
