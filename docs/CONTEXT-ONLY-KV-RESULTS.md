# Context-only KV 압축 — 구현·실험 결과 (VLM 화면 표본, 20% 유지 기준)

작성일: 2026-09-07 · 명세: [CONTEXT-ONLY-KV-COMPRESSION.md](CONTEXT-ONLY-KV-COMPRESSION.md) · 결과 요약 파일: `results/context_only/*_summary.md`

## 0. 한 줄 요약

> 미래 질문을 보지 않고 **최초 context prefill 한 번**에서 나오는 MLP·hidden-state 통계(MLP norm, R, D, hidden 변화)로
> KV 쌍을 고르면, **모든 유지율(50/20/10/5%)에서 무작위 선택보다 나쁘다.** 20% 유지에서 "낮은 점수 삭제 < 무작위 < 높은 점수 삭제"라는
> 문서의 가설 방향은 신호 10개 중 0개가 만족한다. 명세 §1.2 기준으로 **dynamics 의 추가 가치는 미확인이 아니라 부정**이다.
> 같은 harness 에서 재구성(설명문 생성) 점수만 무작위를 크게 넘는다(5% 유지에서 +0.465).

## 1. 명세에서 바꾼 것

| 항목 | 명세 | 이번 실행 | 이유 |
|---|---|---|---|
| 모델·데이터 | text Qwen2.5-0.5B, 합성 context | Qwen2.5-VL-7B (fp16, V100), ScreenQA 화면 172장 (개발 40 / 평가 132, 화면 단위 분할), 화면당 질문 3개 | 사용자 지시 ("지금 VLM 표본으로") |
| 단계 4 유지율 | 80% 유지 | **20% 유지** | 사용자 지시 |
| 보호 special token | tokenizer special id 전부 | 이미지/비디오 placeholder 제외 | placeholder(1,222개)를 보호하면 보호 쌍이 예산을 넘어 실행 불가. 기록에 남김 |
| 그 외 | — | 명세대로 | 단일 prefill, (layer, KV head, token) 쌍 단위 실제 삭제, 전역 top-B, seed 동점 처리, 독립 분기 평가, 비용 회계 |

## 2. 구현 (모두 CPU 단위 테스트 포함, 전체 264개 통과)

| 파일 | 내용 |
|---|---|
| [core/mlp_dynamics.py](../vlm_diagnosis/core/mlp_dynamics.py) | §4 수집기. x(층 입력), r(MLP 직전 잔차), m(MLP 출력), x_next(층 출력)를 hook 으로 받아 R, D, hidden 변화(크기·방향)를 FP32로 계산. `x_next ≈ r + m` 층별 검사(최대 상대오차 기록), 층 재호출 시 실패, 예외 시 hook 제거. stats 모드는 d 차원을 즉시 축약 |
| [core/static_pair_select.py](../vlm_diagnosis/core/static_pair_select.py) | §4.3·§5. 명시적 mapping(`d_same_zero0`, `d_shift_prev`, `*_same`), `B = round(keep_ratio·L·H·T)`, 보호 쌍 예산 포함, seed 순열 동점 처리, `keep.sum()==B` 검사, 층별 삭제 수 맞춤 대조군, 경계 대조군(B0 규칙), 층 순서 섞은 D, anchor D, 평균 순위 Spearman |
| [core/context_only_cache.py](../vlm_diagnosis/core/context_only_cache.py) | §3. `compress_context(model, processor, image, method, keep_ratio, seed)` — 질문 미입력, prefill 1회, 기존 `RaggedKVCache` 로 실제 쌍 삭제, dense 참조 해제. `clone_owned()` 독립 분기, `answer_from_cache`(suffix 먼저 처리 후 greedy), `answer_nll`(정답 본문 token 만, EOS 제외) |
| [exps/context_only_kv.py](../vlm_diagnosis/exps/context_only_kv.py) | 단계별 runner (`full / probe / deletion / sweep / profile`), §8.3 기록 형식 (run/build/answer/diagnostic/parity/error) |
| [scripts/context_only_analysis.py](../vlm_diagnosis/scripts/context_only_analysis.py) | 로그 검증(중복·미지 record·metadata 불일치 거부), FULL 참조 없는 질문 제외 수 표시, context 단위 paired bootstrap |
| tests | `test_mlp_dynamics.py`(7), `test_static_pair_select.py`(10), `test_context_only_cache.py`(4) |

결과 파일: `results/context_only/` (`*_summary.md` 만 커밋, 원본 jsonl 은 gitignore).

## 3. 단계 1 — parity (완료 기준 통과)

유지 100% 의 물리 캐시 경로(context prefill → 질문 suffix 만 처리)와 일반 FULL forward `[context + question]` 을 비교.
개발 화면 8장 × 질문 2개 ([full_dev_summary.md](../results/context_only/full_dev_summary.md)).

| 검사 | 결과 |
|---|---|
| 질문 suffix 의 RoPE 위치가 dense 와 일치 | 16/16 |
| 첫 답 token(= suffix 마지막 위치 logits 의 argmax) 일치 | 16/16 |
| suffix 전체 argmax 일치율 | 15개 1.000, 1개 0.947 |
| 정답 NLL 차이 (cached − dense) | 최대 0.009, 중앙값 0.0002 |
| logits 최대 절대 오차 / 평균 | 0.08~0.33 / 0.010~0.017 |

logits 오차는 fp16 에서 ragged backend(FP32 softmax, head 별 계산)와 eager attention 의 계산 순서 차이다. 질문 순서 불변·master 불변·분기 독립은 단위 테스트로 확인.

## 4. 단계 2 — 관측 (단일 prefill 통계)

화면 8장, prefix 1,238 token (시각 1,222). 잔차 검사 최대 상대오차 4.9e-4 (fp16). ([probe_dev_summary.md](../results/context_only/probe_dev_summary.md), heatmap `results/context_only/probe_figs/`)

- **R 과 D 모두 보호 대상 sink·서식 token 이 지배**한다. 상위 D token 은 항상 줄바꿈(index 2), `<|im_start|>`(0), `<|im_end|>`(9). raw heatmap 은 그 한 점(R≈120)만 보인다. 명세가 예고한 대로 표시용 clip heatmap 을 따로 둔다.
- 시각 token 안에서 R 평균 0.32, 비시각 0.68; D 는 시각 0.07, 비시각 0.71 (10배).
- 표시용 heatmap 에서 R·D 의 분산은 **token 축이 아니라 층 축**에 있다(마지막 층은 모든 token 이 높고, 중간 층은 띠 모양). token 사이 차이는 잡음처럼 보인다. 이것이 §5·§6 결과의 시각적 설명이다.
- R–D token 순위 상관 0.59~0.75: D 는 대체로 R 을 따라간다.

## 5. 단계 4 — 삭제 민감도 (개발 40장, 질문 120개, **20% 유지**, 실제 쌍 삭제)

FULL 정답률 0.858. 무작위 20% (seed 5개) 0.575~0.625. ([deletion_dev_summary.md](../results/context_only/deletion_dev_summary.md))

| 신호 | 낮은 점수 삭제(keep_high) | 무작위 | 높은 점수 삭제(keep_low) | 낮은−무작위 [95% CI] | 높은−무작위 | 가설 방향 |
|---|---|---|---|---|---|---|
| D (`d_same_zero0`) | 0.292 | 0.600 | 0.450 | −0.308 [−0.400, −0.208] | −0.150 [−0.250, −0.050] | 아니오 |
| D anchor 대조 | 0.208 | | 0.508 | −0.392 | −0.092 | 아니오 |
| D 층 순서 섞음 | 0.283 | | 0.583 | −0.317 | −0.017 [−0.108, +0.083] | 아니오 |
| R | 0.192 | | 0.267 | −0.408 | −0.333 | 아니오 |
| R 표준편차 | 0.333 | | 0.200 | −0.267 | −0.400 | 아니오 |
| MLP norm | 0.200 | | 0.100 | −0.400 | −0.500 | 아니오 |
| hidden 상대 변화 | 0.225 | | 0.242 | −0.375 | −0.358 | 아니오 |
| hidden 방향 변화 | 0.217 | | 0.392 | −0.383 | −0.208 | 아니오 |
| K norm | 0.100 | | 0.167 | −0.500 | −0.433 | 아니오 |
| V norm | 0.317 | | 0.100 | −0.283 | −0.500 | 아니오 |

읽는 법: 명세의 가설은 "낮은 점수 삭제 손실 < 무작위 < 높은 점수 삭제 손실". 실제로는

1. **두 방향 모두 무작위보다 나쁘다** (10개 신호 × 2 방향 = 20개 조건 전부, 무작위 대비 차이의 95% CI 가 0 아래).
2. D·R·hidden 변화·K norm 은 **높은 점수를 지우는 편이 덜 아프다**. 큰 쓰기 변화량은 보존 가치가 아니라 약하게 "지워도 되는 쪽"과 관련된다.
3. 층 순서를 섞은 D 는 높은 점수 삭제에서 무작위와 같아진다. 인접 층 dynamics 가 주는 정보는 확인되지 않았다.
4. ΔNLL 도 같은 방향이다 (D keep_high +0.83 vs 무작위 +0.21).

대조군: 층별 삭제 수를 맞춘 D 0.350 > 전역 D 0.292 — `d_same_zero0` 이 layer 0 에 보호분 외 아무것도 남기지 않는 것이 손실의 일부이지만, 경계 대조군(layer 0 무작위 공유) 0.283 이라 경계 규칙만의 문제는 아니다. recent(최근 token 유지) 0.317.

**mask 시뮬레이션과의 일치**: 같은 화면·질문(12장, 35개)에서 앞선 attention-mask 방식과 물리 삭제가 같은 값을 낸다 — FULL 0.886 = 0.886, 무작위 20% 0.65 vs 0.57~0.60, MLP 계열 0.14~0.21 vs 0.20~0.29. 앞선 mask 결과가 물리 삭제로 재현된다.

## 6. 단계 5 — 유지율 sweep (평가 132장, 질문 396개, 오류 0)

FULL 0.803 [0.755, 0.846]. ([sweep_eval_summary.md](../results/context_only/sweep_eval_summary.md))

정답률 (행 = 방법, 열 = 유지율):

| 방법 | 비용 표시 | 0.5 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|
| 무작위 | 없음 | 0.778 | 0.609 | 0.374 | 0.227 |
| recent | 없음 | 0.528 | 0.273 | 0.149 | 0.114 |
| K norm | 없음 | 0.649 | 0.088 | 0.096 | 0.078 |
| V norm | 없음 | 0.773 | 0.313 | 0.182 | 0.121 |
| MLP norm | 단일 prefill 통계 | 0.747 | 0.210 | 0.131 | 0.096 |
| R | 단일 prefill 통계 | 0.548 | 0.194 | 0.141 | 0.109 |
| D (`d_same_zero0`) | 단일 prefill 통계 | 0.684 | 0.232 | 0.141 | 0.111 |
| context-only attention (attn1) | 같은 prefill 의 attention 재계산 +0.20 s | 0.785 | 0.631 | 0.384 | 0.225 |
| 재구성 (설명문 생성, 계약 밖) | 생성 87 token + forward 1회, +10.3 s | **0.798** | **0.763** | **0.720** | **0.692** |

무작위 대비 짝지은 차이 (같은 유지율):

| 방법 | 0.5 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|
| D | −0.093 [−0.134, −0.053] | −0.376 [−0.432, −0.321] | −0.232 | −0.116 |
| R | −0.230 | −0.414 | −0.232 | −0.119 |
| MLP norm | −0.030 [−0.063, +0.005] | −0.399 | −0.242 | −0.131 |
| attn1 | +0.008 [−0.020, +0.035] | +0.023 [−0.023, +0.068] | +0.010 | −0.003 |
| 재구성 | +0.020 [−0.008, +0.051] | **+0.154 [+0.109, +0.199]** | **+0.346** | **+0.465 [+0.409, +0.520]** |

FULL-correct 보존율도 같다: 20% 유지에서 무작위 0.720, D 0.288, MLP norm 0.261, 재구성 0.928.

보조 진단(재구성 점수와 평균 순위 Spearman, 보호 쌍 제외): attn1 0.38, V norm 0.31, MLP norm 0.21, **D −0.00, R −0.01**, K norm −0.17. 명세 §1.2 의 경고("상관이 높아도 삭제 성능이 나쁘면 성공이 아니다")는 여기서 반대 방향으로 적용된다 — 상관이 0 이고 삭제 성능도 나쁘다.

## 7. 비용 (단계 5 profile, 개발 20장, 배포 경로 `compress_context`: dense 해제, 다른 방법 동반 실행 없음)

| 방법 | prefill s (통계 수집 포함) | scorer overhead s (plain 대비) | score+select+prune s | build 총 s | 초기 peak GB (모델 제외) | persistent MB (KV+ID+template) | 질문 prefill s | decode s / token | 정답률 (20장) |
|---|---|---|---|---|---|---|---|---|---|
| plain (FULL, 통계 없음) | 0.959 | — | 0.060 | 1.071 | 0.15 | 68.8 | 0.099 | 0.088 | 0.800 |
| D (20%) | 0.994 | +0.035 | 0.064 | 1.117 | 0.10 | 13.8 | 0.099 | 0.089 | 0.350 |
| R (20%) | 0.992 | +0.033 | 0.065 | 1.112 | 0.10 | 13.8 | 0.097 | 0.087 | 0.217 |
| MLP norm (20%) | 0.998 | +0.038 | 0.056 | 1.108 | 0.10 | 13.8 | 0.095 | 0.084 | 0.233 |
| K norm (20%) | 0.955 | −0.004 | 0.065 | 1.074 | 0.10 | 13.8 | 0.100 | 0.085 | 0.117 |

- 통계 수집(hook, FP32 norm, 층마다 `.cpu()`)의 추가 비용은 prefill 당 약 0.035 s (3.7%). 선택+삭제 복사는 0.06 s. 즉 "압축 준비 비용이 낮다"는 명세의 전제는 성립한다 — 성능이 성립하지 않을 뿐이다.
- 재구성 방식의 준비 비용은 같은 화면에서 +10.3 s (설명문 87 token 생성 + forward 1회; §6). 비용 비율 약 300배.
- persistent bytes 는 68.8 MB → 13.8 MB (정확히 20% + 생존 ID 8 B/쌍 + template). CUDA timing 은 synchronize 후 perf_counter.
- decode 시간이 FULL 과 20% 에서 같은 것은 Python head-loop ragged backend 때문이며 속도 향상 주장이 아니다 (명세 §8.2).
- 초기 peak 는 모델 상주분(15.5 GB) 을 뺀 값이며 vision encoder 활성값과 dense KV(68.8 MB)를 포함한다.

## 8. 판정 (명세 §1.2 기준)

| 기준 | 판정 |
|---|---|
| 동일 예산에서 성능 비슷 + 준비 비용 낮음 → 효율 기여 | **아니오**. 준비 비용은 낮지만(통계 수집 ≈ 0.1 s 미만) 성능이 무작위보다 낮다 |
| 동일 성능을 더 적은 저장량으로 → 압축 품질 기여 | **아니오**. 어떤 유지율에서도 무작위를 넘지 못한다 |
| D 가 R·norm 보다 낫지 않으면 dynamics 추가 가치 미확인 | D 는 R 보다는 낫지만(20%: 0.232 vs 0.194) 둘 다 무작위 아래. 층 순서 섞은 D 와도 차이 없음 → **미확인** |
| 상관 높아도 삭제 성능 나쁘면 실패 | 해당 없음 (상관 자체가 0) |
| 작은 모델·소수 context 로 novelty 주장 금지 | 이번 결과는 7B VLM·화면 172장·질문 516개에서의 **부정 결과**이며, 텍스트 모델·자연 문서로의 일반화는 주장하지 않는다 |

해석: 단일 prefill 의 MLP·hidden 통계는 "이 token 의 표현이 이 층에서 얼마나 바뀌었나"(쓰기 크기)를 재고, KV 선택이 맞혀야 하는 것은 "나중에 어떤 질문이 이 token 의 K/V 를 읽는가"(읽기 필요도)다. 관측상 두 사건은 무관하거나 약하게 반대다. 읽는 행위가 있는 신호(재구성 0.69 @5%)만 무작위를 넘고, 읽는 목적이 없는 attention(attn1)은 무작위와 같다.

## 9. 남은 제약과 다음 실험

- 텍스트 전용 모델·합성 context 에서는 실행하지 않았다. 명세의 원래 순서(텍스트 → VLM)를 뒤집은 것이므로, 텍스트에서 D 가 다르게 행동할 가능성은 열려 있다. 다만 같은 harness 로 30분이면 돌릴 수 있다 (`--model` 에 text adapter 추가 필요).
- MLP 신호가 답할 수 있는 질문은 "어느 **층**에서 처리가 일어나나"다. token 선택이 아니라 **층 간 예산 배분**에 쓰는 변형(층 안 순위는 attention/재구성)은 아직 시험하지 않았다. 층별 맞춤 대조군이 전역보다 나았던 것(0.35 vs 0.29)이 유일한 긍정 신호다.
- 재구성의 비용(+10 s, 생성 87 token)을 줄이는 쪽 — 짧은 고정 지시문 행만 쓰는 probe — 은 12장 부분 결과에서 20% 유지 0.44 로 무작위(0.65)보다 낮았다 (`results/smoke/sp1_q25_sqa.shard0.jsonl`, 중단됨). 확정하려면 재실행이 필요하다.
- Qwen3-VL(DeepStack) 은 hook 위치가 달라 이번 수집기를 그대로 쓰지 않는다.
