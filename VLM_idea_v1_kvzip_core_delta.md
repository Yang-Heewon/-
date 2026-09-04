# VLM-specific Core–Delta KV Compression

> **보관 사본**: 2026-09-04 아침에 사용자가 작성한 원안. 같은 날 오후 다른 세션이 `VLM_idea.md`를
> "Dual-Prefill Importance Union"으로 덮어써서, 원안의 내용을 이 파일에 그대로 복원해 둔다.
> (복원 출처: vlm-78 세션이 아침에 읽은 원문. 문구·수식·표 모두 원문 그대로.)

> **상태**: method seed / 검증 전 가설  
> **작성일**: 2026-09-04  
> **출발점**: KVzip의 reconstruction-based query-agnostic eviction을 VLM에 확장하되,
> 실제 질문이 알려진 뒤 필요한 visual/text KV를 query-conditioned delta로 추가한다.

## 1. 한 문장 아이디어

> **제한된 전체 KV 예산 안에서 이미지 자체를 보존하는 query-independent core와 현재
> 질문에 필요한 query-conditioned delta를 layer·head·token별로 공동 선택한다.**

이를 다음과 같이 표현한다.

\[
M_{l,h}(I,q)
=
C_{l,h}(I)
\cup
\Delta_{l,h}(I,q),
\qquad
\operatorname{Cost}(M)\le B.
\]

- \(I\): 이미지 또는 multimodal context
- \(q\): 현재 질문·instruction
- \(C\): 이미지의 범용 구조와 내용을 보존하는 **core KV**
- \(\Delta\): 현재 질문에 직접 필요한 **query-specific KV**
- \(B\): layer/head/token 전체를 합산한 실제 KV byte budget

이 방법의 단일 motivation은 다음과 같다.

> **강한 VLM KV 압축에서는 generic reconstruction만으로는 질문 관련성이 부족하고,
> query attention만으로는 답에 간접적으로 필요한 시각적 문맥을 놓칠 수 있다. 두 신호를
> 동일 예산에서 함께 최적화해야 한다.**

두 번째 절반인 “query attention이 범용 문맥을 놓친다”는 아직 검증된 결론이 아니라
이 방법이 검증해야 할 핵심 가설이다.

---

## 2. KVzip에서 무엇을 계승하고 무엇을 바꾸는가

### 2.1 KVzip

[KVzip](https://arxiv.org/abs/2505.23416)은 원문을 다시 reconstruction할 때 각 KV pair가
받는 attention으로 중요도를 추정한다.

- 선택 단위: \((\text{layer},\text{KV head},\text{token position})\)
- 점수 형태: \(S\in\mathbb{R}^{L\times H_{KV}\times N}\)
- 예산: 전체 pair에 대한 non-uniform allocation
- 압축 시점: context마다 한 번
- 이후 동작: 압축된 동일 cache를 여러 미래 질문에 재사용

즉 KVzip은 layer/head별로 서로 다른 token을 남기지만, 압축 이후 질문이나 decoding
step마다 기존 context mask를 다시 계산하지는 않는다.

### 2.2 제안 방법

제안 방법은 KVzip의 reconstruction score를 **core branch**로 사용하고, 실제 질문에서
나오는 relevance score를 **delta branch**로 추가한다.

```text
                         ┌─ Core scorer C(I) ───────────┐
Image/context → full KV ─┤                              ├→ active KV cache
Question q ──────────────└─ Query scorer Δ(I, q) ──────┘
```

따라서 이 방법은 순수 KVzip이 아니라 다음과 같은 발전형이다.

> **KVzip-derived query-agnostic core + query-conditioned residual cache**

단순히 layer별로 다른 KV를 남긴다는 것은 이미 KVzip에 있으므로 novelty가 아니다.
차별화의 중심은 **core와 delta를 하나의 global byte budget 안에서 joint optimization**하는
데 있다.

---

## 3. 중요한 구조적 사실: 두 종류의 KV를 만드는 것이 아니다

Qwen 계열 decoder-only VLM에서는 일반적으로 visual token이 질문 token보다 앞에 놓인다.
causal attention 때문에 뒤에 있는 질문은 앞의 visual hidden state와 KV 값을 바꿀 수 없다.

따라서 다음 표현은 부정확하다.

> image-only KV와 image+prompt KV를 각각 만든 뒤 합친다.

정확한 표현은 다음과 같다.

> **동일한 visual KV에 대해 image-derived core importance와 prompt-conditioned relevance를
> 각각 측정하고, 두 selection mask를 결합한다.**

```text
visual KV value K/V       : 질문을 뒤에 붙여도 동일
core importance C(I)      : 질문 없이도 보존할 가치
query relevance Δ(I, q)   : 질문 token이 해당 KV를 얼마나 필요로 하는지
```

K와 V는 따로 선택하지 않고 같은 위치의 KV pair로 함께 보존한다.

---

## 4. Phase A — NN 없이 아이디어 존재 여부부터 검증

NN 학습 전에 기존 score를 이용해 두 신호가 실제로 complementary한지 확인한다.

### 4.1 초기 score

#### Core score

첫 후보는 현재 구현된 KVzip-VLM score다.

- `score_kvzip`: “보이는 내용을 정확히 반복하라”는 generic reconstruction에서 visual
  KV가 받은 maximum attention
- `score_s5`: 상세 설명 생성 token이 visual KV에 준 attention mass
- 보조 후보: spatial coverage

순수 K-norm이나 encoder norm은 core 후보로 비교할 수 있지만, 현재 결과만으로 주력
신호로 채택하지 않는다.

#### Query score

- `score_s1`: 실제 질문과 assistant header token이 visual KV에 준 attention mass

현재 구현 위치는
[signals.py](/root/research/heewon/VLM/vlm_diagnosis/core/signals.py)다. 현재 score는
layer/head를 최종적으로 합친 visual-token scalar이므로 Phase A는 **공통 token mask를
사용하는 진단 실험**이다. 최종 per-layer/head method와는 구분한다.

### 4.2 고정 예산 결합

전체 keep budget을 \(B\), core 비율을 \(\alpha\)라고 하자.

\[
B_C=\operatorname{round}(\alpha B),
\qquad
B_Q=B-B_C.
\]

선택 절차는 다음과 같다.

1. core score 상위 \(B_C\)개를 먼저 보호한다.
2. 아직 선택되지 않은 token 중 query score 상위 \(B_Q\)개를 추가한다.
3. 최종 token 수가 항상 정확히 \(B\)인지 확인한다.
4. 모든 비교군에 동일한 serialized KV byte budget을 적용한다.

단순히 core top-k와 query top-k를 합집합으로 묶으면 최종 예산이 커질 수 있다. 그 경우
성능 향상이 selector 때문인지 KV 증가 때문인지 구분할 수 없으므로 사용하지 않는다.

### 4.3 첫 sweep

- KV budget: 1%, 2%, 5%, 10%, 20%
- core 비율 \(\alpha\): 0, 0.1, 0.25, 0.5, 0.75, 1.0
- \(\alpha=0\): pure query-aware S1
- \(\alpha=1\): pure KVzip-VLM
- 중간 \(\alpha\): core–delta

20% 구간에서는 S1이 이미 포화에 가깝기 때문에, 판정의 중심은 1–10% 강압축 구간으로
둔다.

### 4.4 필수 비교군

1. Full KV
2. Random
3. Spatial-uniform
4. KVzip-VLM only
5. S1/query-only
6. Core–delta fixed split
7. 가능하면 answer-aware swap/search upper-bound probe

### 4.5 Phase A 채택 조건

중간 \(\alpha\)가 동일 byte budget에서 다음을 만족해야 한다.

- pure KVzip보다 현재 질문 성능이 높다.
- **pure S1보다도 일관되게 높다.**
- 두 모델 및 GUI/자연 이미지 중 하나의 특수 조건에만 의존하지 않는다.
- 화면 또는 이미지 단위 bootstrap CI가 개선 방향을 지지한다.
- 개선이 attention 측정 오류나 mask simulation artifact로 설명되지 않는다.

KVzip만 이기고 S1을 이기지 못하면 “generic score에 query score를 추가한 것”일 뿐,
새로운 방법의 핵심 가설은 성립하지 않는다.

---

## 5. 현재 결과가 말해 주는 것

현재 동결 결과의 5% visual-token 예산에서는 query-aware S1과 KVzip-VLM 사이에 큰
격차가 있다.

| 설정 | KVzip-VLM | S1 | S1 − KVzip |
|---|---:|---:|---:|
| Qwen2.5 × GUI | 0.542 | 0.930 | +38.8%p |
| Qwen3 × GUI | 0.430 | 0.910 | +48.0%p |

전체 표는 [FINDINGS.md](/root/research/heewon/VLM/docs/FINDINGS.md)의 D6에 기록되어 있다.
이 결과는 query branch가 반드시 필요하다는 것을 보여 주지만, core와 query의 결합이
S1보다 낫다는 것을 보여 주지는 않는다.

한편 5% 예산에서 S1 subset으로 시작한 swap search가 EM 0.83에서 0.92로 개선된 파일럿은
**질문을 알고도 attention top-k가 최적 subset은 아닐 수 있음**을 시사한다. 다만 그
개선분을 core score가 실제로 회수할 수 있는지는 별도 실험으로 확인해야 한다.

따라서 현재 증거의 정확한 해석은 다음과 같다.

1. generic-only 방식에는 큰 한계가 있다.
2. query-aware 신호는 매우 강하다.
3. query-aware attention보다 좋은 subset이 존재할 여지는 있다.
4. 그 여지를 core–delta가 채울 수 있는지는 아직 미검증이다.

---

## 6. Phase B — Phase A가 성공할 때만 lightweight NN 학습

### 6.1 무엇을 학습하는가

전체 VLM을 처음부터 다시 학습하지 않는다. VLM은 frozen 상태로 두고 작은 selector만
학습한다.

```text
Frozen VLM
   ├─ Core gate  g_c(I, K, V, position, modality)
   └─ Query gate g_q(q, K, V, position, modality)
                     ↓
          layer/head/token별 keep score
```

초기 구현은 linear/bilinear scorer 또는 작은 MLP로 충분하다. 큰 네트워크를 추가하면
KV를 줄여 얻는 이득보다 selector 비용이 커질 수 있다.

### 6.2 최종 score 예시

\[
s_{l,h,i}
=
g_{\mathrm{core}}(z_{l,h,i})
+
\gamma(I,q,l,h)\,
g_{\mathrm{query}}(z_{l,h,i},q_l),
\]

여기서 \(z_{l,h,i}\)는 다음과 같은 값싼 feature만 사용한다.

- compressed K/V feature 또는 norm
- token position과 visual spatial coordinate
- modality ID: image/text
- layer/head embedding
- pooled question hidden state
- 선택적으로 training-free core/query score

### 6.3 학습 목적

\[
\mathcal L
=
\mathcal L_{\mathrm{answer}}
+\lambda_{KD}
\operatorname{KL}
\left(
p_{\mathrm{full}}\;\Vert\;p_{\mathrm{masked}}
\right)
+\lambda_{rec}\mathcal L_{\mathrm{core-recon}}
+\lambda_B\mathcal L_{\mathrm{budget}}.
\]

- `answer loss`: 압축 cache로 정답을 유지
- `logit distillation`: full-cache 동작을 압축 cache가 모사
- `core reconstruction`: query branch만 남고 core가 붕괴하는 것을 방지
- `budget loss`: 실제 저장 byte가 목표를 넘지 않도록 제한

학습 시 정답과 full-cache teacher를 사용할 수 있다. 그러나 inference selector는 이미지와
현재 질문만 입력받아야 하며 정답 token을 사용해서는 안 된다.

### 6.4 왜 NN이 필요한가

NN은 아이디어의 필수 출발점이 아니라 다음 문제를 해결하기 위한 후속 단계다.

- 이미지와 질문마다 최적 \(\alpha\)를 다르게 배분
- layer/head마다 다른 cache budget을 자동 할당
- expensive S1 attention materialization을 값싼 gate로 근사
- visual과 text KV를 하나의 global budget에서 E2E 최적화

Phase A에서 fixed split이 S1을 이기지 못하면 NN 학습으로 넘어가지 않는다.

---

## 7. Visual-only 진단에서 multimodal E2E로 확장

현재 인프라는 visual token subset을 고른 뒤 모든 layer/head에 공통 mask를 적용하는
진단에 가깝다. 최종 목표는 다음 전체 공간에서 선택하는 것이다.

\[
\mathcal K
=
\{(l,h,i,m_i)\mid
m_i\in\{\text{vision},\text{text}\}\}.
\]

최종 method에서는:

- visual/text token을 동일한 byte 단위로 회계한다.
- 같은 token position도 layer/head에 따라 keep 여부가 달라질 수 있다.
- 짧은 현재 질문과 system anchor는 초기에는 보호하고, 이후 ablation에서 global budget에
  포함한다.
- GQA에서는 query head가 아니라 실제 KV-head 단위 비용을 계산한다.
- mask simulation뿐 아니라 ragged/variable-length KV cache로 실제 메모리 감소를 검증한다.

이 단계가 완료되어야 “visual token pruning”이 아니라 **VLM end-to-end KV cache
optimization**이라고 주장할 수 있다.

---

## 8. 시스템 관점에서의 주장 범위

실제 질문의 attention을 얻으려면 일반적으로 이미지와 질문을 full cache로 한 번
prefill해야 한다. 그 뒤 pruning하면 다음은 줄일 수 있다.

- answer decoding 중 읽는 active KV 수
- 질문 이후 유지되는 cache 크기
- 긴 답변 또는 multi-turn에서의 decoding latency

그러나 다음은 자동으로 줄어들지 않는다.

- 최초 image/question prefill FLOPs
- pruning 직전 peak KV memory
- full attention을 materialize하는 selector overhead

따라서 첫 시스템 주장은 **prefill 절감**이 아니라 **post-prefill active-cache 및 decoding
절감**으로 제한한다. peak memory까지 줄이려면 별도의 two-stage 구조가 필요하다.

```text
write/prefill time : 작은 core를 먼저 유지
read/query time    : cold/offloaded KV에서 query delta를 검색
decode time        : core + delta만 GPU active cache로 사용
```

이 two-tier 확장은 Phase A/B 이후 과제로 둔다. 처음부터 포함하면 selection 효과와 storage
system 효과를 구분하기 어렵다.

---

## 9. Novelty 경계

### 약한 형태

- S5와 S1 score를 단순 가중합
- 두 top-k 집합을 예산 제한 없이 union
- 모든 layer/head에 동일한 visual-token mask 사용
- accuracy만 보고 실제 cache memory와 latency를 측정하지 않음

이 경우 “KVzip + query-aware heuristic”으로 보일 가능성이 높다.

### 강한 형태

> **A layer/head/token-wise multimodal core–delta cache that jointly allocates a global KV budget
> between query-independent visual sufficiency and query-conditioned relevance.**

필요한 요소는 다음과 같다.

1. KVzip-derived core와 query-conditioned delta의 명시적 분해
2. 고정된 전체 byte budget
3. image/text를 포함한 per-layer/head/pair allocation
4. full-cache output을 보존하는 E2E objective
5. training-free fusion, query-only, KVzip-only에 대한 ablation
6. 실제 KV memory와 decoding latency 감소

---

## 10. 가장 먼저 할 구현

### Step 1. Training-free fixed-budget fusion

현재 `score_kvzip`과 `score_s1`을 재사용해 다음 함수를 추가한다.

```python
def core_delta_keep(core_score, query_score, keep_count, alpha):
    core_count = round(keep_count * alpha)
    core = topk(core_score, core_count)

    remaining = indices_not_in(core)
    query = topk(query_score[remaining], keep_count - len(core))
    return core | query
```

실제 구현에서는 `keep_count=0`, tie-breaking, NaN, 중복 index, exact byte budget을
명시적으로 처리한다.

### Step 2. 가장 싼 판정 실험

- 기존 동결 표본에서 5% budget 우선 실행
- \(\alpha\in\{0,0.1,0.25,0.5,0.75,1\}\)
- pure S1 대비 paired difference와 bootstrap CI 계산
- 유망한 중간 \(\alpha\)가 있을 때만 1/2/10/20% 및 다른 모델로 확장

### Step 3. 실패/성공 판정

- **실패**: 모든 중간 \(\alpha\)가 pure S1 이하  
  → single-query 목적에서 core branch를 기각하고 query gate 압축으로 축소한다.
- **부분 성공**: 평균은 상승하지만 특정 도메인·모델에만 국한  
  → core 유형 또는 evidence distribution에 따른 조건부 method로 제한한다.
- **성공**: 동일 budget에서 중간 \(\alpha\)가 S1과 KVzip을 반복적으로 상회  
  → lightweight per-layer/head gate 학습으로 진행한다.

---

## 11. 최종 연구 질문

> **Can a VLM retain full-cache behavior under aggressive KV budgets by decomposing multimodal
> cache utility into a reusable visual core and a query-conditioned residual, and allocating both
> jointly across layers and KV heads?**

이 질문에 답하기 위한 최소 증거 순서는 다음과 같다.

1. 같은 budget에서 core와 query score가 실제로 상보적인가?
2. 고정 결합보다 학습된 gate가 나은가?
3. gate 비용을 포함해 end-to-end latency가 감소하는가?
4. visual-only가 아니라 text를 포함한 전체 KV budget에서도 효과가 유지되는가?

1번이 통과하기 전에는 NN을 학습하지 않는다. 이 순서를 지키면 학습 비용을 낭비하지
않으면서, 단순 heuristic과 실제 E2E method 사이의 경계를 명확히 할 수 있다.
