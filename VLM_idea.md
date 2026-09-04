# Dual-Prefill Importance Union for VLM KV Compression

> **상태**: training-free method seed / 구현 완료 / 실험 전 가설  
> **개정일**: 2026-09-04  
> **구현**: `vlm_diagnosis/exps/core_delta_full_kv.py`,
> `vlm_diagnosis/exps/core_delta_write_read.py`

## 1. 한 문장 아이디어

> **이미지만 넣은 prefill과 이미지+기존 text prefix를 넣은 prefill에서 각각 중요하다고
> 판단된 KV pair를 고른 뒤, 같은 전체 byte 예산 안에서 합집합으로 사용한다.**

선택 단위가 \((l,h,i)\), 즉 layer·KV-head·token 위치일 때 최종 mask는 다음과 같다.

\[
M_{l,h}(I,t)
=
M^{\text{image}}_{l,h}(I)
\cup
M^{\text{joint}}_{l,h}(I,t),
\qquad |M|=B.
\]

- \(I\): 이미지
- \(t\): 현재 사용 중인 기존 text prefix(질문·instruction·assistant header 포함)
- \(B\): 실제 KV pair 수로 환산한 고정 예산
- \(M^{\text{image}}\): image-only prefill에서 중요한 항목
- \(M^{\text{joint}}\): image+text-prefix prefill에서 중요한 항목

K와 V는 따로 고르지 않고 같은 위치의 pair로 함께 보존한다.

## 2. 정확히 두 번의 signal prefill

### 2.1 Image-only prefill

입력은 모델의 정규 chat template에서 text가 시작되기 직전까지 자른다.

```text
[system / user control][vision_start][visual tokens][vision_end]
```

- 모델 forward는 이미지당 정확히 1회다.
- reconstruction instruction을 넣지 않는다.
- description을 생성하지 않는다.
- visual-token 행들이 앞선 prefix KV 열에 준 attention 평균을
  \((layer, KV-head, position)\)별 image importance로 쓴다.
- image-only 입력에 없는 뒤쪽 text 위치는 선택 불가능한 좌표로 명시한다. 단순히 낮은
  점수로만 두지 않아, image branch가 text token을 잘못 고르는 일을 막는다.

구현에서는 processor가 올바른 multimodal placeholder를 만들도록 임시 text가 있는 template를
구성하되, 모델 forward 전에 `vision_end`까지만 자른다. 따라서 임시 text와 assistant header는
image-only prefill에 들어가지 않는다.

### 2.2 Image + existing text-prefix prefill

```text
[같은 image prefix][현재 질문/instruction][assistant header]
```

- 현재 text prefix당 정확히 1회 prefill한다.
- text-prefix 행들이 전체 prompt KV 열에 준 attention 평균을 joint importance로 쓴다.
- 정답 token과 생성된 answer token은 selector에 사용하지 않는다.
- FULL 기준 답 생성은 이 prefill 뒤의 decode이며, selection용 prefill을 한 번 더 만들지 않는다.

두 점수 모두 같은 attention 집계법을 쓰므로, 기존의 `KVzip max + S1 mean`처럼 scorer 정의가
서로 달라 생기는 교란을 제거한다.

## 3. 두 중요 집합을 모으는 법

전체 예산을 \(B\), image-only 몫을 \(\alpha\)라고 한다.

\[
B_I=\operatorname{round}(\alpha B),
\qquad B_J=B-B_I.
\]

선택 절차는 다음과 같다.

1. image-only 순위에서 상위 \(B_I\)개를 고른다.
2. joint-prefill 순위에서 상위 \(B_J\)개를 **독립적으로** 고른다.
3. 두 집합을 합치고 겹친 항목을 한 번만 센다.
4. 중복 때문에 비는 자리는 joint-prefill 순위의 다음 항목으로 채운다.
5. 최종 항목 수가 정확히 \(B\)인지 검증한다.

```python
def dual_prefill_union_keep(image_score, joint_score, B, image_quota):
    image = stable_topk(image_score, image_quota, eligible=image_prefix)
    joint = stable_topk(joint_score, B - image_quota)
    keep = image | joint
    keep |= next_joint_items_excluding(keep, B - len(keep))
    assert len(keep) == B
    return keep
```

중요한 불변식은 다음과 같다.

- 합집합 때문에 예산이 늘어나지 않는다.
- 중복 때문에 실제 사용량이 예산보다 작아지지도 않는다.
- 동점은 낮은 flat index를 먼저 골라 결정적으로 처리한다.
- NaN/Inf는 최하위로 보내되 exact budget이 필요하면 후보에서 사라지지는 않는다.
- `alpha=0`은 joint-prefill only, `alpha=1`은 image-only endpoint다.
- sink/text 강제 보존을 켠 경우에도 그 항목은 예산 안에 포함한다. 강제 항목이 예산보다
  많으면 조용히 초과하지 않고 오류로 중단한다.

## 4. 점수만 합치고 KV 값은 섞지 않는다

두 forward에서 나온 K/V tensor 자체를 이어 붙이는 방식은 사용하지 않는다. 두 prefill은
importance mask를 얻는 데만 쓰고, 실제로 보존할 K/V 값은 **정규 image+text joint prefill
cache**에서 가져온다.

```text
image-only prefill ── importance mask ─┐
                                      ├─ union/dedup ─ selected joint-cache K/V
image+text prefill ─ importance mask ─┘
```

Qwen처럼 text가 visual token 뒤에 놓이면 causal attention상 앞쪽 visual KV는 동일해야 한다.
다만 sequence length에 따른 fp16 kernel 반올림 차이는 생길 수 있으므로, 서로 다른 forward의
실제 KV 값을 섞지 않는 편이 안전하다. text가 image 앞에 오는 다른 모델에도 같은 원칙을
적용하면 position/context 불일치가 없다.

image-only와 joint 입력의 공유 prefix token ID가 정확히 같은지도 매 표본 확인한다. 길이나
token ID가 다르면 ordinal이 같아 보이더라도 자동 remap하지 않고 즉시 실패시킨다.

## 5. 후보 공간과 예산

전체 KV 판의 후보는 다음이다.

\[
\mathcal K
=
\{(l,h,i)\mid i\in
\text{system, vision boundary, visual, text, assistant header}\}.
\]

- head granularity: layer·KV-head마다 서로 다른 token 위치를 선택한다.
- token granularity: 모든 layer·KV-head에서 같은 token mask를 공유한다.
- GQA 모델의 비용은 query head가 아니라 실제 KV head 수로 센다.
- pair 하나의 payload는 `K + V = 2 × head_dim × element_bytes`다.
- 결과에는 payload byte와 mask index byte를 따로 기록한다.
- image-only branch는 공유 image prefix만 선택할 수 있다.
- joint branch는 image와 text를 포함한 전체 prompt를 선택할 수 있다.

## 6. 두 실행 모드

### 6.1 고정 전체 예산 sweep

`core_delta_full_kv.py`는 전체 prompt KV의 일정 비율 \(B\) 안에서 image/joint 몫을
`alpha`로 나눈다.

```bash
python -m vlm_diagnosis.exps.core_delta_full_kv \
  --budgets 0.01,0.02,0.05,0.1 \
  --alphas 0,0.25,0.5,0.75,1 \
  --granularity head,token
```

기본 `row-start=generation`은 full joint prefill 뒤 cache를 줄여 decode에서만 선택 mask를
적용한다. `row-start=question`은 text가 image를 읽기 전부터 mask를 적용하는 더 엄격한
진단 조건이다.

### 6.2 쓰기/읽기 격자

`core_delta_write_read.py`는 image-only prefill에서 크기 \(C\)의 core를 먼저 두고, text가
도착하면 joint prefill 순위에서 크기 \(D\)의 delta를 더한다.

```bash
python -m vlm_diagnosis.exps.core_delta_write_read \
  --core-sizes 0,0.025,0.05,0.1 \
  --delta-sizes 0,0.01,0.025,0.05
```

delta는 core와 겹치지 않는 joint 순위의 다음 항목으로 채우므로 최종 크기는 정확히
\(C+D\)다. cold budget \(B<1\)을 쓰면 image-only 순위 상위 \(B\) 밖의 KV는 삭제된 것으로
간주하고 delta도 그 범위 안에서만 가져온다.

## 7. 필수 비교군과 판정

동일한 byte 예산에서 다음을 비교한다.

1. Full KV
2. Random
3. Random + sink protection
4. Spatial/token uniform
5. Joint-prefill only (`alpha=0`)
6. Image-only (`alpha=1`)
7. Dual-prefill union (`0<alpha<1`)

중간 alpha를 방법으로 채택하려면 다음을 모두 만족해야 한다.

- joint-only보다 높다.
- image-only보다 높다.
- paired bootstrap CI가 개선 방향을 지지한다.
- 같은 방향이 두 모델 또는 두 도메인에서 재현된다.
- 개선이 강제 sink 보존이나 더 큰 실제 byte 사용량으로 설명되지 않는다.

모든 중간 alpha가 joint-only 이하라면, image-only signal은 단일 질문 목적에서 추가 가치가
없는 것으로 판정한다. 이 경우 selector를 복잡하게 학습하는 Phase B로 넘어가지 않는다.

## 8. 시스템 주장 범위

현재 질문에 대한 joint importance를 정확히 얻으려면 full image+text prefill이 한 번 필요하다.
image-only signal까지 더했으므로 selector용 forward는 총 두 번이다. 따라서 현재 구현이 직접
줄이는 것은 다음이다.

- prefill 뒤 GPU에 유지하는 active KV 크기
- answer decoding 중 읽는 KV 수
- 긴 답변 또는 후속 turn에서의 decoding 비용

다음은 줄였다고 주장하지 않는다.

- 최초 prefill FLOPs
- selection 직전 peak KV memory
- 두 signal prefill 자체의 지연
- head-ragged mask를 실제 packed cache로 만들기 전의 물리 메모리

실제 시스템 이득은 selector 2회 비용까지 포함한 TTFT/decode benchmark로 따로 검증해야 한다.
현재 head 단위 실험은 mask simulation이고, token 공통 mask만 물리 cache slicing으로 직접
검증할 수 있다.

## 9. 구현 위치와 검증 항목

- `vlm_diagnosis/core/core_delta.py`
  - `dual_prefill_union_keep`: 독립 top-k, union, 중복 제거, joint backfill
- `vlm_diagnosis/core/kv_select.py`
  - `select_dual_prefill_triples`: layer·KV-head·position 선택
  - `select_dual_prefill_tokens`: 공통 token mask 선택
- `vlm_diagnosis/exps/core_delta_full_kv.py`
  - `image_prefill_stats`: 생성 없는 image-only prefill 1회
  - joint prefill capture와 전체 KV sweep
- `vlm_diagnosis/exps/core_delta_write_read.py`
  - image-only core + joint text-conditioned delta 격자

단위 테스트는 다음을 고정한다.

- overlap 없음 / 일부 overlap / 완전 overlap
- overlap 후 exact-budget backfill
- alpha 양 끝점
- 0·음수·전체 초과 예산
- 결정적 tie-breaking과 NaN/Inf
- image-only 후보가 text suffix를 선택하지 못함
- head/token granularity와 강제 sink 회계
- image score의 full-prompt 좌표 정렬

## 10. 다음 실험 순서

1. 1개 표본 smoke로 두 prefix ID 정렬, score shape, mask 적용을 확인한다.
2. 5% 예산에서 `alpha={0,.25,.5,.75,1}`을 GUI와 자연 이미지에 실행한다.
3. joint-only 대비 paired EM/ANLS와 이미지 단위 bootstrap CI를 계산한다.
4. 중간 alpha가 이길 때만 1/2/10/20%와 두 번째 모델로 확장한다.
5. 마지막에 실제 packed/token-sliced KV와 latency를 측정한다.

최종 연구 질문은 다음이다.

> **Does the union of image-intrinsic importance from an image-only prefill and
> text-conditioned importance from a joint prefill preserve VLM behavior better than either
> prefill ranking alone under the same end-to-end KV byte budget?**
