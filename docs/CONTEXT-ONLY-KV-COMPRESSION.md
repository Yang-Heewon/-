# Context-only KV 압축: 구현·실험 실행 명세

작성일: 2026-09-07

상태: **구현 계획 문서. 아래 신규 모듈·CLI·실험 결과가 이미 구현되었다는 뜻이 아니다.**
기존 코드 확인 결과와 앞으로 할 작업을 구분한다. 이 문서 작성으로 실행 코드나 기존 실험 결과를 변경하지 않는다.

## 1. 연구 목표와 범위

> 미래 질문을 보지 않고, 최초 context prefill에서 이미 계산되는 MLP·hidden-state 신호만으로
> 보존할 KV를 저렴하게 선택하여, 압축 준비 비용과 저장량을 줄이면서 이후 질문의 성능을 보존할 수 있는가?

여기서 중요도는 보편적인 의미적 중요성의 정답이 아니라 **제한된 KV 예산을 배분하는 점수**다.
MLP dynamics는 후보 신호이며, 큰 변화량이 높은 KV 보존 가치를 뜻한다는 것은 검증 대상이다.

### 1.1 고정 조건

- 모델은 frozen, `eval()` 및 inference/no-grad 모드. 별도 NN/gate 학습 없음.
- 후보 압축기의 입력은 현재 context뿐. 실제 첫 질문 Q1, 미래 질문, 정답을 입력하지 않는다.
- 압축에 필요한 context prefill은 1회. 추가 reconstruction, 설명 생성, context replay를 하지 않는다.
- 선택 단위는 `(layer, KV head, logical token)`의 **K 벡터와 V 벡터 한 쌍**.
- 선택하지 않은 K/V는 실제 삭제. 모든 head에 공통인 token mask나 단순 attention masking으로 대체하지 않는다.
- 첫 연구는 정적 압축: 한 번 확정한 context cache로 여러 독립 질문을 평가한다.
- 기존 recurrent state 갱신, 비전 특화 점수, 학습된 gate는 첫 구현에서 제외한다.

평가용 FULL 실행, 정답 NLL 계산, 인과 개입, KVzip 기준 실행은 별도로 가능하다.
**배포 후보의 단일 prefill 제약과 연구 검증에 필요한 추가 실행을 혼동하지 않는다.**

### 1.2 성공과 실패의 해석

- 동일 KV 예산에서 성능이 비슷하고 압축 준비 비용이 더 낮으면 효율 측면의 기여 후보다.
- 동일 성능을 더 적은 실제 저장량으로 유지하면 압축 품질의 기여 후보다.
- D가 R·norm보다 낫지 않으면 dynamics의 추가 가치는 미확인이다.
- KVzip과 점수 상관이 높아도 실제 삭제 성능이 나쁘면 성공이 아니다.
- 작은 모델·소수 context의 결과만으로 novelty나 VLM 범용성을 주장하지 않는다.
- 한 번의 full prefill 뒤 삭제하는 설계는 최초 full KV 생성 비용이나 초기 peak를 제거하지 않는다.

## 2. 현재 저장소에서 재사용할 부분

아래 상태는 문서 작성 시점의 코드 확인 결과다. 구현 시작 시 다시 확인한다.

| 기존 구성 | 확인된 내용 | 이번 작업에서의 처리 |
|---|---|---|
| [MLPWriteCapture](../vlm_diagnosis/exps/mlp_signal_probe.py) | MLP 출력 norm / 정규화 전 residual norm 수집 | 정확한 hook 위치 재사용, 독립 collector와 테스트 추가 |
| 같은 파일의 `shift_to_kv` | R을 한 층 이동시켜 head들에 복제 | D와 다름. 기존 mapping을 새 기본값으로 몰래 복사하지 않음 |
| 같은 파일의 `main` | score 필터 이전에 reconstruction 생성·replay를 실행 | 새 single-prefill runner와 분리 |
| [kv_select](../vlm_diagnosis/core/kv_select.py) | 기존 probe 평가에 head별 attention mask 사용 | 품질 참조용만 사용. 실제 압축·메모리 증거로 사용하지 않음 |
| [ragged_kv](../vlm_diagnosis/core/ragged_kv.py) | head별 owning tensor, 물리 삭제, logical ID 보존 | 저장 구조 재사용, 작은 text Qwen용 attention 연결 추가 |
| [session_adapters](../vlm_diagnosis/core/session_adapters.py) | 현재 Qwen VL 단일 이미지 context 어댑터 | text-only context 어댑터는 새 작업 |
| [pair_importance](../vlm_diagnosis/core/pair_importance.py) | 전역 pair 예산·보호분 검사·매핑 | 계약 재사용 가능. 정적 압축에 recurrent state를 의무적으로 넣지 않음 |
| [mlp_signal_analysis](../vlm_diagnosis/scripts/mlp_signal_analysis.py) | 이미지 단위 bootstrap, FULL 조건부 결과 | 통계 아이디어 재사용, 새 schema의 엄격한 분석기 작성 |

기존 probe의 일부 score는 query-agnostic이다. 모든 질문 행에서 삭제 열을 차단하는 masked 평가를
그 자체로 질문 누수라고 부르지는 않는다. 다만 평가마다 context를 다시 처리하고 dense cache를 유지하므로,
**한 번 만든 압축 cache의 재사용·실제 저장량 감소를 검증한 경로는 아니다.**

재사용 시 주의할 확인 사항:

- 기존 `select_triples`는 보호 pair 수가 목표 예산보다 클 때 예산 초과를 거부하지 않는다.
- 기존 double-argsort rank는 동점을 평균 순위로 처리하지 않는다. MLP head 복제 점수에 가짜 차이를 만들 수 있다.
- 기존 분석기는 metadata를 버리고 일부 오류·누락을 건너뛴다. 새 결과를 그대로 합치지 않는다.
- 기존 MLP collector는 같은 layer의 재호출을 덮어쓸 수 있다. 새 collector는 호출 횟수를 검증한다.
- 기존 실험과 출력은 보존한다. 공유 함수를 바꾸면 기존 회귀 테스트도 실행한다.

## 3. 시스템 경계와 cache 수명

```text
ContextInput                    평가 질문·정답은 별도 보관
    │                                      │
    ▼                                      │
context prefill 1회 + 통계 수집             │
    │                                      │
점수 계산 → 전역 B개 선택 → 실제 삭제        │
    │                                      │
CompressedMemory 생성                      │
원본 full cache/seed 참조 해제              │
    │                                      │
    ├─ 독립 clone → Q1 suffix → 답변/NLL ◀───┤
    ├─ 독립 clone → Q2 suffix → 답변/NLL ◀───┤
    └─ 독립 clone → Q3 suffix → 답변/NLL ◀───┘
```

제안 API 계약이다. 기존 API가 이미 이 형태라는 뜻은 아니다.

```python
memory, build_report = compress_context(
    model, context, method, keep_ratio, selection_seed
)  # question/answer를 받지 않음

for question in evaluation_questions:
    generation_branch = memory.clone_owned()
    prediction = answer_from_cache(model, generation_branch, question.text)
    del generation_branch

    nll_branch = memory.clone_owned()
    nll = answer_nll(model, nll_branch, question.text, question.answer_token_ids)
    del nll_branch
```

- dataset row 전체를 scorer에 전달하지 않는다. context와 evaluation 항목을 분리한다.
- 같은 context에서 평가 질문만 바꾸어도 score·선택 ID가 같아야 한다.
- master memory는 평가로 수정되지 않는다. 질문 처리 순서를 바꿔도 질문별 결과가 같아야 한다.
- 생성과 teacher-forced NLL도 독립 분기에서 실행하여 생성 이력이 섞이지 않게 한다.
- 남은 토큰 위치를 `0..B-1`로 재번호화하지 않는다. logical clock과 RoPE 위치를 보존한다.
- 압축 이후 context를 다시 token-forward하거나 원본 이미지 encoder를 재실행하지 않는다.
- full prefix에서 얻은 마지막 logits로 답변 첫 토큰을 생성하지 않는다. 새 질문 suffix를 먼저 처리한다.
- 원본 context의 텍스트·이미지를 평가 harness가 보유할 수는 있지만, 압축된 응답 경로의 입력/복구 수단으로 쓰지 않는다.
- FULL 기준은 별도 cache·조건이다. 방법별 메모리 측정 중 FULL cache를 함께 상주시켜 숨은 백업으로 만들지 않는다.

현재 정적 실험의 예산은 **context KV**에 적용한다. 질문·답변의 새 KV는 분기 안에서 추가되며
모든 방법에 같은 정책을 적용한다. 질문 처리 중 총 resident bytes도 별도로 기록한다.

## 4. 수식·hook·점수 계약

### 4.1 pre-norm decoder의 관측 위치

초기 대상은 일반적인 sequential pre-norm text Qwen decoder다.
다른 구조나 Qwen3-VL DeepStack에 그대로 일반화하지 않는다.

```text
x[l]    = decoder block 입력 residual
a[l]    = Attention(Norm1(x[l]))
r[l]    = x[l] + a[l]              # MLP 직전, 정규화 전 residual
z[l]    = Norm2(r[l])              # MLP 모듈에 실제 들어가는 입력
m[l]    = MLP(z[l])
x[l+1]  = r[l] + m[l]              # block 출력, 모델 최종 norm 이전
```

| 값 | hook 위치 | 역할 |
|---|---|---|
| x | decoder layer pre-hook | hidden-change 진단 |
| r | `post_attention_layernorm` pre-hook | R의 분모 |
| z, m | MLP 입력/출력 | module 경계 확인, MLP 출력 통계 |
| x_next | decoder layer output hook | residual 관계·hidden-change 확인 |

필수 assertion: `x_next ≈ r + m`. 허용오차는 dtype/backend별로 고정하고 최대 오차를 기록한다.
`output_hidden_states`의 마지막 원소는 최종 norm 적용 여부를 확인하지 않고 block 출력으로 쓰지 않는다.

### 4.2 기본 통계

배치 크기 1, layer 수 L, context 길이 T, hidden 차원 d로 시작한다.

```text
MLP_norm[l,i] = ||m[l,i]||_2
Residual_norm[l,i] = ||r[l,i]||_2
R[l,i] = MLP_norm[l,i] / (Residual_norm[l,i] + eps)
D[l-1,i] = abs(R[l,i] - R[l-1,i]),  l = 1, ..., L-1

R.shape == (L,T)
D.shape == (L-1,T)
eps = 1e-6  # 초기 실험 고정값, metadata에 기록
```

norm과 집계는 FP32. 기존 collector의 `clamp(min=eps)`와 새 `+eps` 정의의 차이는 명시한다.
기존 결과와 완전히 같은 score라고 주장하지 않는다.

추가 대조군은 block hidden relative change와 cosine change다.
`||x_next-x||/(||x||+eps)`는 attention+MLP 변화이며 MLP-only 변화와 구분한다.

토큰 표에는 원래 token index, token ID, 읽을 수 있는 조각, special 여부,
MLP norm mean, R mean/max, D mean/max/top-3 mean, R의 표준편차를 기록한다.
top-k는 `min(k,L-1)`을 사용하며, L<2인 D 조건은 명시적으로 거부한다.

- D mean과 D sum은 같은 L에서 순위가 같다. 독립 방법 수로 세지 않는다.
- D max와 top-1도 같은 방법이다.
- D는 크기 비율의 변화이지 벡터 방향 변화 전체가 아니다.
- 큰 D는 상승과 하강을 모두 포함하며 residual 분모의 변화로도 생긴다.
- raw heatmap과 필요 시 표시용 정규화 heatmap을 구분한다. 표시용 정규화를 selector에 몰래 적용하지 않는다.

### 4.3 layer/token 신호를 KV-pair 점수로 바꾸는 규칙

최초 구현은 mapping 이름까지 condition에 포함한다.

```text
mlp_norm_same[l,h,i] = MLP_norm[l,i]
r_same[l,h,i]        = R[l,i]
d_same_zero0[0,h,i]  = 0
d_same_zero0[l,h,i]  = D[l-1,i], l >= 1
```

이는 **head-shared, same-layer 예측 proxy**이지 head별 MLP 기여도나 인과적 KV attribution이 아니다.
K/V는 같은 layer의 MLP보다 먼저 만들어지므로, MLP 관측값과 KV 삭제 효과 사이의 관계는 실험으로 판단한다.

`d_same_zero0`의 첫 layer 0은 경계 처리일 뿐 중요도가 0이라는 증거가 아니다.
layer별 보존량을 필수 기록하고, 첫 layer 선택을 모든 방법에서 동일하게 맞춘 별도 대조 실험으로
경계 규칙이 성능 차이를 만드는지 확인한다. 이 대조군은 전역 무할당량 방법과 다른 조건으로 표시한다.

경계 대조군의 공통 선택 규칙: 첫 layer의 총 pair 수 N0, 보호 수 P0, 다른 layer의 보호 수 Pother에 대해
`B0 = max(P0, min(N0, B - Pother, int(round(B / L))))`로 고정한다.
layer 0은 보호분 + context ID와 고정 boundary seed로 뽑은 비보호 pair를 모든 방법에서 공유한다.
나머지 layer에서 보호분을 포함한 `B-B0`개를 방법별로 선택한다. 경계 대조군은 별도 condition으로 보고한다.

기존 `shift_to_kv` 같은 이전-layer mapping은 후속 ablation이다. 구현 시 첫/마지막 layer 처리와
사용되지 않는 관측값을 명시한다. 여러 mapping 중 평가 세트에서 잘 나온 것을 사후 선택하지 않는다.

K/V norm은 head별 후보 신호다. MLP와 결합하기 전 각각을 독립 평가한다.
정규화·곱·가중합은 개발 세트에서만 선택하고 식·축·계수를 모두 기록한다.
특히 layer별 정규화는 전역 예산의 layer 배분을 바꾸므로 별도 ablation으로 취급한다.

### 4.4 수집 비용

- `stats` 모드: forward 중 필요한 norm/scalar만 유지. tensor당 d 차원을 즉시 축약한다.
- `debug` 모드: 짧은 context에서만 raw activation을 저장해 hook·수식을 확인한다.
- MLP-only 경로는 QKCapture, 전체 attention 재계산, reconstruction import/실행을 필요로 하지 않는다.
- layer마다 `.cpu()`가 유발하는 동기화·복사도 실제 비용이다. scalar 수집과 전송 시점을 프로파일한다.
- hook 호출은 layer마다 정확히 한 번. 재사용·중복 forward는 조용히 덮어쓰지 않고 실패시킨다.
- hook은 예외가 나도 제거하고 모델 동작을 복구한다.

## 5. 전역 예산과 실제 삭제 계약

L, Hkv, T는 모델 config와 실제 prefix에서 얻는다. head dimension/dtype이 균일한 초기 모델에 대해:

```text
N = L * Hkv * T
B = int(round(keep_ratio * N))
bytes_per_pair = 2 * head_dim * element_size
context_kv_bytes = B * bytes_per_pair
```

- `keep_ratio`는 (0,1]만 허용. `round` 규칙과 계산된 B를 기록한다.
- 첫 pilot의 보호 정책은 prefix 첫 `min(4,T)` 위치와 실제 prefix 안의 tokenizer special 위치의 합집합.
  각 head의 보호 pair 모두 B에 포함한다. 미래 질문 special token은 초기 예산에 포함하지 않는다.
- `B < protected_pairs`면 오류. 예산을 자동으로 늘리거나 작은 실행을 몰래 제외하지 않는다.
- 보호분 외에는 전역 top-B. 기본 조건에 layer/head/modality별 quota를 넣지 않는다.
- 동점은 고정 seed의 pair별 독립 순열로 해소한다. score에 작은 잡음을 더해 비동점 순서까지 바꾸지 않는다.
- 통계용 rank는 동점에 평균 순위를 부여한다. 선택용 tie-break와 통계 rank를 분리한다.
- 보호분 제외 후보에서 선택하고, 최종 `keep.sum() == B` 및 보호분 유지 여부를 검사한다.
- pair 수, head별 길이, logical ID, 실제 tensor storage를 함께 검사한다.
- 원본 dense tensor의 view만 보관하지 않는다. 생존 K/V는 축소된 owning storage여야 한다.
- 원본 full seed/cache, capture 안의 K/V, 임시 변수의 참조도 해제한다. CPU backup 금지.
- 정적 선택이 끝나면 불필요한 score state도 해제한다. 남겨 두는 부가 상태는 bytes에 포함한다.

## 6. 구현·실험 순서와 완료 기준

### 단계 1 — context-only FULL 평가기

구현:

- [ ] 작은 text decoder 로더: `Qwen/Qwen2.5-0.5B-Instruct`, batch=1, 모델 revision·dtype·backend 기록.
- [ ] 모델·tokenizer·라이브러리 실제 버전과 GPU 가용 상태 확인. 공유 GPU 작업을 방해하지 않음.
- [ ] context와 질문의 append-compatible token 경계 정의. 전체 encoding과 prefix+suffix ID가 정확히 일치하는지 검사.
- [ ] `use_cache=True`로 context를 1회 prefill하고, 질문 suffix만 처리하는 FULL cache 평가기.
- [ ] 질문별 독립 clone, 생성과 NLL 분기 분리, 정답 토큰만 loss에 포함.
- [ ] raw context·질문·정답을 구분하는 manifest와 고정 seed 합성 데이터 생성기.

실험:

- hook 확인용 1개 짧은 context → 실행 확인용 32 context × 4 질문.
- 이름·장소·코드·날짜를 무작위로 배정하고 위치를 섞어 사전 지식·고정 위치 shortcut을 줄인다.
- 압축 전 FULL이 과제를 수행하는지 먼저 확인. 필요하면 개발 데이터 난이도나 모델 크기를 조정한다.

완료 기준:

- [ ] 일반적인 FULL `[context + question]` forward와 분리 실행의 질문 logits가 허용오차 안에서 일치.
- [ ] FULL `[context + question + gold]`와 cached-context 이후 `[question + gold]`의 정답 token별 log-probability·평균 NLL 일치.
- [ ] 첫 답변 token의 logits source가 마지막 질문 suffix 위치임을 확인.
- [ ] 질문 순서 변경 시 결과 불변, master cache ID·내용 불변.
- [ ] FULL 실패·분모·예외를 모두 기록. FULL 성공 사례만 몰래 남기지 않음.

### 단계 2 — 단일 prefill MLP collector와 관측 결과

구현:

- [ ] §4 수집기를 기존 VLM probe와 분리해 추가. 기존 probe는 우선 그대로 보존.
- [ ] R, D, raw MLP norm, residual norm 계산 및 token 표·heatmap 출력.
- [ ] collector 없는 실행과 있는 실행의 KV·logits parity 테스트.
- [ ] norm 수식 직접 계산, shape, finite, layer 호출 횟수, 예외 후 hook 제거 테스트.
- [ ] constant R → D=0, 한 번의 peak → 상승·하강 두 변화 검출 테스트.

완료 기준:

- [ ] 질문·정답·reconstruction 없이 context prefill 정확히 1회에서 통계 생성.
- [ ] 신호를 재계산하는 추가 모델 forward 없음. stats/debug 비용 분리.
- [ ] 관측 결과를 해석할 수 있는 작은 표와 heatmap 확보. 아직 압축 우위를 주장하지 않음.

**첫 구현 묶음은 단계 1–2까지다. 결과·테스트를 보고한 뒤 단계 3으로 진행한다.**

### 단계 3 — pair selector와 물리 cache 연결

구현:

- [ ] text Qwen attention을 기존 ragged 저장 구조에 연결. VL 전용 type guard를 무작정 제거하지 않음.
- [ ] 정적 selector 추가: 신호, 보호 mask, 전역 B, tie seed → head별 생존 logical ID.
- [ ] §4.3의 명시적 score mapping과 §5의 exact budget 적용.
- [ ] 생존 storage만 소유하는 compressed memory와 `clone_owned()` 구현.
- [ ] debug dense-mask 기준과 물리 ragged 실행을 동일한 survivor ID로 비교.

완료 기준:

- [ ] keep=100%에서 단계 1 FULL과 KV·logits parity.
- [ ] head별 서로 다른 길이·동점·보호 예산 초과·가능한 empty-head 경계 테스트.
- [ ] 실제 storage 감소와 원본 full cache 참조 해제 확인.
- [ ] 삭제한 pair는 복구할 수 없고 logical position은 그대로 유지.
- [ ] 기존 VL ragged·pair/session 회귀 테스트 통과.

### 단계 4 — 작은 삭제 민감도 실험

설정: **80% 유지 = 20% 삭제**. 같은 context, 삭제 pair 수, 보호 정책을 사용한다.

- [ ] 낮은 score부터 삭제, 높은 score부터 삭제, random 삭제 비교.
- [ ] random은 최소 5개 고정 seed. 모든 조건에서 같은 질문과 generation 설정 사용.
- [ ] raw MLP norm, R, D, K/V norm을 독립 비교.
- [ ] D에 대해 R 표준편차 및 layer-order-shuffled D 대조군 추가.
- [ ] layer-order shuffle은 수집한 R의 layer 행 순서만 바꿈. 모델 layer 순서를 바꾸지 않음.
- [ ] primary global 선택과 별개로 layer별 삭제 수를 맞춘 대조군을 두어 단순 layer 배분 효과 확인.

shuffle은 고정 seed의 순열 pi에 대해 `D_shuffle[l-1,i] = abs(R[pi[l],i]-R[pi[l-1],i])`로 정의한다.
pair 점수는 원래 cache layer l에 배치하며 layer 0은 동일하게 0으로 둔다. 순열 자체를 로그에 기록한다.
이 조건은 인접성과 실제 cache layer에 대한 score 배치를 함께 바꾸므로, 단독으로 인접성만의 효과를 증명하지 않는다.
보완 대조군은 anchor layer l을 유지하고 `abs(R[l,i]-R[j_l,i])`를 사용한다.
`j_l != l`인 비교 layer를 고정 seed로 선택하고 매핑을 기록하며, layer 0 경계 규칙은 동일하게 유지한다.
layer별 삭제 수를 맞춘 조건에서는 각 layer의 eligible pair를 해당 방법의 순위 또는 고정 random seed로 고른다.

관찰:

```text
가설이 지지되는 방향(보장 아님):
낮은 점수 삭제의 평균 손실 < random 삭제의 평균 손실 < 높은 점수 삭제의 평균 손실
```

정답률과 answer NLL 증가량을 함께 본다. 높은 score 삭제가 더 해롭다는 결과 하나만으로
작은 budget에서 좋은 조합을 남긴다는 결론을 내리지 않는다. 중복 정보·상호작용이 있을 수 있다.

진행 판단:

- [ ] D가 단순 norm/R 대비 이득을 보이는지 paired 결과로 확인.
- [ ] 순서를 섞은 D와 차이가 없으면 인접 layer dynamics의 추가 가치를 미확인으로 보고.
- [ ] 차이가 불명확하면 불확실성을 보고하고 표본·가설을 개발 세트에서 재검토. VLM 확장을 성공 근거로 대신하지 않음.

MLP zero ablation은 후속 기전 진단이다. KV 삭제와 같은 개입으로 취급하지 않는다.
선택적으로 context의 마지막 layer MLP만 zero하고 prefix logits를 버린 뒤 새 질문을 처리하는
negative control을 둔다. 이 표준 sequential text 모델에서는 이미 작성된 context KV가 바뀌지 않아야 한다.

### 단계 5 — budget sweep와 비용–성능 비교

- [ ] 유지율 `1.0, 0.8, 0.5, 0.2, 0.1`. 0.05는 이후 강한 압축 조건으로 추가.
- [ ] FULL, random, recent, K/V norm, MLP norm, R, D 비교.
- [ ] 비용이 명시된 context-only attention과 원 방식에 충실한 text KVzip 추가.
- [ ] 공식 KVzip 지원 모델·backend 제약을 확인. 지원되지 않는 작은 모델에 조용히 다른 방법을 대신 넣지 않음.
- [ ] 동일 checkpoint, prefix, 질문, 실제 context budget, 보호 정책, decode 설정으로 비교.
- [ ] 효과 비교는 공통 backend, native 구현 속도 비교는 backend 차이를 별도 표시.
- [ ] 개발 context에서 score/정규화/mapping을 선택·고정한 뒤 별도 context에서 평가.
- [ ] 방법별 독립 비용 실행 및 §8의 통계·메모리 회계 수행.

기존 이미지 설명문 생성 기반 `kvzip` proxy를 원 논문 KVzip과 동일하다고 표기하지 않는다.
재구성 기반 baseline은 명시적으로 선택한 별도 실행에만 포함한다.

### 단계 6 — VLM, 이후 recurrent session

- [ ] 정적 text 결과 이후 image-only 또는 image+context text 입력으로 확장. 평가 Q1은 제외.
- [ ] 같은 이미지에 여러 질문을 독립 평가. 데이터 분할은 질문 단위가 아니라 이미지/context 단위.
- [ ] decoder MLP 신호와 비전 encoder 신호를 구분하고 모달리티별 저장량·성능 기록.
- [ ] 공간 coverage 등 비전 후보는 한 번에 하나씩 추가하여 순수 추가 효과 측정.
- [ ] 일반 text에서 실패한 D의 VLM 실험은 새 모달리티 가설로 표시하고 성공을 전제하지 않음.
- [ ] 마지막에 기존 recurrent framework 연결. static 압축과 세션 적응 효과를 별도 조건으로 비교.

세션에서 이미 완료된 질문·답변은 현재 context에 포함할 수 있다. 그러나 아직 도착하지 않은
다음 질문을 압축에 사용하지 않는다. 현재 질문 처리 후 갱신한 효과를 질문 도착 전 압축 효과로 보고하지 않는다.

## 7. 데이터와 평가 규칙

권장 시작 규모는 실행 확인용이지 통계적 검정력을 보장하는 수치가 아니다.

| 단계 | 데이터 | 목적 |
|---|---|---|
| hook 확인 | 짧은 context 1개, 128–512 token 범위 | 수식·shape·경계 확인 |
| pilot | 개발용 32 context × 4 질문 | FULL 수행 여부, 삭제 실험 동작 |
| 확대 screening | 별도 128개 이상 context × 4–5 질문, 복수 길이 | 효과 크기·변동성 추정 |
| 최종 검증 | 개발/시험 분리, 과제·길이·모델 확대 | 추정된 변동성에 따라 필요한 표본 결정 |

- 관계 조회, 여러 항목 구분, 분산된 정보 결합 등으로 질문 유형을 나눈다.
- 단순한 synthetic 사실만으로 일반 문서 이해를 주장하지 않는다. 이후 자연 문서도 추가한다.
- 최종 시험 데이터의 질문·정답은 점수 선택·hyperparameter 조정에 사용하지 않는다.
- 주어진 context를 제거하거나 사실을 바꾸는 진단으로 context 의존성을 확인할 수 있다. 이는 평가 전용이다.
- NLL target tokenization과 답변 시작 경계를 FULL/압축 조건에서 정확히 공유한다.
- 다중 정답의 NLL은 고정 canonical answer 등 사전 규칙을 사용. 조건별로 유리한 답을 선택하지 않는다.
- 첫 NLL 규칙은 답변 본문 token만 포함하고 EOS/chat 종료 token은 제외한다. EOS 확률은 필요 시 별도 지표로 기록한다.
- 빈 답변 또는 본문 token이 0개인 항목은 입력 오류로 기록하고 비교에서 제외된 수를 보고한다. NLL을 0으로 대체하지 않는다.
- prompt template, truncation, 길이 측정 기준, stop token, max_new_tokens를 고정·기록한다.

## 8. 측정·분석·보고

### 8.1 품질

```text
answer_NLL = -mean(log p(gold_answer_token | prefix, question, earlier_gold_tokens))
delta_NLL = compressed_answer_NLL - FULL_answer_NLL
FULL_correct_retention = both_correct_count / FULL_correct_count
```

- NLL의 loss mask는 context·질문을 제외한 정답 token만 포함한다. shift가 한 칸 어긋나지 않는지 작은 예제로 검사.
- delta_NLL은 음수일 수 있다. 음수를 0으로 잘라 삭제가 도움이 된 경우를 숨기지 않는다.
- 전체 EM과 FULL-correct retention을 모두 보고. FULL-correct 분모가 0이면 null/정의 불가로 기록.
- FULL 답변과 일치율(loyalty)은 정답률과 구분한다. FULL의 오답 재현도 일치일 수 있다.
- 기본 집계는 context별 질문 평균을 낸 뒤 context 평균. 질문 수 가중 micro 평균은 별도 명시.
- paired bootstrap은 context를 재표집하며 그 안의 질문 묶음을 유지한다. random seed 평균과 분산도 보고.
- 검정할 비교와 허용 가능한 성능 손실 폭은 개발 단계에서 사전 고정. 결과를 보고 기준을 바꾸지 않는다.
- Spearman은 average rank 사용. 상수 점수의 상관은 정의 불가로 표시한다.
- KVzip correlation/overlap은 보조 진단이다. 보호 pair 제외 결과도 별도 보고한다.

### 8.2 비용

| 측정값 | 포함할 것 |
|---|---|
| context build time | 입력 처리, 첫 prefill, 실제 필요한 통계 수집, score 계산, 선택, 물리 복사·삭제 |
| scorer overhead | plain prefill 대비 paired 차이; 관측·전송·선택 비용 포함 |
| persistent bytes | KV + logical ID + 필요한 template/state tensor |
| initial peak | prefill activation, full KV, 선택 중 임시 tensor·복사 포함 |
| query prefill time | 압축 context를 읽는 새 질문 suffix 처리 |
| decode time | 생성 길이를 함께 기록한 실제 생성 시간 |
| query peak | 분기 clone, 새 질문·답변 KV 등을 포함한 최대량 |

- CUDA timing은 warmup과 적절한 synchronization 또는 CUDA event를 사용. 비동기 enqueue 시간으로 비교하지 않는다.
- 단일 prefill은 `compress_context`의 한 build당 계약이다. warmup·반복 측정은 fresh cache/collector의 별도 build이며 각각 호출 수를 기록한다.
- plain-prefill 기준 비용은 별도 profile 실행으로 측정한 뒤 동일 context·환경의 결과를 짝짓는다. 후보 build 안에서 기준 forward를 추가하지 않는다.
- hook은 prefill에 포함되어 실행되므로 통계 수집 시간과 prefill 시간을 중복 합산하지 않는다.
- model load/download, 결과 저장/heatmap, KVzip 참조 실행은 후보 compressor 시간과 별도 기록.
- 반대로 후보에 필요한 CPU↔GPU 전송, score 계산, pruning 복사를 비용에서 빼지 않는다.
- diagnostic 전체 activation dump는 production-like 비용 경로에서 비활성화한다.
- 방법별 메모리 실행을 격리한다. 모델 baseline, allocated/reserved, 논리 tensor bytes를 구분한다.
- allocator의 reserved bytes가 즉시 줄지 않는다는 이유만으로 삭제 실패라고 판단하지 않는다. 실제 storage 소유권도 확인한다.
- master memory와 clone을 동시에 둔 평가 harness overhead는 숨기지 말고 배포 단일 cache 비용과 분리한다.
- stateless read에 필요 없는 score 배열은 삭제한다. 보관하면 persistent bytes에 포함한다.
- `KV bytes 감소`와 `실제 속도 향상`은 별도 결론이다. Python ragged backend를 fused kernel처럼 설명하지 않는다.
- 재사용 질문 수에 따른 총비용도 보고: `T_build + sum(T_question_j)`. KVzip 초기 비용의 분산 효과를 고려한다.

### 8.3 결과 파일 계약

새 schema를 사용하고 기존 `RECURRENT_PAIRS`·MLP probe 로그와 섞지 않는다.

- Run: schema_version, run_id, model/revision, code revision+dirty 상태, 환경 버전, device/dtype/backend,
  manifest hash, split, method, 수식/mapping/eps, 보호 정책, budget, seed, 생성·프로파일 설정.
- Build: context_id, token 수, context hash, logical clock, prefill 호출 수, 초기/생존 pair 수,
  실제 ratio, KV/metadata/state bytes, layer/head별 개수, 선택 ID digest, 시간·peak.
- Answer: run_id, context_id, question_id, condition, prediction, gold, EM, answer NLL, FULL 참조 key,
  생성 길이, query/decode 비용, status/error.
- Diagnostic: R/D 통계·표·그림 경로. 작은 correctness 실험은 실제 생존 ID도 저장.
- 오류/중단 기록을 남기고 누락 condition을 명시. malformed JSON, 중복 key, 호환되지 않는 metadata를 조용히 무시하지 않는다.
- merge는 schema/model/revision/manifest/method 설정의 호환성을 검사. 의도적 shard 병합만 허용.
- 모든 비교는 동일 context/question의 공통 완료 집합과 제외 수를 명시. FULL 참조가 없는 결과는 비교를 거부.
- 출력은 새 경로에 생성하고 기존 결과는 덮어쓰지 않는다. resume는 설정 일치·중복 방지를 검사한 경우에만 허용.

## 9. 제안 파일 배치와 CLI 계약

**아래는 앞으로 만들 파일 이름의 제안이다. 현재 실행 가능하다는 뜻이 아니다.**
기존 구조와 겹치면 기능 경계를 보존하는 범위에서 조정하고 변경 이유를 보고한다.

```text
vlm_diagnosis/core/context_only_cache.py        # text context 준비, memory/clone, suffix 평가
vlm_diagnosis/core/mlp_dynamics.py              # collector, 수식, shape/호출 검증
vlm_diagnosis/core/static_pair_select.py        # 명시적 mapping, exact budget, tie 정책
vlm_diagnosis/exps/context_only_kv.py           # 단계별 runner; benchmark와 diagnostic 분리
vlm_diagnosis/scripts/gen_context_only.py       # 재현 가능한 합성 context + 독립 질문
vlm_diagnosis/scripts/context_only_analysis.py  # 엄격한 로그 검증, paired 분석, 표/그림
tests/test_mlp_dynamics.py
tests/test_context_only_cache.py
tests/test_static_pair_select.py
tests/test_context_only_analysis.py
```

기존 `ragged_kv.py`에는 text Qwen adapter/dispatch가 필요할 수 있다.
이를 새 모델용으로 확장하더라도 기존 VL guard·예외 복원·causal mask 계약을 유지한다.

제안 CLI의 최소 옵션:

```text
--stage full|probe|deletion|sweep|profile
--model-id --revision --device --dtype --attention-backend
--manifest --split --limit
--method --mapping --keep-ratios --protected-prefix --seed
--capture-mode stats|debug
--max-new-tokens --out
```

- `full`: FULL cache와 suffix parity 확인.
- `probe`: 통계/표/heatmap. reconstruction이나 답변 평가를 암묵 실행하지 않음.
- `deletion`: low/high/random 삭제 민감도.
- `sweep`: 명시된 방법·budget의 품질 평가.
- `profile`: 선택한 한 방법의 비용 실행. 다른 방법·oracle·reconstruction을 동반 실행하지 않음.
- KVzip은 명시적으로 고른 method에서만 추가 재구성 실행을 허용하며 forward 종류·횟수를 보고.

## 10. 구현 에이전트의 작업·검증 원칙

1. 이 문서와 관련 기존 코드를 읽고 현재 상태를 확인한다. 계획을 완료 상태로 착각하지 않는다.
2. 처음에는 단계 1–2만 구현한다. 수식·shape·parity와 작은 실행 결과를 먼저 보고한다.
3. 각 단계마다 변경 파일, 실행 명령, 성공/실패 테스트, 결과 경로, 아직 검증하지 않은 내용을 보고한다.
4. 작은 CPU/tiny-config 단위 테스트 → 작은 실제 모델 실행 → 해당 기존 회귀 테스트 순으로 검증한다.
5. 큰 GPU sweep·장시간 실험으로 바로 넘어가지 않는다. 먼저 가용 자원과 pilot을 확인한다.
6. 첫 결과가 나쁘면 그대로 보고한다. score나 split을 몰래 바꿔 좋은 결과만 선택하지 않는다.
7. 직접 관측된 효과와 해석을 구분한다. heatmap, MLP ablation, KV 삭제, 비용 검증을 서로 대체하지 않는다.

최종 산출물은 코드뿐 아니라 다음을 포함한다.

- [ ] 신호 수식과 tensor shape가 대응되는 구현 설명.
- [ ] context-only·단일 prefill·독립 질문·실제 삭제를 확인하는 테스트.
- [ ] 토큰 표·heatmap, low/high/random 삭제 표, budget–품질 곡선.
- [ ] FULL 대비 보존 성능과 방법별 실제 비용 표.
- [ ] 실패 사례·불확실성·단순 baseline 대비 추가 가치에 대한 판정.
- [ ] VLM/recurrent 확장 전에 남은 제약과 다음 실험.

## 참고

- [기존 recurrent session 구조](RECURRENT-SESSION.md): 물리 pair 삭제와 현재 모델 지원 범위.
- [KVzip 논문](https://arxiv.org/abs/2505.23416): context reconstruction 기반 query-agnostic KV eviction.
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): 첫 text pilot 모델 후보.
- [Transformers Qwen2 구현, v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2/modeling_qwen2.py): pre-norm residual/MLP/cache 순서 확인용. 실제 실행 버전은 별도 고정한다.
