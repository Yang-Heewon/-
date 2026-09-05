# 이미지와 문맥 중요도를 갱신하는 세션 KV 실험

상태: **미선택 KV를 실제로 삭제하는 training-free recurrent prototype**.
기본 `--storage delete`는 CPU 백업도 남기지 않는다. 학습된 LSTM이나 성능 개선이 입증된 방법은 아니다.

## 현재 기본값: 전체 KV-pair 예산으로 head별 물리 삭제

현재 runner의 기본값은 **`--granularity kv_pair`**다. 모든 layer/head에 같은 토큰 mask를
씌우지 않고, **`(layer, KV head, logical token)`의 K 벡터 + V 벡터 한 쌍**을 선택 단위로 쓴다.
벡터 내부의 scalar 원소를 20% 남기는 방식도, 각 head에 똑같이 20%씩 할당하는 방식도 아니다.
GQA에서는 여러 query head가 공유하는 **KV head**가 저장 단위다.

```text
초기 layer 수 L, KV head 수 H, prefix 길이 P
전체 초기 pair 수 = L × H × P
고정 예산 B = round(0.2 × L × H × P)

예: 2 layers × 2 KV heads × 4 tokens = 16 pairs, B = 8
             남는 token ID       pair 수
layer0/head0: [0, 1, 3]             3
layer0/head1: [2]                   1
layer1/head0: [0, 2]                2
layer1/head1: [1, 3]                2
                                  합계 8
```

위 예시는 `n_sink=0`일 때다. 기본 보호 prefix는 head마다 처음 4개이며 **이 pair들도 B에 포함**한다.
보호 pair보다 작은 예산은 거부한다. 보호분 이외에는 고정 head quota가 없으므로 점수에 따라
어떤 head에는 많이, 다른 head에는 적게 남는다. 한 token의 KV가 특정 head에서 삭제되어도
다른 head에는 남을 수 있다. 각 pair의 bytes가 같은 모델 계약 안에서 전역 개수 예산이 bytes 예산에 대응한다.

현재 구현 경로:

| 구성 | 파일 | 역할 |
|---|---|---|
| 초기 관측·모델 연결 | `core/session_adapters.py`, `core/ragged_kv.py` | `QwenPairAdapter`, 실제 prefill attention의 `[L,H,P]` 점수 |
| 전역 중요도 | `core/pair_importance.py` | pair마다 prior/history state, 보호분 포함 전체 B개 선택 |
| 물리 저장·attention | `core/ragged_kv.py` | head마다 서로 다른 길이의 owning K/V tensor, 해당 생존 항목에만 attention |
| 반복 세션 | `core/pair_session.py` | 생존 pair + 새 문맥 → 관측·갱신 → B개 선택·실제 삭제 |

초기 점수는 이미지-only decoder prefill 한 번에서 얻는다. visual query row들과 같은 KV head를
공유하는 GQA query heads만 평균하고, **서로 다른 layer/KV head의 점수를 평균하지 않는다**.
기존 token 점수를 모든 head에 복제한 것도 아니다. 현재 prior는 image 위치만 점수가 있고,
prefix 제어 토큰은 보호분으로 남긴다. 다음 질문이나 정답은 초기 선택에 사용하지 않는다.

매 대화 턴에서는 직전 생존 cache로 현재 질문과 자체 생성 답변을 처리하며 pair별 attention을
누적한다. 아래의 training-free gate 공식을 pair별로 적용하고 전체 후보에서 B개를 재선택한다.
**여기서 STEP은 완료된 대화 턴**이다. 생성 token마다 관측은 하지만 삭제·재선택은 턴 끝에 한다.
갱신되는 것은 선택 점수이지 K/V 벡터 값의 LSTM식 혼합이 아니다.

직전 B개와 새 토큰 N개에 대응하는 `L × H × N` pairs만 후보에 들어간다. 선택되지 않은 K/V와
그 pair의 importance state를 실제 축소 복사 후 해제한다. CPU reservoir나 padded full-history
백업은 없고, 한 번 버린 pair는 후속 질문에서 복구할 수 없다. 새로운 문맥이 같은 token 위치의
원본 이미지 K/V를 재생성하는 방식도 아니다. 원래 논리 위치와 mRoPE는 보존한다.

각 head의 K/V 길이가 다른 ragged cache를 attention이 직접 읽는다. **현재는 batch=1, eager,
inference용 Python 참조 backend**이며 Qwen2.5-VL/Qwen3-VL 이미지 1개 + 텍스트 세션을 검증했다.
fused kernel, sliding-window attention, 병렬/nested ragged forward는 지원하지 않는다.
FULL도 동일 ragged backend에서 모든 pair를 쓰므로 실행 경로를 맞추지만 속도 개선을 입증한 것은 아니다.

모달리티 ID와 pair 중요도 엔진은 확장 가능하지만 실제 native pair 세션은 현재 Qwen 이미지·텍스트
어댑터만 받는다. audio/video 원본 입력과 추가 `token_features` 연결은 아직 지원하지 않으며
조용히 무시하지 않고 거부한다. 아래 기존 token 모드의 feature/selector 확장 API까지
새 pair 세션에 전부 연결된 것으로 해석하면 안 된다.

### 실행과 pair 로그

```bash
python -m vlm_diagnosis.exps.recurrent_session \
  --model qwen25vl --device cuda:1 --limit 1 --steps 4 \
  --granularity kv_pair --budget 0.2 --storage delete --max-new-tokens 8 \
  --out results/smoke/recurrent_session_pairs_example.jsonl

python -m vlm_diagnosis.scripts.pair_session_analysis \
  --input results/smoke/recurrent_session_pairs_example.jsonl \
  --out results/smoke/recurrent_session_pairs_example_summary.md
```

기존 출력은 덮어쓰지 않는다. pair 로그는 schema `2.0` / `RECURRENT_PAIRS`이며 별도 analyzer를 쓴다.
비교 요약에는 `full,image_static,recurrent` 모두 필요하다. `--conditions recurrent`는 단독 메모리
실행용이며 비교 analyzer의 입력은 아니다. `--storage offload`는 기존 `--granularity token`에서만 가능하다.

`retained_kv_pairs`, `retained_kv_bytes`, `selection_after.pairs_by_layer_head`로 실제 저장량과
head별 배분을 확인한다. `distinct_logical_tokens`는 한 head 이상에 남아 있는 서로 다른 token ID 수로,
pair 수나 모든 head에 공통으로 남은 token 수와 다르다. selector state는 pair당 **34 bytes**,
cache의 원래 token ID는 추가 **8 bytes**이며 template tensor와 함께 별도 회계한다.
턴 중 후보는 `B + L×H×N` pairs이고 복사 중에는 더 커진다.
`cache_storage_peak_bytes_upper_bound`는 후보 KV bytes의 2배인 보수적 cache-storage 상한으로,
초기 full prefill, activation, model weights, selector 임시 배열의 peak를 포함하지 않는다.
`persistent_session_tensor_bytes`는 턴 사이의 KV + selector state + cache ID/template tensor 합이다.
GPU 전체 allocation peak는 다른 비교 조건 cache와 공유 모델까지 포함하므로 단독 방법 메모리가 아니다.

### pair 경로 검증 결과

- `tests/test_ragged_kv.py`: 실제 작은 Qwen2.5/3 decoder에서 per-layer/head dense-mask 기준과
  출력·attention 점수 비교, 길이가 다른/비어 있는 head, 실제 storage 축소와 비복구 검사.
- `tests/test_pair_importance.py`, `tests/test_pair_session.py`: 전역 예산, head별 다른 선택,
  실제 두 턴 cache/state 정렬, static/FULL/recurrent 조건, 삭제 항목 비복구 검사.
- `tests/test_pair_session_analysis.py`: 잘못된 예산·head 회계·FULL 비교·턴 연속성 로그 거부.
- [Qwen2.5-VL pair 결과](../results/smoke/recurrent_session_q25_pairs_summary.md): ScreenQA 1이미지·4턴,
  초기 138,656 pairs 중 **27,731 pairs**, 14,198,272 bytes 유지. recurrent head별 최소–최대는
  28–559 → 36–541 → 51–538 → 64–519로 변화했다.
- [Qwen3-VL pair 결과](../results/smoke/recurrent_session_q3_pairs_summary.md): 같은 1이미지·3턴,
  초기 588,960 pairs 중 **117,792 pairs**, 60,309,504 bytes 유지. recurrent head별 최소–최대는
  29–1345 → 54–1308 → 75–1251로 변화했다.

두 실행에서 세 조건 모두 각각 4/4, 3/3 정답을 유지했다. 이는 작은 실행 검증일 뿐 일반적인
성능 보존, KVZIP 대비 우위나 novelty의 증거가 아니다. 각 조건이 자체 생성 답변 이력을 쓰므로
동일 과거 답을 강제한 selector ablation도 아니다. pair ID 전체는 로그에 저장하지 않으므로
로그의 head 개수만으로 비복구를 증명하지 않으며, 실제 ID와 storage는 위 unit tests로 확인한다.

전체 테스트: `python -m pytest -q tests`.

## 기존 token 모드: 모달리티 확장 구조

**이하의 token 예산·저장량·과거 결과 설명은 `--granularity token` 경로**다.
새 pair 모드의 예산/메모리 회계와 혼합하지 않는다. 상태 갱신 원리는 두 경로가 공유한다.

공통 실행 경로는 `MultimodalSession`과 `MultimodalImportance`로 분리했다.
**현재 실제 모델 어댑터는 Qwen2.5-VL/Qwen3-VL의 이미지 1개 + 텍스트 대화만 지원한다.**
audio/video/sensor는 공통 토큰 종류와 합성 테스트에서 다루며, 원본 입력 encoder나
실제 해당 모델의 성능 검증이 추가된 것은 아니다. 서로 다른 모델의 K/V를 합치는 API도 아니다.

| 구성 | 파일 | 책임 |
|---|---|---|
| 입력 계약 | `core/session_types.py` | `SessionSeed`, `SessionInput`, modality ID/이름, 토큰별 부가 정보 |
| 모델·모달리티 어댑터 | `core/session_adapters.py` | prefill, 새 요청 인코딩, 위치 처리, attention 관측, 출력 decoding |
| 공통 중요도 | `core/recurrent_importance.py` | 초기 prior + recurrent state, 고정 예산 selector |
| 공통 cache 실행 | `core/session_cache.py` | KV 추가·선택·물리 삭제, 동일 slot의 state/metadata 축소, 메모리 회계 |

기존 `ImageSeed`, `prefill_image`, `SessionTemplate`, `RecurrentSession`은 이미지 호환 진입점으로
남겨두었다. 기존 `RecurrentImportance(image_score, image_mask, ...)`도 호환 wrapper다.
기존 token 경로는 아래처럼 공통 엔진과 어댑터를 직접 사용한다.

```python
from vlm_diagnosis.core.session_adapters import QwenImageAdapter
from vlm_diagnosis.core.session_cache import MultimodalSession

adapter = QwenImageAdapter()
seed = adapter.prefill(model, processor, image, device)
session = MultimodalSession(
    model, processor, seed, device,
    budget=round(seed.prefix_ids.shape[1] * 0.2),
    adapter=adapter, prior_floor=0.35, storage="delete",
)
del seed  # 전체 prefill KV의 호출자 참조도 해제
result = session.answer("What is shown?", max_new_tokens=16)
```

다른 모달리티를 추가하려면 `SessionAdapter`를 구현하고, 그 입력을 처리하는 **같은 모델의
공유 decoder cache**를 사용해야 한다. `prefill`이 생성한 seed와 session의 `adapter_id`도
일치해야 한다. 별도 encoder/cross-attention cache는 이 계약 밖에 있다.
layer/head별 서로 다른 token set은 이 token-common 엔진이 아니라 위 `PairSession` 경로가 담당한다.

`SessionInput`은 새 토큰의 ID, 실제 modality ID, 선택적 prior/위치/model kwargs를 전달한다.
이미지 이외를 모두 text로 취급하지 않는다. 단, 기존 이미지 실험의 수치 보존을 위해
Qwen 어댑터는 요청의 chat header와 assistant 종료 경계도 기존과 같이 text로 센다.
후속 이미지/audio/video 입력은 이 어댑터에서 거부하며, 전용 어댑터가 필요하다.
`SessionSeed`의 `modality_names`는 사용자 정의 종류까지 등록할 수 있지만, 이름을 등록하는
것만으로 모델이 해당 원본 입력을 처리할 수 있게 되는 것은 아니다.

토큰별 좌표·시간 등은 `token_features`의 `{이름: Tensor[N, ...]}`로 전달한다. schema는 seed에
선언하며, 새 토큰에서 누락되면 float는 NaN, signed integer는 -1, bool은 False로 채운다.
삭제 시 부가 정보도 같은 물리 slot으로 줄인다. 이는 원본 pixel/audio나 full KV를 보관하는
공간이 아니다. 현재 Qwen 어댑터는 좌표 특징을 생성하지 않으므로 공간 기반 연구에는 추출부를
추가해야 한다. 이 metadata는 자동으로 점수에 반영되지 않는다.

초기 점수는 어댑터의 `prior_scores`로 바꿀 수 있고, 선택 규칙은
`selector(scores=..., protected=..., budget=..., modality_ids=...) -> bool mask` 함수로 주입할 수 있다.
함수는 복사본을 받으며, 정확히 B개와 보호 토큰을 유지해야 한다. 결정적이고 부작용 없는
함수를 사용해야 한다. 별도 연구용 selector가 큰 tensor나 원본 KV를 closure에 저장한다면
그 메모리는 자동 회계에 포함되지 않으므로 금지하거나 별도로 측정해야 한다.
기본 규칙은 기존과 동일한 global top-k이며, 모달리티별 quota/점수 보정은 새로 도입하지 않았다.
새 토큰의 raw prior도 최초 normalization scale을 사용한다. 최초 scale이 0인 경우 양수
신규 prior는 거부한다. 서로 다른 모달리티 점수의 calibration은 어댑터/연구 방법의 책임이다.

새 진단은 `selected_tokens_by_modality`, `deleted_tokens_by_modality`, `token_feature_bytes`를
포함한다. canonical selector state는 token당 **18 bytes**(prior/history float32 각 4,
modality int64 8, protected/observed bool 각 1)다. 이미지 결과의 기존 JSONL 진단 이름은
호환성을 위해 유지하지만 `image_weight`는 공통 엔진에서 `prior_weight`의 legacy alias다.
KV token 크기가 동일한 shared decoder 안에서만 B개가 일정 bytes 예산에 대응한다.

## 기존 token 모드 실행

```bash
python -m vlm_diagnosis.exps.recurrent_session \
  --model qwen25vl --device cuda:1 --limit 1 --steps 3 \
  --granularity token --budget 0.2 --prior-floor 0.35 --adapter qwen_image --storage delete --max-new-tokens 16 \
  --out results/smoke/recurrent_session_example.jsonl
```

기존 `core_delta_incremental.py`와 결과 파일은 과거 실험의 재현을 위해 유지한다.
새 runner는 `full,image_static,recurrent`를 기본 비교한다. 같은 이미지의 질문들은
manifest 순서대로 같은 세션에 들어가며, **각 조건이 실제 생성한 답의 KV**가 다음 턴에
이어진다. 답을 문자열로 다시 tokenize해서 과거 KV를 대체하지 않는다.
생성한 본문 token ID는 그대로 보존하며 EOS의 여러 종류는 chat template의 assistant 종료
경계로 정규화한다. `predicted_stop_token_id`, `hit_generation_limit`, `termination_policy`에
종료 방식을 기록한다. 즉 원본 생성 EOS까지 그대로 보존하는 token-level replay는 아니다.
모델 family는 `qwen25vl`, `qwen3vl`을 받지만 실제 검증한 모델/설정은 아래 기록을 따른다.
이미 존재하는 출력 경로는 덮어쓰지 않는다.
`--image-floor`는 `--prior-floor`의 기존 이름으로 계속 동작한다.

## 한 세션의 흐름

1. 정규 chat template에서 image 종료 경계까지만 prefill한다. 미래 질문이나 범용 설명
   prompt는 넣지 않는다. decoder의 visual-row attention을 축약하여 image prior를 만든다.
2. 초기 image prior만으로 B개를 선택한다. 선택된 K/V를 별도 storage에 복사하고,
   미선택 K/V와 해당 토큰의 중요도 state를 삭제한다. 초기 full CPU seed도 해제한다.
3. 새 요청이 오면 GPU에 남아 있는 B개 KV에 질문 suffix만 forward한다.
4. 답변을 생성하면서 실제 eager attention을 층별로 즉시 축약한다. FULL trace, 정답,
   별도 teacher-forcing/scoring forward는 사용하지 않는다.
5. 생성한 마지막 토큰과 assistant 종료 경계까지 KV에 반영한다.
   선택 후보는 **직전까지 살아남은 B개 + 이번 턴의 새 질문/답변 토큰**뿐이다.
6. 이번 interaction으로 recurrent importance를 갱신하고 B개를 재선택한다.
   미선택 K/V와 해당 state는 실제로 삭제하고, **다음 질문을 받기 전에** 저장 집합을 고정한다.
   모델에 전달하는 K/V 값은 평균하거나 섞지 않고 그대로 선택한다.

예를 들어 초기 prefix가 1,000개면 200개만 저장한다. 다음 턴에서 질문/답변 50개가
추가되면 250개 중 다시 200개를 남기고 50개를 삭제한다. 그 다음 턴의 후보에 처음 버린
800개나 직전 턴에서 버린 50개를 다시 넣을 수 없다. 이미지가 바뀌지 않는 세션에서는
살아남은 원본 image KV 집합은 유지되거나 줄어들며, 과거 이미지 영역을 되살리는 기능은 없다.

초기 image forward에서만 vision encoder를 사용한다. 후속 턴에는 pixel 입력이 없다.
원래 mRoPE 위치를 유지하며, compact cache의 물리 `cache_position`과 구분한다.
이 기존 token 모드는 layer/head별 mask simulation 대신 모든 layer/head에 같은 token 위치를 적용하는
physical cache gather를 사용한다. granularity가 다르므로 과거 per-head b5 결과와 직접
동일 조건으로 취급하지 않는다.

## 상태 갱신

정확한 정의는 `vlm_diagnosis/core/recurrent_importance.py`에 있다. 초기 이미지 점수와
각 interaction의 관측된 attention을 각각 최대값으로 정규화하고, 토큰별 새 증거와 기존
state의 상대 크기로 input/forget gate를 계산한다.

```text
p_i = image_score_i / max(image_score)
x_i = observed_attention_i / max(observed_attention[observed & ~protected])
input_i  = x_i / (x_i + h_i)              # 분모가 0이면 input_i = 0
forget_i = decay * (1 - input_i)
h_i      = forget_i * h_i + input_i * x_i

image_weight(T)   = image_floor + (1 - image_floor) / (T + 1)
history_weight(T) = 1 - image_weight(T)
score_i          = image_weight(T) * p_i + history_weight(T) * h_i
```

T는 attention을 관측한 완료 interaction 수다. 기본 `image_floor=0.35`, `decay=0.9`는
검증용 heuristic 값이며 학습하거나 성능에 맞춰 튜닝한 값이 아니다. 시간에 따른 전체
mixing weight는 정해진 schedule이고, **토큰별 input/forget gate는 관측에 따라 달라진다**.
weight는 score 계수이며 image/text token 보존 비율을 강제하는 quota가 아니다.

강제 보호 prefix는 선택 경쟁에서 제외되므로 정규화와 gate 계산에서도 제외한다.
기본 delete 모드에서는 살아남은 KV와 새 문맥 KV만 state를 가지며 모두 이번 턴에서
관측한다. 삭제된 토큰의 image prior나 history state도 남지 않는다. 생존 토큰의 image
prior는 최초 normalization 값을 그대로 유지한다. 따라서 과거 선택의 편향과 비가역적인
정보 손실이 남는다. 처음 버린 시각 근거의 중요도를 새로 알아내는 기능은 없다.
별도 offload 모드에서만 미관측 cold KV의 기존 state를 유지한다.
원시 attention의 sink 편향, 이미지 priors의 causal 위치 편향도 후속 검증 대상이다.

## 예산과 저장량

`B = round(budget * initial_image_prefix_tokens)`는 세션 전체에서 변하지 않는다.
기본 보호 prefix 4개도 B에 포함한다. 이전 질문/답 KV도 B 안에서 이미지 KV와 경쟁한다.
압축 조건의 **턴 사이에 영구 저장하는 KV가 실제로 B개**다. CPU KV는 0 bytes다.
매 턴의 KV H2D/D2H 복사도 없으며, 초기 seed로부터 선택 KV를 구성하는 복사는 별도 setup이다.
현재 질문/답변 처리 중에는 새 토큰 N개가 추가되어 `B + N`개를 사용한다. 재압축에서는
선택된 B개를 새 storage로 복사하고, 현재 Transformers의 cache 생성자가 이를 한 번 더
복사하므로 기존 `B + N`개 + gather B개 + 최종 cache B개가 잠시 공존할 수 있다.
KV만의 보수적 peak 상한은 `3B + N`개다. 초기 이미지 prefill도 full prefix가 필요하다. 따라서 실행 전체의
GPU peak까지 20%라는 뜻은 아니다. 해제된 tensor 메모리를 PyTorch allocator가 reserve하는
것과, 접근 가능한 KV tensor를 저장하는 것은 구분한다.

K/V에는 slice view가 아닌 `index_select` 복사를 사용한다. 따라서 B개 tensor가
삭제 전 큰 storage를 참조하지 않는다. `active_indices`는 위치를 추적하는 전역 논리 ID이고,
실제 cache와 selector state의 인덱스는 생존 토큰으로 조밀하게 재배치한다.

JSONL의 주요 회계는 `retained_kv_tokens/bytes`(턴 종료 저장량), `cold_kv_bytes`,
`peak_active_kv_bytes`(forward 중 KV), `compaction_peak_kv_bytes_upper_bound`,
`selector_state_bytes`, `session_metadata_bytes`다. `persistent_session_tensor_bytes`는
저장 KV + selector state + 세션의 ID/mask/template tensor bytes를 합한다.
`combined_kv_and_state_bytes`는 CPU KV + compaction GPU KV 상한 + selector state 합이며
forward activation, 임시 selector tensor와 Python 객체 오버헤드는 포함하지 않는다.
`initial_deleted_tokens`, `deleted_tokens_this_turn`, `deleted_image_tokens_this_turn`으로
초기/매 턴의 삭제량을 기록한다. `logical_history_tokens_after`는 이미 삭제된 토큰까지 센
누적 처리 개수이지 저장 중인 KV 개수가 아니다.
`selected_history_text_tokens`는 실제 이전 요청/응답 토큰 수이며 prefix control과 구분한다.
GPU 전체 할당 peak인 `peak_gpu_allocated_bytes`에는 weights/activation뿐 아니라 같은
프로세스에서 비교 중인 다른 조건의 resident cache도 포함된다. 이를 특정 방법의 단독
메모리로 해석하면 안 된다. 단독 측정에는 `--conditions recurrent` 등으로 따로 실행한다.

과거 CPU reservoir 방식은 `--storage offload`로만 사용한다. 이 모드는 전체 KV를 CPU에
보관하므로 GPU working-set 실험이지 전체 저장량 압축이 아니다. 과거 결과는 그대로 보존한다.

## 비교와 판정

- `full`: 모든 과거 KV 사용, 자체 full-cache 답변 이력.
- `image_static`: 초기 image-only 선택을 고정. delete 모드에서는 이번 질문/답 KV를 턴 끝에 전부 삭제.
- `recurrent`: 이미지 prior와 자체 interaction state로 생존 KV + 새 문맥 중 고정 예산만 남기고 삭제.

세 조건은 같은 image seed와 질문 순서를 쓰지만 **이전 답변이 달라질 수 있다**. FULL 비교는
end-to-end trajectory 비교이며, 같은 과거 답을 강제한 독립적인 selector ablation은 아니다.
`full_correct_retained`는 해당 턴 FULL이 맞힌 경우에만 압축 조건의 EM을 기록한다.
전체 EM/ANLS, FULL 정답 보존율, 답 일치율을 함께 평가해야 한다. 많은 질문이 같은 이미지에
속하므로 유의성 검증은 이미지 단위 paired bootstrap 등으로 해야 한다.

이 runner는 과거 b5와 달리 질문/응답을 실제 한 세션으로 연결한다. ScreenQA/GQA의
manifest 순서는 자연스러운 대화라는 보장이 없으므로, follow-up 및 topic-shift 평가가
별도로 필요하다. 현재 작은 smoke를 novelty나 KVzip 대비 성능 향상의 증거로 사용하지 않는다.

## 검증

```bash
python -m unittest discover -s tests -p 'test_recurrent*.py' -v
```

단위 검증은 고정 예산, 이미지 초기화, 문맥 추가, 동적 gate, 미관측/0점 구분,
실제 token gather와 위치 보존, 세션 격리, 실제 생성한 마지막 토큰 반영을 다룬다.
추가로 삭제 후 cache/state의 실제 storage 축소, 원본과 storage 비공유, 비선택 KV의 비복구,
생존/신규 토큰만 후보가 되는지, 삭제 후 논리 위치 보존을 검사한다.
GPU smoke 결과와 모델별 적용 범위는 별도 validation 보고서에 기록한다.

모달리티 분리 리팩토링 검증:

- `tests/test_multimodal_session.py`: 합성 image/audio/video/sensor/text 토큰의 공통 cache 경로,
  typed metadata 삭제·저장량 회계, 비복구, 잘못된 입력의 거부를 검증한다. 실제 음성/영상 모델 검증은 아니다.
- `tests/test_session_adapters.py`: Qwen prefix/template/EOS와 논리·물리 위치 분리를 검사한다.
- `results/smoke/recurrent_session_q25_adapter_refactor_summary.md`: Qwen2.5-VL 1이미지·4턴.
- `results/smoke/recurrent_session_q3_adapter_refactor_summary.md`: Qwen3-VL 같은 이미지·3턴.

리팩토링 전 `_delete_checked` 실행과 후 `_adapter_refactor` 실행은 세 조건 모두에서
답변 문자열, EM, 매 턴 전/후 선택 ID, 저장 KV bytes가 정확히 같았다(각 12/9행).
selector의 modality ID 저장으로 state bytes는 token당 11 → 18로 변경됐으며 별도 회계된다.
비전 특화 선택법이나 네이티브 audio/video 어댑터를 새로 구현한 것은 아니다.

기존 token-common delete 방식의 실행 결과(pair 경로의 결과가 아님):

- `results/smoke/recurrent_session_q25_delete_checked_summary.md`: Qwen2.5-VL-7B,
  ScreenQA 이미지 1개, 4턴. 초기 prefix 1,238개에서 990개를 삭제한 뒤 매 턴 248개 유지.
  저장 KV 14,221,312 bytes(약 13.56 MiB), CPU KV 0 bytes.
  종료 시 이미지/과거 문맥/control 구성은 231/13/4 → 209/35/4 → 186/58/4 → 162/82/4.
- `results/smoke/recurrent_session_q3_delete_checked_summary.md`: Qwen3-VL-8B,
  같은 이미지 1개, 3턴. 초기 prefix 2,045개에서 1,636개 삭제 후 매 턴 409개 유지.
  CPU KV 0 bytes. 종료 시 이미지/과거 문맥/control은 383/22/4 → 354/51/4 → 329/76/4.

두 실행 모두 FULL/static/recurrent가 각각 평가한 4개/3개 질문을 모두 맞혔다.
후보 ID가 직전 생존 집합과 이번 턴 신규 ID의 합집합 안에 있는지, 삭제 후 과거 ID가
되살아나지 않는지, 저장량이 고정 B인지 검증한다. 데이터가 작으므로 성능 보존이나
KVzip 대비 개선의 근거는 아니다. 생성 제한은 두 실행 모두 8 tokens다.
이름에 `_checked`가 없는 초기 delete JSONL은 cache 생성자의 추가 복사를 peak 회계에서
빠뜨린 진단 로그다. 최종 메모리 검증과 요약에는 위 `_delete_checked` 파일만 사용한다.

과거 offload 방식의 실행 결과(삭제 압축의 증거로 사용하지 않음):

- `results/smoke/recurrent_session_q25_checked_summary.md`: Qwen2.5-VL-7B,
  ScreenQA 이미지 1개, 4턴, 초기 prefix 1,238개 대비 고정 248개.
- `results/smoke/recurrent_session_q3_smoke_summary.md`: Qwen3-VL-8B,
  같은 이미지 1개, 3턴, 초기 prefix 2,045개 대비 고정 409개.

Qwen2.5의 선택된 과거 문맥 토큰은 13 → 39 → 66 → 89개, Qwen3는 22 → 51 → 76개로
증가하면서 전체 hot-history 예산을 유지했다. 세 조건은 각각 평가한 4개/3개 질문을 모두
맞혔다. 이는 구조와 실행 경로의 smoke 검증이며 방법 간 성능 우위를 보여주지 않는다.
이후 종료 정책 기록과 attention hook의 forward 완전성 검사를 추가했고,
두 모델 계열의 작은 실제 decoder 및 세션 unit tests로 회귀 검증했다.

요약 생성:

이 비교용 analyzer는 세 조건 `full,image_static,recurrent`가 모두 있는 로그를 요구한다.
`--conditions recurrent` 단독 메모리 측정 로그는 이 비교 요약의 입력이 아니다.

```bash
python -m vlm_diagnosis.scripts.recurrent_session_analysis \
  --input results/smoke/recurrent_session_q25_delete_checked.jsonl \
  --out results/smoke/recurrent_session_q25_delete_checked_summary.md
```
