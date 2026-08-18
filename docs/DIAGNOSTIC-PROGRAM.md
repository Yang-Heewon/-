# Agent visual long-term memory: diagnosis-first program

> 상태: **진단 계약 초안** (2026-08-18)  
> 목표: Method를 먼저 고르지 않고, Agent가 과거 시각 상태를 어떤 표현으로 보관해야
> 하는지 결정하는 데 필요한 실패 원인을 분리한다.

## 1. 중심 연구 질문

Agent가 write 시점에 이미지 `x`를 보았고 미래 질문·행동 `q`는 아직 모른다고 하자.
read 시점에는 원본 `x`에 접근할 수 없고, 저장 패키지 `M(x)`만 사용할 수 있다.

> **실제 직렬화 byte 상한 `B` 아래에서 어떤 `M(x)`가 미래의 QA/action utility를
> 가장 잘 보존하며, 그 실패는 정보 유형·검색 간섭·상태 변화 중 어디에서 생기는가?**

이 질문은 세 문제를 분리한다.

1. **durable substrate**: pixel, OCR/layout text, encoder embedding, visual token, KV 중 무엇을 저장하는가?
2. **memory management**: 여러 기억에서 무엇을 검색하고, 충돌·오래된 기억을 어떻게 다루는가?
3. **serving acceleration**: 원본 또는 durable memory를 backing store로 둘 때 hot KV가 TTFT를 얼마나 줄이는가?

KV가 빠르다는 사실만으로 durable memory라고 부르지 않는다. 저장 효율과 serving 효율은
별도 축으로 보고한다.

## 2. 모든 실험에 적용할 측정 계약

### 2.1 미래 질문 비공개

- representation 생성·압축·token 선택은 평가 질문을 볼 수 없다.
- 과거 episode를 쓰는 조건은 episode 질문·답·행동의 provenance를 기록한다.
- 평가 질문 또는 정답을 본 selector는 `diagnostic upper bound`로만 표시한다.

### 2.2 source unavailable

write와 read를 별도 프로세스로 실행한다.

- write 프로세스 입력: 원본 이미지와 write-time history
- write 프로세스 출력: self-contained memory package와 manifest
- read 프로세스 입력: memory package와 새 질문만
- read 프로세스에는 원본 이미지 경로를 전달하지 않는다.
- reader가 `data/` 아래 파일을 열지 않았음을 `strace -e openat` 감사로 확인한다.
- 원본을 실제 삭제하지 않는다. 접근 금지 계약으로 같은 효과를 안전하게 검증한다.

package에는 payload뿐 아니라 decode에 필요한 grid/span/position, dtype/quantization,
tokenizer·model revision을 포함한다. byte 수는 추정식이 아니라 **실제 package 파일 크기**로 잰다.

### 2.3 byte cap

동일-byte는 “정확히 B byte를 채움”이 아니라 `actual_package_bytes <= B`로 정의한다.
padding으로 크기를 맞추지 않으며 utilization도 보고한다. 어떤 표현의 최소 package가 B보다
크면 accuracy 0으로 처리하지 않고 `infeasible at B`로 기록한다.

두 frontier를 분리한다.

| frontier | 제안 byte cap | 비교 대상 | 의미 |
|---|---:|---|---|
| archive | 32, 64, 128, 256 KiB | JPEG, WebP, AVIF, OCR+layout, dense text, hybrid | 오래 보관할 backing store |
| hot state | 1, 2, 4, 8 MiB | 위 조건 + encoder embedding, visual token, sparse/quantized KV | 빠른 재사용용 RAM/GPU cache |

원본 파일 크기는 upper-bound 표현의 실제 byte로 함께 기록한다. codec별로 목표 byte 이하를
만족하는 최고 품질 설정을 탐색하고, 불가능한 budget은 숨기지 않는다.

### 2.4 결과 지표

- `base_score`: 원본 IMAGE로 푼 task score
- `memory_score`: source-unavailable memory로 푼 task score
- `conditional_retention`: IMAGE가 맞은 표본에서 memory가 맞은 비율
- `unconditional_score`: 모든 표본의 memory score
- `false_preservation`: IMAGE가 틀렸는데 memory가 우연히 맞은 비율
- 실제 package bytes, write/read wall time, read peak GPU memory, TTFT
- model/version portability와 package build provenance

IMAGE 자체가 못 푼 표본을 representation 손실로 세지 않기 위해 retention을 주 지표로 쓰되,
난이도 선택 편향을 숨기지 않도록 unconditional score를 항상 병기한다. CI의 표본 단위는 질문이
아니라 image/episode cluster이다.

## 3. 순차 진단

### D0. Source-denial 및 실제-byte gate

목적은 실험 구현이 몰래 원본을 다시 읽거나 추정 byte를 실제 저장량처럼 쓰지 않는지 확인하는
것이다. IMAGE, text, embedding, KV 각각 최소 3개 표본에서 write→serialize→새 프로세스
load→answer를 통과해야 다음 단계로 간다.

완료 조건:

- reader의 원본 open 0회
- package hash와 actual bytes 기록
- 같은 package를 두 번 읽었을 때 greedy 결과 재현
- KV/embedding read가 pixel re-encode를 호출하지 않음

### D1. Byte-matched substrate frontier

표현 팔:

1. original image (실제 파일 byte, upper bound)
2. JPEG / WebP / AVIF
3. OCR+bbox, dense description, UI tree, 이들의 결합
4. encoder output embedding
5. LLM 입력 visual token
6. full/sparse/quantized visual KV
7. `compressed raster + structured text` hybrid

각 budget에서 미래 질문 비공개로 package를 한 번 만들고 동일 이미지의 여러 질문에 재사용한다.
archive와 hot-state frontier를 따로 그린 뒤, `(bytes, read latency, accuracy)` Pareto set을 보고한다.

핵심 판정:

- raster가 archive budget 전반을 지배하면 “KV를 장기 저장”하는 방향은 기각하고 KV는 hot cache로 한정한다.
- OCR/layout text가 raster와 상보적이면 structured text + tiny raster hybrid가 Method 후보가 된다.
- embedding/KV가 추가 byte만큼 정확도 또는 TTFT 이득을 못 내면 persistent payload에서 제외한다.

### D2. M4: 같은 이미지 안의 정보 유형별 손실

유형은 mutually exclusive class 하나가 아니라 요구 능력의 다중 라벨로 기록한다.

- `requires_text`: OCR 문자열·숫자
- `requires_semantics`: entity/state
- `requires_layout`: 상대 위치·reading order
- `requires_grounding`: bbox/click target
- `requires_icon`: 비텍스트 icon/affordance
- `requires_count`: 반복 요소 수

각 질문은 evidence bbox/element와 답 형식(text/click/action)을 가진다. 같은 이미지에 최소 4개
유형을 붙여 dataset 난이도와 정보 유형 효과를 분리한다. 분석은 IMAGE-correct subset의
representation × information-type interaction과 image-cluster bootstrap으로 한다.

현재 `t4_pilot`은 OCR/semantic-ambiguous, count, state grounding, location proxy의 smoke에는
쓸 수 있지만 icon·click과 OCR/semantic 분리는 부족하다. 이를 논문 M4로 승격하기 전에 별도
PCTD annotation과 검수가 필요하다.

오류는 다음으로 분해한다.

- `base_failure`: IMAGE도 실패
- `not_stored`: evidence가 payload에 없음
- `stored_not_retrieved`: 관련 memory가 검색되지 않음
- `retrieved_not_used`: evidence가 있는데 답/action 실패
- `stale_conflict`: 오래된 증거가 현재 증거를 이김
- `uncertain`: annotation/metric으로 원인 판정 불가

### D3. 여러 memory: 검색·간섭·조립 분리

세 실험을 섞지 않는다.

1. **retrieval**: N개 중 relevant memory의 Recall@k와 oracle-selected task score
2. **interference**: relevant block은 항상 포함하고 distractor 0/1/3개만 추가
3. **composition**: 같은 block들을 independent serialize+concat과 joint prefill로 비교

관련 memory를 놓친 오류와, 찾았지만 distractor 때문에 틀린 오류, KV position/context 조립 오류를
각각 분리한다. N은 smoke `1/2/4`, discovery `1/4/16/64`로 늘리며 actual total bytes를 고정한다.

### D4. 시간과 화면 상태 변화

먼저 통제된 2×2로 시간과 상태 변경을 분리한다.

| | 가까운 read | 먼 read |
|---|---|---|
| content 불변 | time/position 효과 | 장거리 사용 효과 |
| content 변경 | update 효과 | staleness+장거리 결합 |

충돌 조건은 `old only / current only / old+current / no memory`이다. 질문은 항상 현재 상태의
정답을 요구하며, 과거 답을 출력한 비율을 `stale capture rate`로 별도 기록한다. 기존 정적
ScreenQA를 편집한 counterfactual은 인과적 smoke로만 쓰고, 실제 trajectory 결과와 구분한다.

### D5. M5: Agent action utility

정적 QA에서 살아남은 조건만 trajectory로 승격한다.

- 1차: offline next-action target/element success
- 2차: action sequence success와 episode success
- 3차: OSWorld류 실제 환경 성공률(환경 실패를 memory 실패와 분리)

Multimodal-Mind2Web/AndroidControl/GUIOdyssey 중 screenshot, action, revision, license가 실제로
재현되는 데이터 하나를 access smoke 후 고른다. local 데이터가 없는 현재는 production M5 수치를
낼 수 없고, controlled static action proxy까지만 가능하다.

### D6. 독립 confirmation (M7)

D0–D5의 discovery 결과를 본 뒤 가장 중요한 주장 **하나**와 Method 후보를 고른다. 그 다음
threshold, budget, metric, exclusion을 동결하고 겹치지 않는 image/episode와 독립 model family에서
한 번만 확인한다. confirmation split은 Method 선택에 사용하지 않는다.

## 4. Method 선택 규칙

진단 결과가 Method를 결정한다.

| 관찰된 주 병목 | Method 방향 |
|---|---|
| archive byte에서 raster가 압도 | compressed source archive + optional hot KV cache |
| OCR/layout만 선택적으로 소실 | structured OCR/layout + tiny raster hybrid |
| relevant memory 검색 실패 | evidence-aware retrieval/index |
| 찾은 뒤 distractor 간섭 | gated read / memory routing |
| stale memory가 현재 상태를 덮음 | versioning, invalidation, conflict-aware update |
| KV 조립/position 실패 | portable latent representation 또는 re-anchoring |
| read latency만 문제 | durable image backing + disposable model-specific cache |

따라서 draft verifier는 이 프로그램의 중심 저장 해법이 아니다. 그것은 이미 선택된 memory에서
답 생성을 빠르게 하는 serving layer이며, 저장·검색·staleness 진단이 끝난 뒤 독립적으로 평가한다.

## 5. 로컬 실행 정책

- 물리 GPU **2·3만** 사용한다.
- 각 worker는 `CUDA_VISIBLE_DEVICES=2` 또는 `3`으로 카드 한 장만 노출하고 내부에서는 `cuda:0`을 쓴다.
- GPU당 7–8B model process 하나, batch 1, 두 번째 worker는 45초 뒤 기동한다.
- V100에서는 fp16 + eager/SDPA를 사용하고 FlashAttention-2와 full attention dump를 쓰지 않는다.
- 실제 payload는 표본별 streaming 생성→평가→삭제하고 결과 JSONL·hash·작은 package만 보존한다.
- 남은 디스크가 약 185GB이므로 full KV corpus를 영구 중복 저장하지 않는다.

## 6. 현재 구현 상태와 추가 준비

| 진단 | 2026-08-18 현재 상태 | 논문용으로 추가할 것 |
|---|---|---|
| D0 source-denial | 3개 실제 이미지에서 source container/OCR layout/projected FP16/full KV의 6/6 coverage·actual byte/hash·open audit·반복 greedy gate 통과 | 독립 sandbox까지 요구할지 결정; discovery runner 전 조건에 같은 audit 적용 |
| D1 frontier | JPEG/WebP/AVIF, 독립 PaddleOCR, projected FP16/INT8/INT4, full KV의 actual-byte 소표본 경로 구현 | archive/hot budget grid, 충분한 실제 표본, equal-byte Pareto와 CI |
| D2 M4 | 합성 24-image fixture와 real T4 4-screen codec smoke 실행 | 실제 icon/click 포함 라벨, OCR/semantic 분리, full controlled 실행과 cluster CI |
| D3 multi-memory | 합성 32-episode oracle/interference 320 trial source-free 실행 | 실제 Recall@k, fixed-total-byte interference, composition, 16/64 memory scale |
| D4 temporal | 합성 2×2 state/time fixture와 old/current/both/none 실행 | 실제 동일 episode temporal pair, order counterbalance, update/invalidation |
| D5 action | 합성 24-episode click/type/scroll strict source-free evaluator 실행 | 공개 real trajectory parser, environment execution, sequence/episode success |

이 상태표에서 `실행`은 contract/mechanism smoke를 뜻한다. discovery나 confirmation으로 자동
승격되지 않으며, 최신 수치와 claim 제한은 `docs/DIAGNOSTIC-STATUS.md`에서 관리한다.

## 7. 승격 전 보완 목록 (설계 리뷰, 2026-08-18)

discovery 승격 전에 다음 8개를 반영한다 (우선순위순):

1. **질문-비공개 감사**: source-denial과 대칭으로, write 프로세스가 평가 질문
   파일을 열지 않았음을 openat 감사로 증명 (V1 질문-KV 반입 사고의 재발 방지).
2. **텍스트 팔의 예산 곡선화**: 극소 예산 눈금(1/2/4/8 KiB) 추가 + 텍스트가
   예산에 따라 확장되는 규칙(요약→상세→전문+레이아웃) 정의. 점 하나로는
   Pareto 비교 불성립.
3. **자기호환 교란 통제**: 같은 모델이 캡션을 쓰고 읽는 프리미엄 — 텍스트
   패키지를 다른 reader(다른 모델)로 읽는 팔 1개 필수 (이식성 주장의 전제).
4. **fixture headroom 기준**: 합성 탐침은 base 정확도가 천장 아래(0.6~0.9)임을
   먼저 보여야 함. 포화된 탐침(D3 32/32, D4 충돌 16/16)은 검출력 0.
5. **D5 전체-정보 상한 팔**: 현재 화면+전체 이력 원본 조건 추가 — grounding
   실패를 base_failure와 분리 (current-only target 3/8이 경고 신호).
6. **표본 조달 계획**: ScreenQA 적격 풀 소진(322장 전부 사용). discovery·M7용
   신규 GUI 소스(좌표 주석 가능) 선정이 선행 과제.
7. **다중비교 정책**: 쌍별 우열 주장은 사전 지정 대비 3~5개로 한정, 나머지는
   Pareto 집합 서술만. primary contrast를 실행 전 이 문서에 기록.
8. **프롬프트 공정성 프로토콜**: 팔당 읽기 프롬프트는 별도 튜닝 표본에서 고정
   후 동결 (프롬프트 민감도: grounding 0.20→0.60 사례).
