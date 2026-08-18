# Agent 장기 시각 메모리 연구 방향 재정리

> 작성 기준: 2026-08-18  
> 목적: 현재 KV 중심 연구를 원래 목표인 **Agent의 장기 시각 메모리에서 이미지를 어떻게 효율적으로 저장할 것인가**에 맞게 재정렬한다.

## 1. 한 문장 결론

이 연구의 핵심 질문은 다음이어야 한다.

> **미래에 어떤 질문이나 행동이 필요할지 모르는 Agent가 과거의 시각 경험을 제한된 저장공간에 어떤 표현으로 보관해야 가장 유용한가?**

같은 이미지에 서로 다른 질문을 던지는 `cross-question reuse`는 그 자체로 novelty가 아니다. 그것은 미래 질문을 보지 않고 만든 기억이 예상하지 못한 시각정보까지 보존했는지를 검사하는 **통제 실험**이다.

현재 연구는 이 전체 문제 중 `미래 질문 일반화`를 깊게 측정했지만, 이미지·텍스트·vision embedding·KV 등 서로 다른 저장 표현의 비교와 실제 Agent memory lifecycle은 아직 충분히 다루지 않았다.

---

## 2. 원래 연구 목표

Agent는 시각 경험을 저장할 때 미래에 무엇이 중요해질지 모른다. 따라서 write 시점에는 현재 이미지와 이미 끝난 episode의 질문·행동·결과만 사용할 수 있고, 미래 질문이나 미래 action은 사용할 수 없어야 한다.

이를 형식화하면 다음과 같다.

```text
write:
    image I_t + past episode context H_t
        -> memory representation M_t

constraints:
    future query/action q_future is unknown
    serialized_size(M_t) <= byte budget B

read:
    future query q_future + retrieved M_t
        -> answer or action
```

최적화해야 하는 것은 단순한 이미지 PSNR이나 KV keep ratio가 아니다.

```text
future agent utility
- persistent storage cost
- write-time computation
- retrieval and read-time computation
- model migration cost
- silent information-loss risk
```

초기 계획도 본래 다음 세 조건을 요구했다.

1. **R1 — Future-query robustness**: 미래 질문을 몰라도 유효해야 한다.
2. **R2 — Context/session portability**: 저장한 표현을 다른 세션·prompt·position에서 재사용할 수 있어야 한다.
3. **R3 — Lifetime Pareto value**: 이미지·텍스트·latent·KV 사이에서 저장량–정확도–회상비용의 비지배 영역이 있어야 한다.

관련 문서: [PLAN.md](PLAN.md), [초기 연구 목표](archive/notes-ko/01-연구목표-KV압축.md)

---

## 3. Cross-question reuse의 정확한 위치

### Novelty가 아닌 이유

다음 주제는 이미 선행연구가 존재한다.

- 같은 visual input이 후속 요청에서 다시 등장할 때 encoder/KV를 재사용하는 연구
- fine-grained visual evidence를 장기 multimodal memory에서 보존하는 연구
- 실제 Agent trajectory에서 우연히 본 pixel-only 단서를 나중에 회상하는 연구
- 미래 task에 무엇을 기억할지를 학습하는 연구
- image-centric latent/token memory를 장기 Agent memory로 사용하는 연구

따라서 다음처럼 주장하면 안 된다.

> “같은 이미지에 서로 다른 질문을 처리하는 최초의 연구”

### 이 실험이 여전히 중요한 이유

Cross-question 실험은 retrieval 오류를 제거한 상태에서 **저장 표현 자체의 미래 정보 보존 능력**을 측정한다.

```text
Stage 1 — Addressed revisit
어느 이미지인지 이미 알고 있음
-> representation fidelity만 측정

Stage 2 — Unaddressed retrieval
수백~수천 개 기억 중 어느 이미지인지 모름
-> retrieval failure와 representation failure를 분리

Stage 3 — Agent utility
검색된 기억을 이용해 실제 grounding/action/state reasoning 수행
```

현재 연구는 주로 Stage 1을 수행했다. 이는 유효한 출발점이지만, 그 자체로 Agent long-term memory 전체를 의미하지는 않는다.

---

## 4. 현재 연구가 원래 목표와 정렬된 정도

| 장기기억 조건 | 현재 상태 | 판단 |
|---|---|---|
| 미래 질문을 모르는 write 조건 | 잘 통제함 | 유지 |
| self/cross/held-out 일반화 | 강하게 측정함 | 핵심 진단으로 유지 |
| evidence displacement | 유의미한 패턴 발견 | 다른 표현까지 확장 |
| 실제 이미지 저장 표현 비교 | 거의 없음 | 최우선 추가 |
| 다른 session/context에 이식 | 미검증 | R2 gate 복원 |
| 여러 기억 중 retrieval | 미검증 | Agent 주장에 필수 |
| 상태 변화·staleness | 미검증 | 후속 Agent 평가에 필요 |
| grounding/action utility | 정적 QA 중심 | 최소 1개 action 평가 추가 |
| lifetime economics | 단일 recall TTFT 중심 | write/storage/reuse/migration 포함 |

현재 결과 중 유지할 것은 다음과 같다.

- 미래 질문 누출을 막은 V1/V2 측정 계약
- self/cross/held-out 분할
- target-aware capacity와 write-time selector의 격차
- evidence 위치가 이동할수록 기억 유지율이 감소하는 현상
- ScreenQA/GQA discovery split과 evidence annotation
- 이미지와 KV의 byte/TTFT 측정
- 실제 KV serialization/load의 M1 경로
- a4 confidence를 fast representation의 sufficiency router 후보로 사용하는 아이디어

현재 논문에서 제거하거나 별도로 분리할 것은 다음과 같다.

- 과거 answer draft를 재사용하는 3단 cascade
- cross-question setting 자체를 novelty로 주장하는 것
- sparse KV를 영구 이미지 저장 압축이라고 부르는 것
- 원본 fallback을 사용하면서 lossless memory라고 주장하는 것
- 실제 저장·복원 없이 attention mask만 적용한 결과를 시스템 성능으로 주장하는 것

---

## 5. 가장 중요한 발견: 저장과 캐시는 다르다

현재 raw 기준 대략적인 크기는 다음과 같다.

| 표현 | 이미지당 크기 |
|---|---:|
| ScreenQA/GQA 원본 이미지 중앙값 | 약 125 KiB |
| Qwen2.5 visual KV 5% | 약 3.34 MiB |
| 실제 held-out UNION KV 약 9.58% | 약 6–7 MiB |
| Qwen2.5 full visual KV | 약 66.8 MiB |

따라서 영구 저장 byte만 보면:

- 5% KV도 이미지보다 약 27배 크다.
- 실제 UNION KV는 이미지보다 약 50배 크다.
- full KV는 이미지보다 500배 이상 크다.

결론은 다음과 같다.

> **KV는 이미지를 작게 저장하는 방법이 아니다. KV의 가치는 저장량이 아니라 recall 시점의 vision encoding과 prefill을 생략하는 데 있다.**

각 표현의 올바른 역할은 다음과 같다.

| 표현 | 저장량 | 미래 세부정보 | 회상 속도 | 모델 이식성 | 올바른 역할 |
|---|---:|---|---|---|---|
| JPEG/WebP/AVIF | 매우 작음 | 높음 | 느림 | 높음 | Durable archive |
| OCR/caption/UI tree | 극소 | 미세 시각정보 손실 | 빠름 | 높음 | Search index·semantic memory |
| Vision embedding/token | 중간 | 비교적 높음 | 중간 | 낮음 | Warm representation |
| Sparse/full KV | 매우 큼 | 압축에 따라 손실 | 매우 빠름 | 매우 낮음 | Disposable hot cache |

즉 다음과 같이 연구 문제를 분리해야 한다.

### Track A — 진짜 장기 저장

```text
이미지를 관측
-> 미래 질문을 모른 채 제한된 byte로 표현을 저장
-> 원본 이미지는 제거
-> 미래에는 저장 표현만으로 답하거나 행동
```

이 track이 “이미지를 어떻게 효율적으로 장기 저장할 것인가”에 직접 답한다.

### Track B — 회상 가속 cache

```text
압축 이미지는 durable archive에 보존
-> 자주 쓰는 항목만 embedding/KV로 승격
-> fast representation이 불충분하면 이미지에서 재생성
```

이 track은 storage compression이 아니라 serving/cache 연구다. 두 track의 정확도와 비용을 한 주장으로 섞으면 안 된다.

---

## 6. Novelty가 가능한 정확한 지점

다음 broad claim은 이미 새롭지 않다.

- Agent가 이미지를 장기 기억한다.
- caption보다 이미지가 fine detail을 잘 보존한다.
- visual token 또는 latent를 압축해 기억한다.
- 미래 task를 고려해 무엇을 저장할지 결정한다.
- image와 text를 함께 저장한다.
- Agent memory를 rate–distortion 문제로 본다.

현재 가장 가능성이 있는 gap은 다음이다.

> **미래 질문을 보지 않은 동일한 visual episode에 대해 pixel codec, structured text, vision embedding, visual token, KV를 실제 serialized byte로 맞춰 비교하고, 정확도·저장량·회상속도·모델 이식성·재사용 횟수의 lifetime Pareto frontier를 측정하는 controlled visual-memory substrate study**

방법론 기여까지 만들려면 다음 방향이 적합하다.

> **각 visual episode를 항상 같은 표현으로 저장하지 않고, 미래 정보손실 위험·관측된 재사용률·hot-memory budget·모델 안정성에 따라 pixel/text/embedding/KV tier에 배치하고 승격·폐기하는 adaptive visual-memory portfolio**

단순한 계층형 cache나 LRU 자체는 novelty가 아니다. 다음을 함께 최적화해야 방법 기여가 생긴다.

- representation별 future-task information loss
- silent error risk
- actual serialized bytes
- 예상 재사용률
- byte당 latency 절감
- 모델 upgrade 시 재생성 비용
- 전체 memory bank의 제한된 global budget

---

## 7. 가장 먼저 수행할 결정 실험

새 selector나 복잡한 시스템을 만들기 전에 **Representation Frontier**를 먼저 측정한다.

### 평가 조건

- ScreenQA discovery 172개 화면
- GQA discovery 300개 이미지
- write 시 미래 평가 질문 완전 비공개
- held-out q4+ 중심 평가
- nominal KV keep ratio가 아니라 실제 serialized byte 사용
- primary archive budget 예시: 16/32/64/128/256 KiB
- model-native hot representation은 별도의 MiB budget으로 보고

### 비교할 표현

1. 원본 source JPEG/PNG
2. tuned JPEG/WebP/AVIF quality sweep
3. resolution-downsampled image
4. fixed grid tile
5. OCR-box 또는 generic saliency crop
6. OCR-only
7. dense caption
8. OCR+bbox+caption+UI tree
9. projected visual embedding fp16/int8/int4
10. quantized visual token
11. 실제 직렬화·복원한 sparse/quantized KV
12. text+compressed-image hybrid

### 측정할 지표

- 전체 future QA 정확도와 FULL-image 정답 조건부 retention
- OCR/layout/grounding/color/count/small-object별 성능
- T2/partial/T3와 evidence distance별 성능
- metadata와 index를 포함한 실제 serialized bytes
- write latency와 GPU/CPU 비용
- cold/warm load latency
- vision encode/projector/language prefill/decode로 분해한 TTFT
- silent wrong rate와 fallback rate
- 다른 모델 및 revision에서의 재사용 가능성
- 이미지당 미래 사용 횟수에 따른 lifetime break-even

### 결과에 따른 의사결정

| 결과 | 다음 방향 |
|---|---|
| tuned AVIF/JPEG가 동일 byte에서 모두 지배 | KV storage 방법은 중단하고 `A Visual Cache Is Not a Memory` 진단 논문 |
| text는 작지만 T3/fine detail에서 붕괴 | semantic base + visual residual hybrid 방법 |
| quantized vision state가 유의미한 중간점 | multi-tier warm memory 방법 |
| KV가 높은 재사용률에서만 이득 | popularity-based hot cache로 한정 |
| 여러 표현이 서로 다른 Pareto 영역 형성 | adaptive representation portfolio 시스템 |

---

## 8. Agent long-term memory를 주장하기 위한 다음 단계

Representation Frontier 이후에는 다음 순서로 확장한다.

### E2 — Unknown-future stress

- cold capture: 과거 질문이 전혀 없는 이미지
- warm capture: 과거 q0와 답을 경험한 이미지
- future-oracle: 미래 질문을 사용하는 상한
- 동일 저장 package를 T2/partial/T3/far 질문에 평가

현재의 evidence displacement 현상이 crop·text·embedding·KV 전반에서 반복되면, 특정 KV selector가 아니라 **prospective visual memory의 일반적 실패 구조**라는 더 강한 진단이 된다.

### E3 — Retrieval-inclusive memory bank

```text
과거 이미지 100~1,000개 저장
-> 새 질문만 도착
-> top-k memory retrieval
-> 검색된 저장 표현으로 답변 또는 행동
```

반드시 다음 오류를 분리한다.

- 정답 이미지가 top-k에 포함되지 않은 retrieval failure
- 정답 이미지를 찾았지만 저장 표현이 정보를 잃은 representation failure

Raw image 조건에도 OCR/caption/embedding retrieval index를 공짜로 주지 않는다. index bytes와 생성 비용을 memory package에 포함한다.

### E4 — Agent utility

- GUI grounding 또는 click action
- 같은 화면의 시간별 revision과 stale memory
- 여러 이미지에 걸친 질문
- model revision 교체
- 실제 cache promotion/demotion과 eviction

정적 QA만으로는 Agent가 과거 화면을 기억해 올바른 위치를 클릭하거나 상태 변화를 추적한다는 것을 보일 수 없다.

---

## 9. 권장 논문 방향

### 가장 안정적인 진단 논문

**A Visual Cache Is Not a Memory: Evaluating Future-Query Robustness, Portability, and Lifetime Cost of Visual Representations**

핵심 메시지:

> Query-conditioned visual caches는 미래 evidence 이동에 취약하고, model/context에 종속되며, source image보다 훨씬 클 수 있다. 따라서 KV는 durable memory가 아니라 derived hot cache로 다뤄야 한다.

### 원래 목표에 가장 가까운 시스템 논문

**What Should a Multimodal Agent Remember? A Byte-Matched Study of Visual Memory Representations under Unknown Future Tasks**

권장 구조:

```text
portable compressed-image archive  <- 영구 source of truth
              +
OCR/caption/embedding index         <- retrieval
              +
vision state / KV hot cache         <- 재사용 항목의 회상 가속
```

최종 방법 후보:

> 미래 정보손실 위험과 lifetime reuse를 기반으로 각 visual episode를 서로 다른 representation tier에 배치하고, 필요에 따라 승격·폐기하는 adaptive visual-memory portfolio

---

## 10. 최종 판단

현재 연구는 실패한 것이 아니다. 다만 지금까지의 결과는 “좋은 sparse KV를 찾았다”는 결론보다 다음을 보여주는 증거에 가깝다.

> **과거 질문에 맞춰 만든 model-native cache는 미래의 알 수 없는 visual evidence를 안정적으로 보존하지 못하며, source image보다 훨씬 클 수 있다.**

따라서 연구의 중심을 다음과 같이 되돌린다.

```text
기존 중심:
어떤 sparse KV를 남길 것인가?

새 중심:
미래 task를 모르는 Agent가 시각 경험을 어떤 representation으로,
어떤 memory tier에, 얼마의 lifetime cost로 저장해야 하는가?
```

Cross-question 결과는 이 논문의 첫 번째 진단 장으로 유지한다. KV는 비교 대상이자 hot-cache tier로 남긴다. 다음 핵심 실험은 cascade 확장이 아니라 **pixel/text/vision-state/KV의 실제 byte-matched Representation Frontier**다.

---

## 11. 2026-08-18 진단 실행 업데이트

위 방향을 말로만 제안하지 않고, 다음의 공통 측정 경로를 실제로 구현하고 검증했다.

- 원본 이미지를 보는 write와 원본 접근이 금지된 read를 별도 프로세스로 분리
- package의 실제 파일 byte와 SHA-256 검증
- `strace openat`으로 reader의 원본 `data/` 접근 감사
- raster, OCR/layout text, projected visual state FP16/INT8/INT4, full KV를 같은 질문 경로에서 비교
- 정보 유형 M4, 여러 memory D3, 화면 revision D4, offline action D5용 controlled fixture
- 물리 GPU는 2·3만 노출하도록 launcher 제한

현재 새 결과의 증거 등급은 `contract validation`, `controlled mechanism`, `real-data smoke`다.
새 discovery 또는 M7 confirmation으로 승격하지 않는다.

### 지금 확인한 핵심

1. ScreenQA 3-image/6-question formal D0 gate에서 source container, OCR layout, projected FP16,
   full KV가 source denial·actual byte/hash·6/6 coverage·반복 greedy 재현성 audit를 통과했다.
   이 소표본에서 full KV는 이미지 평균 byte의 752.72배, projected FP16/INT8/INT4 state는
   각각 93.34/46.96/23.74배였고 JPEG/AVIF 32 KiB는 약 0.34배였다. 이는 accuracy 우열이나
   equal-byte frontier가 아니라 **durable archive와 hot state의 byte 역할이 다르다**는 구현 확인이다.
2. 실제 ScreenQA/T4 4-image/24-question smoke에서는 source, JPEG 32 KiB, AVIF 32 KiB가
   모두 21/24였고 source-correct 21문항을 codec이 모두 보존했다. 더 공격적인 8 KiB gate에서
   선택적 state failure 후보가 생겼지만 n=1이며 비단조적이어서 claim으로 쓰지 않는다.
3. 32개 합성 episode의 D3에서는 distractor 0/1/3개 모두 32/32였지만 평균 read time이
   1.14→2.13→4.31초로 늘었다. 이는 retrieval 성능이 아니라 oracle/interference sanity check다.
4. 합성 D4에서 화면이 바뀐 뒤 old-only memory는 current answer 0/16, stale answer 16/16이었다.
   current-only와 old+current는 각각 16/16이었다. 오래된 visual memory의 오류가 단순 재읽기
   비용이 아니라 **조용한 과거 상태 답변**일 수 있음을 evaluator 수준에서 재현했다.
5. 24개 합성 offline action episode에서는 ordered history가 memory-dependent click을 8/8
   성공했지만, type은 값을 8/8 회상하고도 target grounding 0/8, scroll은 완료된 `OPEN`을
   8/8 재생했다. 즉 content recall, spatial grounding, temporal invalidation은 서로 다른 병목이다.

### 아직 Method를 확정하지 않는 이유

실제 데이터에서 다음 네 가지가 아직 비어 있다.

- representation × information-type의 충분한 동일-byte discovery
- 16/64개 이상 memory에서의 실제 retrieval Recall@k
- 실제 temporal revision과 conflict/invalidation
- 실제 trajectory의 action/sequence/episode success

따라서 현재 시스템 가설은 `compressed raster source of truth + structured retrieval index +
optional hot visual state/KV`까지다. 어느 부분이 novelty 있는 Method가 될지는 위 진단에서
실제 병목이 확인된 뒤 정한다.

상세 측정 계약과 최신 결과 구분:

- [진단 프로그램](docs/DIAGNOSTIC-PROGRAM.md)
- [증거 상태와 정확한 보유 정보](docs/DIAGNOSTIC-STATUS.md)

---

## 주요 직접 관련 선행연구

- [VLCache — repeated multimodal-input encoder/KV reuse](https://arxiv.org/abs/2512.12977)
- [MemEye — fine-grained visual evidence in multimodal agent memory](https://arxiv.org/abs/2605.15128)
- [DMV-Bench — interactive long-horizon incidental visual recall](https://arxiv.org/abs/2606.27499)
- [MIRIX — long-term multimodal screenshot memory](https://arxiv.org/abs/2507.07957)
- [Mem-W — latent memory-native GUI agents](https://arxiv.org/abs/2605.09317)
- [AstraNav-Memory — compressed image-centric long memory](https://arxiv.org/abs/2512.21627)
- [TaskMem — task-focused memorization policy](https://arxiv.org/abs/2605.31075)
- [PMMC — prospective multimodal memory compilation](https://arxiv.org/abs/2608.00962)
- [Decision-centric rate–distortion for agent memory](https://arxiv.org/abs/2605.10870)
