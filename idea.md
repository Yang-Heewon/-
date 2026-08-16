# 아이디어 등록부

> **지위**: Phase-2 방법 후보 등록부(사용자 제안). Phase-1에서는 문제와 실패 원인을
> 먼저 확정한다. 이 문서는 방법을 채택하는 문서가 아니라, 각 아이디어의 가설·적용
> 범위·검증 경로를 고정하는 문서다.
>
> **측정 절차는 [docs/IDEA-VALIDATION-METHODS.md](docs/IDEA-VALIDATION-METHODS.md)로
> 분리**한다 — 공통 원칙 P1–P7(held-out이 유일한 성패 기준, task metric이 판정,
> byte 예산, 잡음 대역, 압축 시점 라벨)과 방법별 측정 사양(층별 스윕 등).

각 아이디어는 다음 순서로 정리한다.

1. 무엇을 해결하려는가
2. 정확히 무엇을 제안하는가
3. 현재 프로젝트 범위에서 허용되는 형태는 무엇인가
4. 어떤 실험으로 채택·기각할 것인가

---

## 아이디어 1 — prefill 층을 따라 점진적으로 토큰을 제거하는 기억 선택

**등록일**: 2026-08-14

**성격**: training-free write-time selector 후보

### 한눈에 보기

| 항목 | 내용 |
|---|---|
| 목표 | 중복 시각 토큰을 층별로 제거해, prefill 종료 시 살아남은 토큰만 기억으로 저장 |
| 핵심 가설 | 매 층에서 남은 토큰을 기준으로 중요도를 다시 계산하면, 한 번의 낱개 점수보다 집합의 중복과 보완 관계를 잘 반영할 수 있음 |
| 허용되는 형태 | generic instruction을 쓰는 query-agnostic 버전 또는 과거 질문·답을 쓰는 source-aware 버전 |
| 범위 밖 형태 | 미래 질문을 넣어 선택하는 query-aware 버전. 이는 저장용 기억 압축이 아니라 read-time serving 최적화 |
| 핵심 비교군 | s5, spatial_uniform, random; query-aware 참고선으로 s1 |
| 결정 실험 | matched static selector 대비 효과, M3 held-out 귀속, write 비용 |

### 1. 문제와 제안

사용자 원안은 다음과 같다.

> 이미지와 지시를 함께 prefill하면서 층을 지날 때마다 불필요한 시각 토큰을 조금씩
> 제거하고, 마지막까지 살아남은 토큰만 장기 기억으로 저장한다.

개념적 절차는 다음과 같다.

1. 이미지와 write-time 지시를 함께 prefill한다.
2. 미리 정한 층에서 현재까지의 사용 신호로 시각 토큰 일부를 제거한다.
3. 다음 층은 제거 후 남은 토큰만을 대상으로 다시 계산한다.
4. 마지막 층까지 살아남은 토큰의 KV만 저장한다.

여기서 “PTQ처럼”은 통상적인 Post-Training **Quantization**이 아니라
“가중치 업데이트 없이 적용한다”는 비유로만 사용한다. 이 방법은 forward 내부 상태를
바꾸므로 엄밀한 post-processing은 아니다. 생존 토큰의 저정밀 저장까지 결합하면
pruning 단독이 아니라 M2-B의 HYBRID 후보로 별도 평가한다.

### 2. 적용 범위: prefill에 무엇을 넣는가

이 아이디어의 연구적 지위는 prefill 지시에 따라 달라진다.

| 형태 | prefill 입력 | 분류 | 이 프로젝트에서의 의미 |
|---|---|---|---|
| 미래 질문 기반 | 앞으로 답할 target 질문 | read-time/query-aware | 질문 도착 뒤 이미지·full state에서 subset을 만들므로 write-time compressed memory가 아님. 재-prefill 비용까지 회계해야 함 |
| A. generic | 범용 재구성·설명 지시 | write-time/query-agnostic | 범용 기억 selector 후보. s5를 층별 과정으로 확장한 형태 |
| B. episode | 과거 질문+답 | write-time/source-aware | 에피소드에서 이미 드러난 관심사를 반영하는 selector. h2o의 층별 확장 |

미래 질문 기반 형태는 기술적으로 가능하지만 Phase-1의 write-time 저장 문제를 해결하지 않는다.
파일럿에서 질문을 아는 s1이 약 0.99였으므로, 이 형태의 연구 질문은 “성능이 좋은가”가
아니라 “같은 query-aware 선택을 더 싸게 할 수 있는가”다. 이후 검증은 A와 B에
집중한다.

### 3. 왜 유망할 수 있는가

#### 3.1 낱개 점수가 아니라 선택된 집합을 다시 평가한다

파일럿에서는 중요도가 토큰 낱개의 고정 속성보다 **선택된 집합의 성질**에 가까웠다.
한 번 계산한 점수로 top-k를 고르면 서로 중복된 토큰이 함께 남을 수 있다. 반면
점진적 pruning은 일부 토큰을 지운 뒤 다음 층에서 남은 집합을 다시 평가한다.

- 같은 정보의 사본 하나가 남으면 다른 사본의 추가 가치가 낮아질 수 있다.
- 이전 층에서 제거된 토큰 때문에 다음 층의 attention 분포가 재형성된다.
- 결과적으로 반복적 greedy 선택과 비슷한 중복 억제 효과를 얻을 가능성이 있다.

#### 3.2 정적 ranking을 적응형 ranking으로 바꾼다

s5는 pruning하지 않은 run에서 얻은 전 층 attention을 정적 scalar 점수로 집계한다.
반면 이 아이디어는 앞선 삭제가 이후 층의 상태와 점수를 바꾸는 **adaptive ranking**이다.
재정규화가 실제로 중복 억제나 더 나은 근거 보존을 유도하는지는 아직 가설이며,
동일 신호의 static top-k와 직접 비교해야 한다.

### 4. 기대 범위와 핵심 리스크

| 리스크 | 의미 | 검증·대응 |
|---|---|---|
| 기존 방법과의 중복 | FastV·PyramidDrop도 forward 중 층별 시각 토큰을 제거함 | “write-time 기억 선택 신호로의 전용”이 신규 기여인지 스쿱 맵에서 재확인 |
| 너무 이른 제거 | FEATHER가 경고한 것처럼 grounding·잔글씨 근거가 먼저 사라질 수 있음 | 삭제 시작 층과 제거율을 grid로 평가하고 grounding을 별도 보고 |
| 실패 원인 귀속 | held-out 결과만으로 estimator와 미래 relevance 중 원인을 구분할 수 없음 | M3의 source/target probe와 source self-fidelity를 함께 보고 귀속 |
| write 비용 증가 | generic prefill이 추가 forward를 요구할 수 있음 | s5와 동일하게 GPU-초·CPU-초·지연을 분리 계상 |

현재 20% 파일럿의 s5 약 0.61과 query-aware s1 약 0.99 사이에는 약 38%p 차이가 있다.
이는 현재 참조 격차이지 s5가 상한이라는 뜻도, 격차 전체가 estimator 문제라는 뜻도
아니다. M3의 네 subset을 통해 selector·capacity·미래 relevance 원인을 분리한다.

### 5. 구현 전에 고정할 결정

- generic 지시 문구와 pruning 점수를 읽을 query row
- 모든 층에 공통 token set을 저장할지, 층별 ragged set을 저장할지
- pruning 시작 층, 층별 제거율, 최종 serialized-byte budget
- attention 외 신호 사용 여부와 tie-breaking
- position·mask·metadata를 포함한 실제 저장 형식

### 6. 최소 검증 계획

1. **A형 구현**: generic prefill + 층별 pruning을 training-free로 구현한다.
2. **메커니즘 대조**: 동일 prompt·동일 신호·동일 실제 byte budget의 static top-k와
   비교해 progressive 재평가 자체의 효과를 분리한다.
3. **동일 조건 비교**: FINDINGS §3의 20%·5% 사다리에서 s5, spatial_uniform,
   random과 비교한다.
4. **M3 귀속**: held-out 점수만 보지 않고 source self-fidelity, target probe,
   실제 selector를 함께 평가한다.
5. **스케줄 민감도**: 삭제 시작 층 × 층별 제거율 × 최종 예산을 비교한다.
6. **취약 유형 확인**: OCR뿐 아니라 grounding·잔글씨 과제를 포함한다.
7. **비용 회계**: 선택을 위한 추가 GPU-초, CPU-초, wall-clock, peak memory를 기록한다.
8. A형이 유망할 때만 B형(source-aware)을 추가해 generic 대비 이득과 일반화 손실을
   비교한다.

### 7. 판정 기준

정량 임계값은 Phase-2 실행 전에 동결한다. 해석 규칙은 다음과 같다.

- **메커니즘 통과**: matched static top-k보다 좋아야 “층별 재형성” 가설을 지지한다.
- **채택 후보**: source self-fidelity를 만족한 조건에서 held-out 성능이
  random·spatial_uniform을 안정적으로 넘고, s5 대비 품질 또는 비용에서 우위를 보인다.
- **범위 재분류**: self/cross 또는 미래 질문 기반에서만 좋아지면 범용 기억 selector가
  아니라 source-aware 또는 read-time serving 방법으로 분류한다.
- **기각**: held-out 이득이 없거나, grounding 손실·추가 write 비용이 품질 이득보다
  크면 Phase-1 기억 압축 방법으로 채택하지 않는다.

### 한 줄 요약

> 한 번의 낱개 점수로 top-k를 고르지 말고, 층마다 남은 집합을 다시 평가하며 기억을
> 고른다. 단, 미래 질문을 사용하면 저장용 기억 압축이 아니며, matched static 대조와
> source self-fidelity를 포함한 M3 held-out 평가로 판정한다.

---

## 아이디어 2 — 기억 관리의 CPU-first 설계

**등록일**: 2026-08-14

**성격**: 시스템 설계 원칙 + 첫 selector 후보

### 한눈에 보기

| 항목 | 내용 |
|---|---|
| 목표 | 기본 episode 처리 뒤의 memory-maintenance 경로가 응답용 GPU와 경쟁하지 않도록 설계 |
| 핵심 원칙 | 신호는 기존 forward의 부산물로 재사용하고, 벡터 후처리와 저장 관리는 CPU에서 수행 |
| 첫 구현 후보 | 시각 key에 대한 CPU k-center 대표 선택(`kv_cluster`) |
| 핵심 비교군 | spatial_uniform 약 0.45, random 약 0.44, s5 |
| 결정 실험 | held-out 품질과 GPU-초/CPU-초/지연/메모리의 공동 비교 |

### 1. 무엇을 제안하는가

CPU-first는 새로운 중요도 함수 자체가 아니라 **시스템 설계 제약**이다.

> 응답 생성에 필요한 GPU는 serving에 우선 배정하고, 기억을 고르고 압축하고 저장하는
> 추가 작업은 가능한 한 CPU·RAM·디스크로 옮긴다.

목표는 GPU 사용량 하나만 줄이는 것이 아니라, 품질과 write SLO를 유지하면서 총비용의
비지배점을 만드는 것이다. “CPU에서 실행된다”는 말은 아래 세 경우를 구분해야 한다.

- **CPU-only 작업**: 해당 작업 자체가 GPU 연산을 전혀 요구하지 않는다.
- **추가 model forward 0**: 필수 GPU forward에서 신호를 캡처하지만, 기억 관리 때문에
  별도 forward를 돌리지는 않는다. 캡처·동기화·device-to-host 전송 비용은 남는다.
- **GPU 비점유 비동기 작업**: 큰 모델 forward도 CPU에서 실행할 수 있지만 느리다.
  사용자 지연 경로 밖에서 수행할 수 있다는 뜻이지, 자동으로 싸다는 뜻은 아니다.

### 2. 작업별 배치 원칙

| 구분 | 작업 예시 | 권장 실행 위치 | 판단 |
|---|---|---|---|
| A. 기존 pass의 부산물 | 에피소드 처리 중 attention·생존 통계 캡처 | 기존 GPU pass에서 캡처 후 CPU로 전달 | 추가 model forward는 0이지만 캡처 비용은 실측 |
| B. 벡터 후처리 | norm, k-center/k-means, 커버리지 선택, 중복 제거 | CPU | CPU-only 후보 |
| C. 저장 처리 | 양자화 packing, 직렬화, 색인, 디스크 저장 | CPU | CPU-only 후보 |
| D. serving 추가 forward | 사용자가 기다리는 경로의 재평가 | GPU | 지연 민감하므로 CPU 이전 대상이 아님 |
| E. background forward | s5 재구성 pass, 아이디어 1의 generic prefill | CPU 또는 유휴 GPU | 비동기 가능하지만 실측 비용으로 결정 |

**미검증 작업 가정**으로, 7B 모델의 CPU 양자화 추론은 GPU보다 약 30–100배 느리고
CPU-초 단가는 GPU-초보다 약 10–50배 낮을 수 있다고 둔다. 이 범위만으로 비용 우위를
결론낼 수는 없다. background forward는 “GPU 점유 0”일 수 있어도 backlog·freshness·
RAM·전력·달러 비용 문제를 남긴다. 수치는 M6의 실측 가격과 처리량으로 교체한다.

### 3. 첫 방법 후보: `kv_cluster`

가장 직접적인 B등급 구현은 저장 시점의 시각 KV 클러스터링이다.

1. 기존 forward에서 시각 key 벡터를 CPU 메모리로 옮긴다.
2. CPU에서 k-center로 실제 토큰 대표를 선택한다.
3. 선택된 token index에 대응하는 key와 value를 함께 보존한다.
4. 선택된 KV만 packing·직렬화해 저장한다.

이 후보의 목적은 큰 norm의 토큰을 낱개로 고르는 것이 아니라, **중복을 줄이면서
전체 집합을 넓게 덮는 것**이다. 이는 파일럿의 “중요도는 집합의 성질”이라는 관찰과
합집합 pool이 높은 중복 덕분에 약 2배 크기로 충분했던 관찰에 대응한다.

첫 검증은 centroid KV를 새로 합성해야 하는 k-means가 아니라 k-center로 제한한다.
실행 전에 사용할 층·head 집계, 정규화, 거리 함수, 초기점·tie-breaking, 전송 정밀도,
모든 층에 공통 token ID를 적용할지를 고정한다.

### 4. 아이디어 1과의 결합

목표 아키텍처에서는 다음처럼 결합할 수 있다.

1. 기본 episode prefill에서 층별 생존·attention 신호를 캡처한다.
2. 신호와 KV를 CPU로 전달한다.
3. CPU에서 중복·커버리지를 고려해 최종 저장 집합을 고른다.
4. CPU에서 양자화·직렬화·색인을 수행한다.

이 결합은 전체 추론이 CPU-only라는 뜻이 아니다. 또한 기존 pass의 정적 신호를
CPU에서 top-k 하는 것만으로는 아이디어 1의 adaptive progressive pruning이 구현되지
않는다. adaptive 버전은 기존 GPU pass 안에서 실제 삭제에 개입하거나 별도 CPU replay를
해야 한다. 현재 h2o 프로토타입의 별도 teacher-forced capture pass도 “공짜 부산물”로
계상하지 않는다.

### 5. 핵심 리스크

| 리스크 | 의미 | 검증·대응 |
|---|---|---|
| key norm 실패의 재현 가능성 | key-norm 단일 scalar ranking은 0.17/0.33으로 random 0.44보다 낮았음 | 값 기반 방법 전체로 일반화하지 말고 k-center 집합 커버리지를 독립 검증 |
| 강한 단순 기준선 | spatial_uniform이 약 0.45이므로 단순한 “골고루 선택”만으로는 신규 가치가 작음 | 동일 예산에서 uniform 대비 추가 이득을 직접 측정 |
| CPU 병목 | PCIe 전송, clustering 지연, RAM 사용량이 GPU 절감분을 상쇄할 수 있음 | 전송량·CPU-초·wall-clock·peak RAM을 포함해 회계 |
| 실패 원인 귀속 | 기하학적 다양성이 보지 못한 질문의 근거를 보장하지 않으며 held-out만으로 원인을 구분할 수 없음 | M3 네 subset과 source self-fidelity로 귀속 |
| background forward 비용 | CPU 추론은 비동기여도 처리량·달러 비용이 나쁠 수 있음 | queue 안정성과 단위 episode 비용을 실측 |

### 6. 최소 검증 계획

1. CPU k-center 기반 `kv_cluster` selector를 구현한다.
2. 기존 20%·5% 예산 사다리에서 random, spatial_uniform, s5와 비교한다.
3. M3의 source/target probe와 실제 selector를 함께 평가해 held-out 실패 원인을 귀속한다.
4. 품질과 함께 GPU-초, CPU-초, host↔device 전송량, wall-clock, peak RAM,
   저장 바이트를 기록한다.
5. `kv_cluster`가 유망할 때만 k-means나 아이디어 1의 층별 생존 신호를 추가한다.

### 7. 판정 기준

품질 동등성 범위와 비용 환산식은 M6 실행 전에 동결한다.

- **selector 채택 후보**: held-out에서 random·spatial_uniform을 넘고, s5 대비 품질 또는
  총비용에서 우위를 보인다.
- **시스템적 승리**: 품질이 동등성 범위 안이고 write SLO를 만족하면서, GPU·CPU·전송·
  RAM·저장 비용을 합친 M6 표면에서 비지배점이어야 한다.
- **selector만 기각**: `kv_cluster`의 품질이 낮더라도 CPU-first 원칙 자체를 기각하지
  않는다. 다른 relevance 신호를 CPU에서 후처리하는 경로는 남는다.
- **원칙까지 재검토**: CPU 지연·전송·RAM 비용 때문에 GPU 절감이 실제 비용이나
  처리량 개선으로 이어지지 않으면 CPU-first 강제 조건을 완화한다.

### 한 줄 요약

> 기존 forward에서 신호만 받아오고, 선택·중복 제거·압축·저장은 CPU에서 처리한다.
> 첫 후보는 KV 클러스터링이며, 품질은 held-out으로, 시스템 가치는 GPU-초와 전체
> 비용으로 따로 판정한다.

---

## 아이디어 3 — 검증된 토큰 초안으로 기억 저장

**등록일**: 2026-08-14

**성격**: training-free write-time text-memory + 검증 방법 후보

### 한눈에 보기

| 항목 | 내용 |
|---|---|
| 목표 | 이미지 KV 대신 짧고 검증된 토큰 시퀀스를 저장하고, read 시 짧은 prefill로 KV를 재생성 |
| 핵심 가설 | 같은 길이의 무검증 설명보다, 원본 문맥과의 행동 일치를 검사한 토큰 초안이 미래 질문에 필요한 정보를 더 안정적으로 보존 |
| 주 구현 | G02의 T_visual을 초안으로 생성하고, target VLM으로 검증·보강한 뒤 통과한 토큰만 저장 |
| 핵심 비교군 | 길이를 맞춘 무검증 T_visual, 원본 이미지, 실제 byte를 맞춘 KV 기억 |
| 결정 실험 | 검증 자체의 추가 가치, M3 held-out 일반화, M4 유형별 손실, M6 write/read 비용 |

### 1. 문제와 제안

사용자 원안은 다음과 같다.

> prefill 이후의 context도 결국 입력 토큰에서 만들어진다. speculative decoding처럼
> 초안을 먼저 만들고 검증한다면, 계산 결과인 KV 대신 검증된 토큰 수준으로 기억을
> 저장할 수 있지 않은가.

주 해석의 write/read 계약은 다음과 같다.

```text
[write] 이미지 → 짧은 자연어 토큰 초안 생성
              → 원본 이미지 문맥을 가진 target VLM으로 초안 검증
              → 미달이면 보강·교체, 통과하면 토큰과 검증 metadata 저장

[read]  저장 토큰 + 새 질문을 짧게 prefill
              → 해당 질문에 사용할 KV를 그 자리에서 재생성
```

저장 artifact는 최소한 draft `token_ids`, tokenizer/model revision, 생성 prompt, 검증
기준과 통과 점수를 포함한다. read에서 재생성되는 것은 원본 visual KV의 복원이 아니라
**저장 텍스트에 조건화된 임시 KV**다. 따라서 둘의 내부 상태가 같다고 가정하지 않고,
공식 task 성능으로 기능적 동등성을 검증한다.

### 2. speculative decoding과의 정확한 관계

이 아이디어는 speculative decoding의 **draft → verify → accept/reject** 구조에서
착안했다. 그러나 표준 speculative decoding처럼 target 모델과 정확히 같은 출력
분포를 보장하는 알고리즘은 아니다.

- 표준 speculative decoding은 다음 생성 토큰을 검증한다.
- 이 아이디어는 미래 여러 질문에 사용할 **기억 표현의 충분성**을 검증한다.
- 따라서 핵심 신규 요소는 speculation 자체가 아니라, 토큰 기억에 명시적인
  검증·보강 루프를 붙이는 것이다.

기존 T_visual은 검증 없는 토큰 기억 비교군이다. 아이디어 3이 방법으로 성립하려면
같은 형식·길이의 T_visual보다 **검증 루프 때문에** 더 좋아져야 한다.

### 3. 주 해석과 대안 해석

| 형태 | 제안 | 연구적 지위 |
|---|---|---|
| **주 해석: verified token memory** | 이미지에서 토큰 초안을 만들고 write-time에 원본 대비 검증한 뒤 토큰만 저장 | 우선 검증할 기본안 |
| **대안 A: speculative selection** | 작은 draft 모델이 남길 KV/token set을 제안하고 큰 모델이 검증 | selector 아이디어로 분리. 아이디어 1·2와 인접 |
| **대안 B: hierarchical fallback** | 토큰 기억으로 먼저 답하고, 불확실할 때만 원본 이미지로 검증·재인코딩 | 원본 보관이 필요한 계층형 read-time 시스템으로 분리 |

세 형태는 저장 artifact와 비용 구조가 다르므로 하나의 결과로 합치지 않는다. 이
문서의 최소 실험은 주 해석만 다룬다.

### 4. 무엇을 기준으로 검증하는가

검증 기준이 이 아이디어의 핵심 설계 변수다.

| 검증에 쓰는 정보 | 분류 | 주의점 |
|---|---|---|
| 고정된 generic probe·재구성 기준 | write-time/query-agnostic | 범용 기억의 기본 후보지만 미래 질문 일반화는 별도 검증 필요 |
| 과거 episode 질문·답 | write-time/source-aware | 과거 관심사에는 강할 수 있으나 held-out 과적합 위험 |
| 앞으로 평가할 target 질문 | read-time/query-aware | persistent write-time 기억 방법과 직접 비교할 수 없는 참고선 |

주 검증안은 미래 평가 질문과 분리된 **고정 generic probe bank**에서 원본 이미지
문맥과 draft-only 문맥의 답 행동이 일치하는지를 보는 것이다. logits 일치는
prompt·position 정렬이 필요한 보조 진단이고, 구조화 재구성 충실도도 미래 relevance를
직접 보장하지 않는다. 어느 목적을 쓸지는 Phase-2 실행 전에 하나로 고정하며, source
질문에 대한 통과만으로 범용 검증을 주장하지 않는다.

### 5. 왜 유망할 수 있는가

#### 5.1 저장 표현이 작다

원 제안의 거친 회계는 token ID 약 2B 대 KV 약 56KB/position으로, 토큰 기억이
약 1000× 작을 가능성을 제시한다. 이는 아직 실측 결과가 아니다. visual-token 수와
draft-token 수가 서로 다르고 2B/token도 serializer에 따라 달라진다. 따라서 토큰
시퀀스 길이, 검증 metadata, tokenizer ID 폭, 보조 payload까지 포함한
`FULL-KV 실제 직렬화 bytes / draft artifact 실제 직렬화 bytes`로 다시 계산해야 한다.

#### 5.2 기존 비교군에서 자연스럽게 출발할 수 있다

M1-F에는 G02 사양의 T_visual이 이미 있다. 따라서 첫 실험은 새 표현을 처음부터
발명하지 않고, 동일 T_visual에 검증 루프를 붙였을 때의 순수한 추가 가치를 측정할 수
있다.

#### 5.3 유형별 보조 payload로 확장할 수 있다

M4에서 토큰 기억이 semantic에는 충분하지만 grounding·표·잔글씨에 약하다고 확인되면,
토큰 초안과 작은 좌표·OCR·시각 payload를 함께 저장하는 계층형 기억으로 확장할 수
있다. 이는 실패 유형을 확인한 뒤에만 연다.

### 6. 핵심 리스크

| 리스크 | 의미 | 검증·대응 |
|---|---|---|
| 검증 기준 과적합 | 과거 질문이나 고정 probe만 통과하고 새 질문에는 실패할 수 있음 | M3 source self-fidelity와 held-out을 함께 보고 귀속 |
| 정보 병목의 재등장 | 짧은 초안도 결국 어떤 잔글씨·표·세부사항을 버릴지 선택해야 함 | 길이별 실패 곡선, M3 T-label, M4 정보 유형별 손실 측정 |
| grounding 표현력 부족 | 자연어 토큰만으로 좌표·배치·비텍스트 근거를 정밀하게 보존하기 어려움 | M4에서 유형별 측정 후에만 보조 payload 추가 |
| write 비용 증가 | 생성·검증·보강 반복이 여러 forward를 요구할 수 있음 | 검증 횟수 상한과 GPU-초/CPU-초를 M6에서 회계 |
| speculative 명칭 과장 | 표준 speculative decoding의 exactness 보장을 제공하지 않음 | “speculative-inspired verified memory”로 한정 |
| 선행연구 중복 | gist/summary token, context distillation과 인접 | training-free 검증 루프의 차별성이 있는지 스쿱 맵 확인 |

### 7. 구현 전에 고정할 결정

- 초안은 직렬화 가능한 고정 vocabulary token으로 제한한다. 학습된 soft/gist token은
  training-free 주 해석에서 제외하고 별도 방법으로 분류한다.
- 초안 생성 모델과 검증 target 모델
- generic 검증 prompt·metric·통과 임계값
- 초안 최대 token/byte budget과 보강 반복 상한
- 원본 이미지를 폐기할지 fallback용으로 보관할지
- grounding·OCR 보조 payload 허용 시점과 형식
- model/tokenizer revision 변경 시 기억 migration 규칙

### 8. 최소 검증 계획

1. **기본안 고정**: G02 T_visual을 자연어 초안으로 사용하고, 미래 평가 질문·이미지와
   겹치지 않는 generic probe bank와 검증 기준을 동결한다.
2. **검증 효과 분리**: 같은 생성기와 draft byte cap에서 무검증 T_visual, 1회 검증,
   반복 보강을 비교한다.
3. **표현 비교**: `IMAGE`, `FULL-KV`, `T_visual`, 동일-byte 무검증 draft, verified draft,
   `KV 20%`를 함께 비교한다. nominal token 수만으로 공정성을 주장하지 않는다.
4. **M3 귀속**:
   - target-aware 동일-byte draft도 실패하면 표현력·용량 문제로 본다.
   - target-aware draft는 성공하지만 합법적인 write-time draft가 실패하면 생성기 또는
     verifier 일반화 문제로 본다.
   - source self-fidelity는 성공하지만 T3/T4에서만 실패하면 evidence/type transfer
     문제로 본다.
5. **M4 안전성**: semantic/OCR/layout/grounding/icon/count별 손실을 보고한다.
6. **M6 회계**: 저장 bytes, 초안 생성·검증 GPU/CPU-초, read prefill 시간, peak RAM,
   background queue를 기록한다.
7. 주 해석이 유망할 때만 대안 A/B와 보조 payload를 별도 실험으로 연다.

### 9. 판정 기준

정량 임계값은 실행 전에 동결한다.

- **검증 루프 통과**: 같은 형식·길이의 무검증 T_visual보다 held-out 품질이 좋아야 한다.
- **표현 채택 후보**: source self-fidelity를 만족하고, actual bytes와 총비용을 포함한
  비교에서 KV·이미지 기억 대비 M6 비지배점이어야 한다.
- **조건부 확장**: semantic은 유지하지만 grounding 등 특정 유형만 실패하면 보조
  payload를 쓰는 계층형 후보로 재분류한다.
- **범위 재분류**: source 질문에서만 통과하면 source-aware 기억으로, 원본 fallback이
  필수면 hierarchical read-time 시스템으로 분류한다.
- **기각**: 검증이 무검증 초안보다 낫지 않거나, held-out 손실·write 비용이 저장 이득을
  상쇄하면 주 방법으로 채택하지 않는다.

### 한 줄 요약

> KV라는 계산 결과 대신 짧은 토큰 초안을 저장하되, write-time에 원본 문맥과의 행동
> 일치를 검증하고 부족하면 보강한다. 핵심은 작은 저장량 자체가 아니라, 검증이 같은
> 길이의 무검증 T_visual보다 미래 질문 성능을 실제로 높이는지다.

## 아이디어 4. 2층 시각 기억: 이미지 아카이브 + KV 캐시 + read-시점 판정기 (2026-08-16)

**한 줄**: 완전성은 원본 이미지(공짜, KV의 1/27)가 담당하고, 속도는 중요 토큰 KV
조각(5~10%)이 담당하며, 새 질문마다 estimator가 "캐시로 충분한가"를 답하기 전에
판정해 모자라면 이미지를 다시 읽는다. CPU 캐시-디스크 계층과 동형.

**근거가 되는 실측** (FINDINGS 결과 15~19):
- 선택은 원리적으로 불완전 (write-time 천장 격차 28~55%p, 3도메인) → 선택 품질에
  정확도를 걸지 않고 "적중률"만 걸게 하는 구조가 필요.
- 실패는 근거 이동에 집중 → read-시점 감지 가능 (AUROC 0.72~0.85).
- 감지-폴백은 무작위 폴백을 전 구간 지배 (40% 폴백으로 격차 62% 회수).
- 이미지 파일이 5% KV보다 27배 작음 → 아카이브 보관은 공짜, KV의 존재 이유는
  오직 재계산 시간(TTFT 20배).
- '중요 토큰'의 정의는 도메인 체제를 따름: 문서=s5, GUI=h2o, 자연=최소 예산.

**남은 검증** (우선순위 순):
1. 배포형 estimator — 전체 KV 없이(보관 조각만으로) 계산되는 커버리지 근사 신호가
   진짜 coverage의 판정력(AUROC ~0.75+)을 유지하는가.
2. 끝-끝 파이프라인 평가 — [체제 인식 선택 → 저장 → 판정 → 폴백] 전체를
   항상-이미지/항상-KV/KVzip류 위에 붙였을 때·뗐을 때로 비교 (P4 성립 문장).
3. 3단 계층 (원본 ↔ 축소 이미지 중간 폴백) — 축소로 충분한 질문 유형의 경계
   (작은 글자 위험, IMAGE-DOWNSCALED 기준선과 연결).
4. 경제성 조건 — 이득이 성립하는 재사용률 임계값 곡선 (M6 잔여).

**관련**: 아이디어 2(CPU-측 기억 관리 — 아카이브/캐시 승격·강등 로직과 결합 가능),
docs/IDEA-VALIDATION-METHODS.md의 P1~P7 공통 규칙 적용.

## 아이디어 5. 자기 생성 질문 앙상블 (self-QA pool) — 이력 없는 화면의 write-time 신호 (2026-08-16)

**한 줄**: 저장 시점에 모델이 그 화면에 대한 질문 K개를 스스로 생성(디코딩)하고,
각 질문을 prefill에 되먹여 얻은 attention의 합집합으로 남길 토큰을 고른다.
실제 질문 이력이 없는 화면에서 h2o(이력 기반)를 대신하는 신호.

**위치**: 자기 서술(s5, GUI에서 random 이하)과 실제 이력(h2o, GUI 최강) 사이의
빈 칸. 실제 질문 3개 pool의 held-out 일반화(+21.1%p, 결과 14)를 이력 0건으로
흉내 내는 시도. 시각 KV 선행연구에 없는 조합.

**상한(우리 실측이 예고)**: write-time 신호이므로 천장(s1)은 못 넘음. 기대 상한 =
실제-이력 pool 수준(GUI +21%p). 문서는 실제 pool도 과적합(0%p)이라 효과 없을 것으로
예상 — 도메인 체제 예측이 맞는지 자체가 검증 항목. 아이디어 4의 캐시 층
적중률을 올리는 부품이지 판정기·아카이브의 대체가 아님.

**실험 설계 (기존 인프라 재사용)**: m3 held-out 러너의 "실제 질문 1-3 s1 합집합"
자리에 "자기 생성 질문 K개(K=3,5,8) s1 합집합"을 넣고, 같은 화면에서
RANDOM_MATCHED / S5_MATCHED / 실제 UNION과 4파전 비교. 질문 생성 프롬프트
다양화(정보 추출형/위치형/비교형) 여부도 축으로. GPU 반나절.

**관련**: [[아이디어 3]](디코딩 산출물의 재활용이라는 점에서 동족),
아이디어 4의 '중요 토큰' 선택기 후보.
