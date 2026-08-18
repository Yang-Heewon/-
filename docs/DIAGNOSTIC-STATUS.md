# Agent visual-memory diagnosis: evidence status

> 갱신일: 2026-08-18  
> 목적: 새로 만든 진단 실험을 기존 discovery 결과나 논문 claim과 섞지 않고,
> 현재 **무엇을 실제로 확인했는지**와 **아직 무엇을 모르는지**를 구분한다.

## 1. 지금의 결론

현재까지의 새 실험은 논문 Method를 확정하지 않는다. 대신 다음 두 사실을 분명하게 했다.

1. 원본 접근을 막고 실제 저장 byte를 재는 공통 평가 경로를 만들 수 있다.
2. durable visual memory, retrieval/interference, temporal state, action utility는 서로 다른
   실패 원인이므로 한 개의 `cross-question QA` 수치로 대체할 수 없다.

현재 가장 중요한 미결정 질문은 다음이다.

> 실제 데이터의 동일-byte 조건에서 `compressed raster`, `OCR/layout text`,
> `projected visual state`, `KV` 중 무엇이 어떤 미래 정보 유형을 보존하며,
> 여러 기억·상태 갱신·행동까지 갔을 때 병목이 어디로 이동하는가?

따라서 지금은 Method를 고르는 단계가 아니라 **측정 계약과 기제 probe를 통과한 단계**다.

## 2. 증거 등급

| 등급 | 의미 | 현재 용도 |
|---|---|---|
| contract validation | source denial, hash, actual bytes, runner가 의도대로 작동하는지 검사 | 구현 신뢰성 |
| controlled mechanism | 합성 자료에서 한 실패 원인만 조작 | 인과적 sanity check |
| real-data smoke | 실제 이미지의 작은 표본 | 난이도·실행 가능성 gate |
| discovery | 사전 동결한 충분한 표본, CI와 cluster 단위 분석 | Method 선택과 가설 형성 |
| confirmation | discovery와 겹치지 않는 자료·독립 model family | 최종 논문 claim 확인 |

아래의 새 결과는 모두 앞의 세 등급에만 속한다. **새 discovery나 M7 confirmation 결과는 아직 없다.**

## 3. D0/D1 — source-unavailable representation gate

### Formal D0 contract gate

ScreenQA 실제 이미지 3장과 manifest에 물리적으로 고정한 질문 6개에서 동일한 Qwen2.5-VL
reasoner를 사용했다. writer만 원본 이미지를 볼 수 있고 reader는 실제 package와 질문만 받았다.

`source image container`, `OCR layout text`, `projected visual tokens FP16`, `full visual KV`의
네 대표 표현은 모두 다음 strict audit를 통과했다.

- read manifest 6/6 질문 coverage와 중복 key 0
- package bytes/hash 필드 검증
- 결과 PID와 첫 read의 `openat` trace 연결
- 원본 exact path, basename, `data/screenqa_pilot` 접근 0회
- 같은 package를 두 번째 프로세스에서 읽었을 때 6/6 greedy prediction 동일

원본 파일을 실제로 삭제한 것은 아니다. 안전한 접근 금지와 syscall audit로 source unavailable을
구현했으며, pre-opened descriptor·IPC·network까지 부재함을 증명하는 보안 sandbox는 아니다.

### D1 implementation smoke

같은 3장/6문항에 아홉 표현을 연결한 descriptive 결과는 다음과 같다.

| 저장 표현 | 이미지당 평균 실제 payload | 원본 대비 | QA EM | 원본과 exact prediction 일치 |
|---|---:|---:|---:|---:|
| source image container | 95,476 B | 1.00× | 6/6 | 6/6 |
| full visual KV | 70,993,261.3 B | 752.72× | 6/6 | 6/6 |
| projected visual tokens FP16 | 8,803,050.3 B | 93.34× | 6/6 | 6/6 |
| projected visual tokens INT8 | 4,428,737 B | 46.96× | 6/6 | 6/6 |
| projected visual tokens INT4 | 2,238,913 B | 23.74× | 6/6 | 6/6 |
| JPEG, 32 KiB cap | 31,933 B | 0.339× | 6/6 | 6/6 |
| AVIF, 32 KiB cap | 31,716 B | 0.337× | 6/6 | 6/6 |
| OCR plain text | 148.3 B | 0.00156× | 5/6 | 5/6 |
| OCR layout text | 1,149 B | 0.0120× | 6/6 | 5/6 |

이 표가 말할 수 있는 것은 구현 gate에서의 **byte order와 기능 보존 여부**뿐이다.
표본이 3장이므로 representation 우열이나 일반 정확도를 주장할 수 없다. 특히 INT4의 6/6 QA는
작은 답 집합에서의 우연한 보존일 수 있고 reconstructed tensor 오차가 작다는 뜻이 아니다.
또한 이 표는 actual-byte를 기록했지만 **동일-byte 비교가 아니다**. JPEG/AVIF만 같은 32 KiB
상한을 사용했고 text, projected state, KV는 서로 완전히 다른 byte 영역에 있다.

직접 근거:

- strict audit: `results/smoke/source_denial_d0g3q2_audit.json`
- 집계: `results/smoke/memory_frontier_d0g3q2_summary.json`
- 물리 OCR payload: `results/smoke/source_denial_d0g3_ocr_payloads/`
- write/read manifests: `results/smoke/source_denial_d0g3_q2_write.jsonl`,
  `results/smoke/source_denial_d0g3_q2_read.jsonl`

현재 안전한 해석은 다음이다.

> 이 model의 full KV와 projected state는 source image보다 작게 저장하는 archive가 아니다.
> 이들은 read compute를 줄일 가능성이 있는 model-specific hot state다. Durable tier의 우열은
> 더 큰 동일-byte discovery에서 결정해야 한다.

## 4. D2/M4 — 정보 유형별 손실 gate

### 실제 ScreenQA/T4 smoke

실제 화면 4장, 24개 질문에서 source image, JPEG 32 KiB, AVIF 32 KiB를 비교했다.
이 M4 runner는 `strict_source_denial=false`인 real-data smoke다. D0처럼 독립 reader의 원본 접근
금지를 증명하는 결과로 사용하지 않는다.

| 조건 | 전체 score | source-correct 조건부 retention |
|---|---:|---:|
| source image | 21/24 = 0.875 | 21/21 |
| JPEG 32 KiB | 21/24 = 0.875 | 21/21 |
| AVIF 32 KiB | 21/24 = 0.875 | 21/21 |

이 크기에서는 codec 손실보다 base model failure가 먼저 나타났다. 1개 화면을 8 KiB와
384/512/768 px long-side로 더 줄인 gate에서는 JPEG의 state/tab 질문이 source 정답에서
오답으로 바뀌었지만, AVIF 결과가 해상도에 따라 비단조적이었다. 이는 정보 유형 가설을 만들
수 있는 신호일 뿐, `state가 먼저 소실된다`는 근거가 아니다.

### controlled M4 fixture

24개의 합성 mobile UI와 이미지당 6개 질문을 만들었다. text, semantics, layout, grounding,
icon, count 요구를 명시적으로 라벨링한다. 그러나 현재 codec 실행은 이 중 1–2개 이미지만 쓴
ceiling smoke다. `24-image M4 결과`나 유형별 차이 부재를 주장할 수 없다. 이 자료는 표현별
실패 원인과 evaluator를 디버깅하는 용도이며 실제 GUI 분포나 논문 성능을 대표하지 않는다.

직접 근거:

- real smoke: `results/smoke/m4_t4_corrected_4img.shard0.jsonl`,
  `results/smoke/m4_t4_corrected_4img.shard1.jsonl`
- resolution gate: `results/smoke/m4_t4_resolution_gate.jsonl`
- controlled manifest: `experiments/manifests/m4_controlled.jsonl`

## 5. D3/D4 — multiple memories와 temporal state

32개의 합성 episode로 320개 source-free trial을 GPU 2·3에서 평가했다. resume 전후의
네 `openat` trace를 결합한 감사에서 320/320 trial과 160/160 package path를 모두 덮었고,
원본 controlled image 접근은 0회였다. 모든 package는 실제 직렬화된 copy-container이며
320/320 조건이 byte cap 안에서 feasible했다.

### D3: 검색과 inference interference

| 진단 | 조건 | n | current-state EM | 평균 read time |
|---|---:|---:|---:|---:|
| oracle task upper bound | stored candidates 1/2/4 | 각 32 | 모두 32/32 | 약 1.15 s |
| inference interference | distractors 0 | 32 | 32/32 | 1.14 s |
| inference interference | distractors 1 | 32 | 32/32 | 2.13 s |
| inference interference | distractors 3 | 32 | 32/32 | 4.31 s |

이 합성 task에서는 distractor가 정확도를 떨어뜨리지 않았지만 read cost는 이미지 수와 함께
증가했다. `oracle task upper bound`는 relevant memory를 이미 알고 선택한 결과다.
**Recall@k나 실제 retrieval 성능을 측정한 것이 아니다.** 또한 candidate/distractor가 늘 때
total package bytes도 함께 늘었으므로 fixed-total-byte interference 비교가 아니며, composition
arm도 아직 실행하지 않았다.

### D4: state change와 stale capture

| 상태 | memory condition | n | current-state EM | stale capture |
|---|---|---:|---:|---:|
| unchanged | old only | 16 | 16/16 | 0/16 |
| changed | old only | 16 | 0/16 | 16/16 |
| changed | current only | 16 | 16/16 | 0/16 |
| changed | old + current | 16 | 16/16 | 0/16 |
| 모든 cell | no memory | 32 | 0/32 | 1/32 |

통제 자료에서는 time gap 자체보다 content revision이 정답을 결정했다. 그러나 old+current
충돌에서도 모두 성공하도록 화면이 단순하게 설계됐으므로, 실제 Agent의 conflict resolution이
해결됐다는 뜻은 아니다. 이 결과의 역할은 오래된 기억을 단순 재사용하면 `불필요한 reread`가
아니라 **조용한 stale answer**가 생긴다는 evaluator sanity check다.

직접 근거:

- read 결과: `results/smoke/memory_dynamics_copy_read.shard0.jsonl`,
  `results/smoke/memory_dynamics_copy_read.shard1.jsonl`
- 집계 및 combined trace audit: `results/smoke/memory_dynamics_copy_summary.json`
- strict read manifests: `results/smoke/memory_dynamics_read.jsonl`
- package manifest: `results/smoke/memory_dynamics_packages.copy.jsonl`
- source-denial audit: `results/smoke/memory_dynamics_copy_read.shard0.openat.log`,
  `results/smoke/memory_dynamics_copy_read.shard1.openat.log`,
  `results/smoke/memory_dynamics_copy_read.full_shard0.openat.log`,
  `results/smoke/memory_dynamics_copy_read.full_shard1.openat.log`

## 6. D5 — action utility 상태

click/type/scroll 각 8개, 총 24개의 controlled offline episode와 old/current 화면 48장을
생성했다. 과거 화면의 cue와 현재 화면의 actionable target을 결합해야 다음 action을 고를 수 있고,
이미 실행되어 무효화된 과거 action도 명시한다.

Qwen2.5-VL FP16 projected visual package를 이용한 strict source-denial v2 evaluator로 4개 arm,
총 96 trial을 실행했다. 두 reader PID와 `openat` trace가 일치했고 원본 PNG 접근은 0회였다.

| action type | arm | n | strict type+args | target success | strict full action | stale replay |
|---|---|---:|---:|---:|---:|---:|
| click | ordered history | 8 | 8/8 | 8/8 | 8/8 | 0/8 |
| click | current only | 8 | 8/8 | 3/8 | 3/8 | 0/8 |
| type | ordered history | 8 | 8/8 | 0/8 | 0/8 | 0/8 |
| type | current only | 8 | 0/8 | 7/8 | 0/8 | 0/8 |
| scroll | ordered history | 8 | 0/8 | 0/8 | 0/8 | 8/8 |
| 모든 type | old only | 24 | 8/24 | 0/24 | 0/24 | 24/24 |

이 결과는 action utility가 단일 점수가 아님을 보여주는 controlled diagnostic이다.

- click에서는 과거의 saved label이 있으면 올바른 현재 button을 8/8 선택했다. 그러나 예측 bbox
  IoU@0.5는 0/8이므로 semantic selection과 spatial grounding이 분리됐다.
- type에서는 과거 token과 action arguments를 8/8 회상했지만 현재 text field grounding이 0/8이었다.
- scroll에서는 과거 cue와 현재 위치를 결합하지 못하고 8/8 모두 이미 끝난 `OPEN`을 재생했다.
- old-only는 모든 type에서 24/24 stale completed action을 재생했다.

v1 prompt는 `arguments:{...}`만 제시해 strict argument score를 인위적으로 0으로 만든 artifact가
있었다. v2는 action별 exact schema와 defaults를 먼저 동결한 새 실행이며 v1 결과를 덮어쓰지 않았다.

이 fixture는 **real trajectory가 아니다**. 환경 실행, sequence success, episode success를
측정하지 않으며 production Agent claim을 허용하지 않는다.

직접 근거:

- fixture: `experiments/manifests/action_proxy_controlled.jsonl`
- v2 results: `results/smoke/action_proxy_eval_v2.shard0.jsonl`,
  `results/smoke/action_proxy_eval_v2.shard1.jsonl`
- package manifest: `results/smoke/action_proxy_projected_fp16_packages.jsonl`
- trace: `results/smoke/action_proxy_eval_v2.shard0.openat.log`,
  `results/smoke/action_proxy_eval_v2.shard1.openat.log`

## 7. 정확히 알고 있는 것과 아직 모르는 것

### 확인된 정보

1. source-unavailable/actual-byte 계약을 raster, OCR text, projected visual state, KV에 공통으로
   적용할 수 있다.
2. 현재 Qwen2.5-VL package에서는 model-native state가 source image보다 수십~수백 배 크다.
3. OCR text는 극도로 작지만 3-image D0에서도 plain text arm은 일부 visual answerability를 잃었다.
4. 실제 4-image smoke의 32 KiB raster는 base-correct 질문을 보존했으나, 더 공격적인 8 KiB
   축소에서 선택적 failure 후보가 나타났다.
5. 합성 temporal intervention에서 obsolete image만 주면 model은 과거 값을 자신 있게 답했다.
6. 여러 이미지를 동시에 읽으면 이 쉬운 합성 task의 정확도는 유지됐지만 latency가 증가했다.
7. 합성 action proxy에서는 memory-dependent click 선택, type content 회상, spatial grounding,
   scroll reasoning이 서로 다른 성공·실패 양상을 보였다.

### 아직 모르는 정보

1. real discovery에서 정보 유형별 JPEG/AVIF/OCR/embedding/KV 동일-byte frontier
2. 16/64/1000개 memory bank의 실제 Recall@k와 index byte/write cost
3. 찾은 뒤의 자연 이미지·GUI distractor interference와 multi-memory composition
4. 실제 시간 순서에서 update, invalidation, contradictory memory를 처리하는 능력
5. real Agent trajectory의 target/action/sequence/episode success
6. 원본 모델 revision과 다른 model family로 package를 읽는 portability
7. discovery와 겹치지 않는 독립 M7 confirmation

## 8. Method를 선택할 수 있는 조건

Method 이름을 먼저 고르지 않는다. 다음 discovery 결과 중 실제 주 병목이 확인된 뒤 결정한다.

| 발견되는 주 병목 | 정당화되는 Method 축 |
|---|---|
| tiny raster에서 특정 OCR/layout/icon 정보만 선택적으로 소실 | structured text/layout + visual residual hybrid |
| representation보다 relevant episode 검색이 먼저 실패 | provenance/evidence-aware index와 retrieval |
| relevant memory를 찾은 뒤 distractor가 추론을 망침 | gated read 또는 memory routing |
| outdated memory가 현재 상태를 덮음 | versioning, invalidation, conflict-aware update |
| read latency만 문제이고 archive는 충분 | compressed image backing + disposable hot visual state/KV |
| 서로 다른 표현이 reuse·utility별로 다른 Pareto 영역을 가짐 | adaptive representation portfolio |

현재의 가장 보수적인 설계 가설은 `compressed raster source of truth + searchable structured
index + 선택적 hot model-native cache`다. 이것은 아직 최종 Method나 novelty claim이 아니라,
D1–D5가 반증하거나 구체화해야 할 시스템 가설이다.

## 9. 다음 승격 순서

1. 실제 ScreenQA/GQA에서 archive 32/64/128/256 KiB와 hot 1/2/4/8 MiB frontier를 discovery로 실행한다.
2. M4 라벨을 실제 GUI/document 표본에 확장하고 image-cluster CI를 낸다.
3. D3 retrieval을 1/4/16/64 memory로 실행해 retrieval, representation, interference를 분해한다.
4. 실제 temporal pair와 offline trajectory 하나를 재현한다.
5. 위 결과로 bottleneck 하나와 Method를 고른 뒤에만 split·threshold를 동결한다.
6. 겹치지 않는 episode와 독립 model family에서 M7 confirmation을 한 번 수행한다.

전체 측정 계약과 설계는 `docs/DIAGNOSTIC-PROGRAM.md`를 따른다.

## 10. 저장·백업 상태

새 runner, tests, manifests, 이 문서는 현재 workspace에 존재하지만 아직 commit하지 않았다.
`results/smoke/`의 JSONL, trace, tensor/image package는 저장소 정책상 대부분 gitignore 대상이며
GitHub backup으로 간주할 수 없다. 특히 약 286 MiB의 D5 projected package와 약 249 MiB의 D0
model-native package는 현재 로컬 재현 artifact다. 논문 discovery를 시작하기 전에는 code/docs를
검토 후 commit하고, raw result manifest·SHA-256 catalog를 off-machine 저장소에 복제해야 한다.
