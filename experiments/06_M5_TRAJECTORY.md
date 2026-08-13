# M5 — 시간·상태 변화·다중 기억

**상태:** `PLANNED` (`HERMES` upstream source만 고정)  
**질문:** 단일 이미지에서 성립한 memory가 실제 trajectory와 여러 memory block에서도 성립하는가?
**실행 계약:** [configs/m5.yaml](configs/m5.yaml)

## 1. 실행 gate

M1–M4에서 단일 이미지 재사용 조건이 하나 이상 성립할 때만 실행한다. 단일 이미지가 깨지는
상태에서 trajectory를 추가하면 위치·압축·staleness가 섞인다.

## 2. 실험 축

| 축 | 조건 |
|---|---|
| 시간 거리 | write frame과 read step 간 거리 |
| 상태 변화 | 같은 영역의 content 유지 vs 변경 |
| 프레임 중복 | 유사 token 유지 vs 실제 제거 |
| staleness | 과거와 현재 정보 일치 vs 충돌 |
| block 합성 | 1/2/4 independent vs joint-prefill |
| interference | 관련 block만 vs 관련+무관 block |

압축 payload를 비교할 때는 20/40/60/80% keep를 유지한다. 계산량 때문에 budget을 하나로
줄이려면 M2–M4에서 사전 결정한 `B*`를 사용하고, 전체 grid를 생략한 범위를 결과에 명시한다.

pre-RoPE cosine 유사도는 관찰값이다. 제거 가능성은 실제 제거 후 action metric으로만 판정한다.

## 3. 데이터 우선순위

1. Multimodal-Mind2Web
2. AndroidControl
3. GUIOdyssey
4. OSWorld/OSWorld-Verified — 최종 확인, 비용이 큼

AndroidControl과 GUIOdyssey는 현재 공식 공개 경로가 있지만, 과거 로컬 점검에서는 접근 불가로
기록됐다. 순위만으로 선택하지 않고 다운로드, 라이선스, screenshot/action schema, parser,
evaluation 재현을 먼저 smoke한다.

## 4. 사용자가 수정·결정할 부분

| 결정 ID | 결정 | 판단 기준 |
|---|---|---|
| M5-01 | 첫 trajectory dataset | screenshot/action 접근성·metric·재현성 |
| M5-02 | 시간 거리와 block 수 | 실제 episode 길이와 V100 memory |
| M5-03 | T_episode 구성 | q/action/outcome/trajectory 중 저장 범위 |
| HERMES | upstream 그대로 vs Qwen adapter 수정 | 비교 공정성과 구현 가능성 |

## 5. 현재 실행 가능 범위

`third_party/HERMES` source와 commit은 고정돼 있지만 로컬 데이터·Qwen2.5-VL trajectory
evaluation과 통합되지 않았다. upstream 명령을 곧바로 본 baseline 결과로 사용하지 않는다.

## 6. 목표 runner

```bash
python -m vlm_diagnosis.exps.m5_trajectory \
  --config experiments/configs/m5.yaml \
  --manifest experiments/manifests/m5_trajectories.jsonl
```

필요 구현:

- episode/frame loader
- dataset access/schema smoke report
- current-vs-stale conflict annotation
- multi-block position/metadata assembly
- action/target/episode metric
- environment failure와 memory failure 분리

## 7. 판정

| 결과 | 해석 |
|---|---|
| content 불변인데 시간만으로 하락 | 위치·문맥 누적 또는 장거리 사용 문제 |
| 상태 변경 때만 과거 memory가 해침 | staleness/update policy 문제 |
| 유사하고 제거해도 유지 | 측정 범위에서 인과적 시간 중복 |
| 유사하지만 제거 시 하락 | 표현 유사도≠정보 중복 |
| independent concat 실패, joint 성공 | block 간 미계산 상호작용/위치 충돌 |
| 무관 block 증가에 따라 하락 | interference/read policy 문제 |
| QA 유지, action 하락 | 정적 QA가 실제 agent 실패를 가림 |

## 8. 완료 조건

- action 또는 episode metric으로 평가
- environment reset/website failure를 별도 상태로 기록
- 같은 content·offset control 포함
- 각 block의 provenance와 position metadata 기록
