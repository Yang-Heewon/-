# M3 — 과거 relevance가 미래 질문으로 전이되는가

> **목적:** 과거 질문에 맞춰 보존한 KV가 미래 질문에도 필요한 정보를 유지하는지 검증한다.
>
> **왜 필요한가:** 실제 agent는 미래 질문을 미리 알 수 없으므로, 현재 질문을 아는 선택 방식은 실용성을 과대평가할 수 있다.

**상태:** `PARTIAL`  
**질문:** 미래 질문에서의 실패는 evidence 변화인가, source subset 실패인가, estimator 실패인가?
**실행 계약:** [configs/m3.yaml](configs/m3.yaml)

## 1. 선행 gate

- M2-A에서 최소 하나의 해석 가능한 budget·task 조건 확보
- M3 질문 쌍을 T0–T4로 라벨링
- source self-fidelity 기준 확정
- [m3.yaml](configs/m3.yaml), `m3_pairs.jsonl` 고정

## 2. 질문 쌍 유형

| 유형 | 변화 |
|---|---|
| T0 | 같은 질문 반복 |
| T1 | paraphrase, 같은 답·근거 |
| T2 | 다른 질문, 같은 근거 |
| T3 | 같은 이미지의 다른 근거 |
| T4 | semantic/OCR ↔ layout/grounding |

임의 K×K 평균만 보고하지 않는다.

현재 보존된 파일에는 과거 T5의 정의가 없다. G10 결정 전에는 임의 T5를 만들지 않으며, 원 정의를
찾으면 T0–T4 흡수 또는 descope를 기록한다.

## 3. 집합과 교차평가

고정 budget `B∈{20,40,60,80}%`에서 만든다.

- `S_w=A_w(B)`: 과거 질문·정답 기반 source probe subset
- `S_r=A_r(B)`: 미래 질문·정답 기반 target probe subset
- `F_w(B)`: 실제 published write-time selector subset
- `S_rand(B)`: random subset

각 집합을 두 질문에 모두 평가한다.

| subset | q_w | q_r |
|---|---|---|
| S_w | source self-fidelity | forward transfer |
| S_r | reverse transfer | target capacity |
| F_w | estimator self-fidelity | 실제 write→read |
| S_rand | random lower bound | random lower bound |

핵심 gap:

```text
capacity_gap(q_r)  = IMAGE(q_r) - Score(q_r; S_r)
relevance_gap(w→r) = Score(q_r; S_r) - Score(q_r; S_w)
estimator_gap(w)   = Score(q_w; S_w) - Score(q_w; F_w)
end2end_gap(w→r)   = Score(q_r; S_r) - Score(q_r; F_w)
```

`S_w→q_w`가 통과한 경우에만 relevance gap을 해석한다.

## 4. 사용자가 수정·결정할 부분

| 결정 ID | 결정 | 판단 기준 |
|---|---|---|
| M3-01 | T0–T4 label guide와 검수자 | wording과 evidence 이동을 재현 가능하게 구분 |
| M3-02 | source self-fidelity 통과 기준 | task 의미와 M0 noise보다 큰 equivalence band |
| M3-03 | 이미지당 질문 쌍 수 | 유형별 효과를 보되 이미지 의존성 유지 |
| budget | 어느 B를 핵심 그림으로 쓸지 | 결과 전에는 전 grid 실행, 그림 선택은 기준 명시 |

## 5. 현재 실행 가능한 legacy 명령

```bash
bash vlm_diagnosis/scripts/launch_d4_mini.sh
python -m vlm_diagnosis.exps.d4_mini --aggregate
```

현재 runner는 20% keep의 K×K teacher-forced logp만 제공한다. 더구나 기존 출력은 FULL과
keep-set의 mask·position 경로가 달랐고 NaN shard가 있으므로 참고치로도 사용하지 않는다.
source/target probe 교차평가와 T0–T4 label을 갖춘 새 runner가 필요하다.

## 6. 목표 runner

```bash
python -m vlm_diagnosis.exps.m3_transfer \
  --config experiments/configs/m3.yaml \
  --manifest experiments/manifests/m3_pairs.jsonl
```

## 7. 판정

| 결과 | 해석 |
|---|---|
| S_r→q_r 실패 | 미래 relevance가 아니라 capacity/probe 문제 |
| S_w→q_w 실패 | source probe가 자기 질문도 보존 못 함; transfer 결론 금지 |
| S_w→q_w 성공, S_w→q_r 실패, S_r→q_r 성공 | 과거 relevance가 미래 evidence를 덮지 못함 |
| S_w는 교차성공, F_w만 실패 | estimator 문제 |
| T0/T1/T2 성공, T3만 실패 | wording보다 evidence 이동이 핵심 |
| set overlap 낮지만 양쪽 성공 | 대체 가능한 근거; Jaccard만으로 실패 판단 금지 |

`S_r`은 성공하지만 실제 selector만 실패한 경우에만 G09의 조건부 진단을 연다.

- D2-style: length/sink/position/token-count 통제 후 signal 재측정
- D3-style: 실제 selector signal과 answer-aware importance의 정렬 측정

## 8. 출력과 완료 조건

```text
results/discovery/m3_cross_eval.jsonl
results/discovery/m3_transfer_report.md
```

완료하려면 각 질문 쌍에 T-label, 네 subset, 2×4 task score, actual bytes, source fidelity
판정이 있어야 한다. K×K matrix는 최종 요약 그림으로만 사용한다.
