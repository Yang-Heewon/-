# M4 — 어떤 시각 정보가 선택적으로 사라지는가

**상태:** `PLANNED`  
**질문:** 평균 성능 뒤에 OCR·공간·아이콘 등 특정 정보의 선택적 손실이 있는가?
**실행 계약:** [configs/m4.yaml](configs/m4.yaml)

## 1. 선행 gate

- M2-A 또는 M3에서 실제 task metric 차이가 난 조건이 있음
- 같은 이미지에서 정보 유형을 비교할 PCTD annotation 준비
- IMAGE base가 각 task를 판별할 만큼 충분함

## 2. 정보 유형

- OCR 문자열·숫자
- semantic entity/state
- layout·상대 위치
- coordinate/click grounding
- icon·비텍스트 affordance
- count·반복 요소

서로 다른 데이터셋의 평균만 비교하면 정보 유형과 dataset 난이도가 섞인다. M4의 중심은
같은 이미지에 여러 task를 부착한 `PCTD`다.

## 3. PCTD 설계

권장 시작점:

```text
document 50
natural image 50
mobile UI 50
web/desktop GUI 50
```

각 이미지에 가능한 정보 유형 중 최소 4개 질문을 만들고 다음을 기록한다.

```yaml
answer: string|coordinate|action
acceptable_answers: list
evidence_bbox_or_element: list
evidence_size: float
requires_text: bool
requires_spatial: bool
requires_icon: bool
pair_labels: [T0, T1, T2, T3, T4]
```

## 4. 비교 조건

- IMAGE
- T_visual
- T_episode — 해당 이미지에 실제 episode가 있을 때만
- FULL-KV
- M2-A에서 차이가 난 sparse 조건
- M2-B에서 차이가 난 family 조건은 M2-B 실행 후 추가

비압축 payload 조합을 무분별하게 늘리지는 않지만, 선택된 **압축 조건**은
20/40/60/80% keep 네 점을 모두 실행한다. 정보 유형별 `B*` 차이 자체가 M4 결과다.

## 5. 사용자가 수정·결정할 부분

| 결정 ID | 결정 | 판단 기준 |
|---|---|---|
| M4-01 | PCTD 이미지 수와 출처 | 도메인 균형·라이선스·annotation 비용 |
| M4-02 | evidence 단위와 검수 방법 | bbox/UI element가 실제 답을 지지하는지 |
| M4-03 | 외부 dataset routing | 발견 정보 유형마다 2개 이상, M7 전에 고정 |
| G02 | T_visual 생성법 | 약한 caption baseline 방지 |

## 6. 목표 runner

현재 통합 runner는 없다.

```bash
python -m vlm_diagnosis.exps.m4_information_types \
  --config experiments/configs/m4.yaml \
  --manifest experiments/manifests/m4_pctd_discovery.jsonl
```

구현해야 할 것:

- PCTD schema validator
- QA/grounding/action metric router
- evidence coverage와 selected visual token 매핑
- domain×task interaction report

## 7. 판정

| 결과 | 해석 |
|---|---|
| 모든 유형 retention 유사 | 정보 선택적 실패 가설 약함 |
| target probe도 grounding만 고예산 | 공간 정보가 KV에 분산됐을 가능성 |
| target probe는 성공, SOTA만 grounding 실패 | selector가 작은 공간 근거를 과소선택 |
| evidence coverage 높지만 task 실패 | selection보다 사용·위치·상호작용 문제 |
| T_visual semantic 성공, grounding 실패; KV 성공 | latent/visual payload의 고유 가치가 grounding에 있음 |
| T_visual≈KV 전 유형 | 비싼 KV memory의 동기가 약함 |

## 8. 완료 조건

- 같은 이미지 내 task-type effect와 dataset effect를 분리
- 각 오류를 `not_selected / selected_not_used / base_failure / uncertain`으로 라벨링
- 외부 dataset 결과를 단일 평균으로 합치지 않고 domain별 보고
- PCTD discovery와 confirmation 이미지가 겹치지 않음
