# 실험 실행 인덱스

이 디렉터리는 Phase 1 실험을 **실행 순서대로** 관리한다. 각 문서는 같은 구조를 사용한다.

1. 질문과 동기
2. 선행 gate
3. 고정 조건
4. 사용자가 수정·결정할 부분
5. 현재 실행 가능한 명령
6. 목표 runner와 구현 부족분
7. 출력과 판정 규칙
8. 다음 단계 조건

## 상태 표기

- `READY`: 본실험 runner·metric·config가 준비됨
- `PARTIAL`: smoke 또는 일부 구성요소만 실행 가능
- `PLANNED`: 설계만 있고 runner가 없음
- `BLOCKED`: 앞 단계 gate 실패로 해석 불가
- `COMPLETE`: 사전 판정과 산출물까지 완료

`PARTIAL`의 명령은 smoke 또는 legacy 결과만 만든다. 문서가 명시적으로 `READY`가 되기 전에는
그 결과를 discovery나 confirmation 결과로 승격하지 않는다.

## 실행 순서

| 순서 | 단계 | 상태 | 문서 | config |
|---:|---|---|---|---|
| 0 | M0 측정 계약 | PARTIAL | [00_M0_MEASUREMENT.md](00_M0_MEASUREMENT.md) | [m0.yaml](configs/m0.yaml) |
| 1 | M1 저장·재사용 경계 | PARTIAL | [01_M1_STORAGE_REUSE.md](01_M1_STORAGE_REUSE.md) | [m1.yaml](configs/m1.yaml) |
| 2 | M2-A 고정 예산 sparse | PARTIAL | [02_M2A_FIXED_BUDGET.md](02_M2A_FIXED_BUDGET.md) | [m2a.yaml](configs/m2a.yaml) |
| 3 | M3 미래 질문 전이 | PARTIAL | [03_M3_TRANSFER.md](03_M3_TRANSFER.md) | [m3.yaml](configs/m3.yaml) |
| 4 | M4 정보 유형 | PLANNED | [04_M4_INFORMATION_TYPES.md](04_M4_INFORMATION_TYPES.md) | [m4.yaml](configs/m4.yaml) |
| 5 | M2-B family stress | PARTIAL | [05_M2B_FAMILY_STRESS.md](05_M2B_FAMILY_STRESS.md) | [m2b.yaml](configs/m2b.yaml) |
| 6 | M5 궤적·다중 기억 | PLANNED | [06_M5_TRAJECTORY.md](06_M5_TRAJECTORY.md) | [m5.yaml](configs/m5.yaml) |
| 7 | M6 시스템 성립 영역 | PARTIAL | [07_M6_SYSTEM.md](07_M6_SYSTEM.md) | [m6.yaml](configs/m6.yaml) |
| 8 | M7 확인 실험 | PLANNED | [08_M7_CONFIRMATION.md](08_M7_CONFIRMATION.md) | [m7.yaml](configs/m7.yaml) |

M0 전에 P0 계약을 확인한다.

- root Git commit 존재
- third-party commit pin 존재
- config의 run kind와 output 경로 일치
- 필수 decision ID와 seed 존재
- READY 단계의 manifest·runner·revision 존재

## 매 단계의 작업 순서

```text
1. 이전 단계 gate 확인
2. 해당 MD의 '사용자가 결정할 부분' 확정
3. configs/*.yaml의 TBD/null 제거
4. validator가 보고한 manifest·runner resource 준비
5. manifest에 실제 sample ID와 dataset revision 고정
6. smoke 실행
7. 결과 schema와 finite 검사
8. discovery 실행
9. 사전 해석표에 따라 E/S/N/I 판정
10. DECISIONS.md와 결과 report에 결정 기록
```

미결정 항목 확인:

```bash
python -m vlm_diagnosis.scripts.validate_experiment_configs --allow-unresolved
```

본실험 직전에는 `--allow-unresolved` 없이 실행한다. 종료 코드 0이어야 한다.

## 지금 바로 실행 가능한 경로 검사

```bash
cd /root/research/heewon/VLM
python -m unittest discover -s tests -v
python -m vlm_diagnosis.scripts.smoke_sanity
python -m vlm_diagnosis.exps.m2_family_baselines --limit 1 --device cuda:0
python -m vlm_diagnosis.scripts.smoke_quantized_cache --backend hqq --nbits 4 --device cuda:0
```

각 명령이 증명하는 범위는 해당 단계 문서를 따른다. 특히 마지막 두 명령은 paper result가
아니라 quality/physical path smoke다. `smoke_sanity`와 mask 기반 quality smoke는 현재 fp16
NaN blocker를 재현할 수 있으며 PASS가 보장된 명령이 아니다.

문서에 적힌 목표 runner 중 아직 없는 module:

```text
m0_measurement, m1_storage_reuse,
m2a_fixed_budget, m2a_answer_probe,
m3_transfer, m4_information_types,
m2b_family_stress, m5_trajectory,
m6_system, m7_confirmation
```

빈 module이나 빈 manifest로 이 목록을 숨기지 않는다. 앞 단계 gate 순서대로 구현한다.

## 공통 문서

- [SHARED_PROTOCOL.md](SHARED_PROTOCOL.md): metric, budget, schema, 통계, 결과 상태
- [DECISIONS.md](DECISIONS.md): 사용자가 확정해야 하는 값
- [MODEL_MATRIX.md](MODEL_MATRIX.md): 모델 계보·position·fusion 기준과 M7 routing
- [configs/README.md](configs/README.md): config 작성 규칙
- [manifests/README.md](manifests/README.md): 표본 ID와 split 고정 규칙
