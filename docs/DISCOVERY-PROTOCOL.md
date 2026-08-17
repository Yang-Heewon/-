# 승격(discovery) 프로토콜 동결 — 2026-08-18

> 이 문서에 적힌 값은 실행 전에 동결되며, 실행 후 변경하지 않는다.
> 파일럿(smoke)과의 차이: 표본이 새것이고 더 크며, 규칙을 미리 정했다.

## 동결 값

| 항목 | 값 |
|---|---|
| 표본 | 도메인당 **신규 300화면** (파일럿 150화면과 겹침 0 — exclude-manifest로 강제) |
| 선택 seed | **777** (파일럿 42와 다른 값, 이후 불변) |
| 도메인 | GUI(ScreenQA) + 자연(GQA). 문서(DocVQA)는 근거 좌표 부재로 파일럿 지위 유지 |
| 모델 | Qwen2.5-VL-7B(주) + Qwen3-VL-8B(재현) — 파일럿과 같은 고정 revision |
| 예산 | 5%, 20% (bytes 기준, sparse 회계 — 파일럿과 동일) |
| 실험 | 기준선(gate EM≥0.30) → 사다리(7 selector) → 교차 → held-out → **끝-끝 캐스케이드** |
| seed 민감도 | held-out만 seed {42, 777, 1234} 3회 (무작위 arm이 있는 유일한 실험) — 나머지는 greedy 결정적 |
| 지표 | EM(주)·ANLS(보조), FULL 정답 조건부 유지율, 이미지 cluster bootstrap 95% CI (n=10,000) |
| estimator | a4(첫 답 토큰 margin), 임계값 τ는 **discovery GUI 절반에서 재보정 후 동결**, 나머지 전부에 이전 적용 |
| 초안 재사용 | greedy 전일치 수락(보증형) 주 결과 + margin 임계 곡선 보조 |
| 판정 기준 | 파일럿 결과의 "방향 유지"(부호 동일 + CI 겹침 허용). 방향이 뒤집히면 그 자체를 결과로 보고 |

## 시각 검증 의무

새 manifest의 자동 쌍 라벨은 도메인당 표본 12쌍(T2 4 / partial 4 / T3 4)을
렌더해 **사람이(=담당 에이전트가) 이미지를 직접 읽어** 판정 후에만 실험에 투입한다.

## 산출물 명명

manifest: `{screenqa,gqa}_discovery{,.meta,_pairs}.jsonl` ·
결과: `results/discovery/…` (smoke와 디렉토리 분리) ·
FINDINGS에는 "결과 D1…"로 파일럿과 구분해 기록.
