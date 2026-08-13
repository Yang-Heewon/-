# Config 작성 규칙

이 디렉터리의 YAML은 현재 **사람이 읽는 실행 계약**이다. 모든 runner가 아직 YAML을 직접
소비하는 것은 아니다. 단계가 `READY`가 될 때 runner CLI가 이 config를 읽도록 연결한다.

## 규칙

- `TBD` 또는 `null`은 사용자가 결정하지 않은 필수 값이다.
- 본실험 전에 모든 `required_decisions`를 해결한다.
- 권장 숫자는 `proposed_*`에 두고 실제 필드는 `TBD/null`로 남겨 결정 완료로 오인하지 않는다.
- 모든 config에 `seed`, `runner`, `run_kind`, `output`을 둔다.
- keep budget은 항상 `[0.2, 0.4, 0.6, 0.8]`을 사용한다.
- 5%·10%는 M2-A diagnostic의 조건부 extreme budget이며 primary grid를 바꾸지 않는다.
- sample ID는 config에 직접 길게 넣지 않고 `manifests/*.jsonl`을 참조한다.
- smoke/discovery/confirmation config를 덮어쓰지 말고 별도 revision으로 보존한다.
- 결과에는 사용한 config의 hash 또는 복사본을 저장한다.
- `run_kind`와 output 디렉터리가 일치해야 한다.
- PLANNED 단계의 manifest/runner 부재는 unresolved resource이고 READY 단계에서는 오류다.

검사:

```bash
# 설계 중: 미결정 경로를 출력하되 성공 종료
python -m vlm_diagnosis.scripts.validate_experiment_configs --allow-unresolved

# 본실험 직전: TBD/null이 있으면 실패
python -m vlm_diagnosis.scripts.validate_experiment_configs
```

## 파일

```text
configs/
├── m0.yaml
├── m1.yaml
├── m2a.yaml
├── m3.yaml
├── m4.yaml
├── m2b.yaml
├── m5.yaml
├── m6.yaml
└── m7.yaml
```
