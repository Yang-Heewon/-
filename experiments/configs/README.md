# Config 작성 규칙

이 디렉터리의 YAML은 현재 **사람이 읽는 실행 계약**이다. 모든 runner가 아직 YAML을 직접
소비하는 것은 아니다. 단계가 `READY`가 될 때 runner CLI가 이 config를 읽도록 연결한다.

## 규칙

- `TBD` 또는 `null`은 사용자가 결정하지 않은 필수 값이다.
- 본실험 전에 모든 `required_decisions`를 해결한다.
- keep budget은 항상 `[0.2, 0.4, 0.6, 0.8]`을 사용한다.
- sample ID는 config에 직접 길게 넣지 않고 `manifests/*.jsonl`을 참조한다.
- smoke/discovery/confirmation config를 덮어쓰지 말고 별도 revision으로 보존한다.
- 결과에는 사용한 config의 hash 또는 복사본을 저장한다.

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
