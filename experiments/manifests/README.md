# Manifest 규칙

manifest는 실제로 실행할 이미지·질문·episode ID를 고정한다. 데이터 파일 자체를 복사하지
않고 원 데이터의 ID와 로컬 경로만 기록한다.

## 권장 JSONL schema

```json
{"dataset":"ScreenQA","split":"discovery","sample_id":"...","image":"...","question_ids":["..."],"task_types":["OCR","layout"],"pair_labels":["T2","T4"]}
```

## 필요한 manifest

- `m0_sanity.jsonl`
- `m1_canonical.jsonl`
- `m2a_full.jsonl`
- `m2a_diagnostic.jsonl`
- `m3_pairs.jsonl`
- `m4_pctd_discovery.jsonl`
- `m4_pctd_confirmation.jsonl`
- `m5_trajectories.jsonl`
- `m7_confirmation.jsonl`

같은 image/episode가 discovery와 confirmation에 동시에 들어가면 안 된다. manifest를 만든
시점과 생성 seed, 제외 사유를 metadata에 기록한다.

