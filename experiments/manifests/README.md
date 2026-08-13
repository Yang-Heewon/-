# Manifest 규칙

manifest는 실제로 실행할 이미지·질문·episode ID를 결과를 보기 전에 고정한다. 빈 파일로 config
검사를 통과시키지 않는다. 데이터 파일 자체는 Git에 넣지 않고 source ID와 로컬 상대 경로를 둔다.

## 공통 JSONL field

```json
{"dataset":"DocVQA","dataset_revision":"<commit>","source_split":"validation","split":"discovery","sample_id":"...","image":"data/.../....png","question_ids":["..."],"task_types":["OCR"],"selection_seed":42,"exclusion_reason":null}
```

필수 공통 field:

- `dataset`
- `dataset_revision`
- `split`
- `sample_id`
- `selection_seed`

stage별로 `image`, `question_ids`, `pair_labels`, `episode_id`, `task_types`, annotation field를 추가한다.
경로는 가능하면 repository root 기준 상대 경로를 사용한다.

## 필요한 manifest 9종

- `m0_sanity.jsonl`
- `m1_canonical.jsonl`
- `m2a_full.jsonl`
- `m2a_diagnostic.jsonl`
- `m3_pairs.jsonl`
- `m4_pctd_discovery.jsonl`
- `m4_pctd_confirmation.jsonl`
- `m5_trajectories.jsonl`
- `m7_confirmation.jsonl`

현재는 모두 미생성 상태다. 앞 단계 gate 순서대로 실제 manifest를 만든다. PLANNED 단계용 빈
placeholder는 만들지 않는다.

## split·통계 규칙

- 같은 image/episode가 discovery와 confirmation에 동시에 들어가면 안 된다.
- manifest 생성 seed, source revision, scan 범위, 제외 사유를 `.meta.json`에 기록한다.
- 같은 이미지의 여러 질문은 하나의 sample record에 묶는다.
- 표본 수는 config의 sampling 계약과 일치해야 한다.

## DocVQA 생성기

`prep_docvqa.py`는 legacy `data/d4_mini/meta.jsonl` 생성기가 아니라 이 규약의 manifest 생성기로
사용한다.

```bash
python -m vlm_diagnosis.scripts.prep_docvqa \
  --dataset-revision <commit> \
  --manifest experiments/manifests/m2a_diagnostic.jsonl \
  --image-dir data/docvqa_manifest \
  --seed 42 \
  --n-docs 32 \
  --k-min 4
```

dataset revision 없이 실행하지 않는다.
