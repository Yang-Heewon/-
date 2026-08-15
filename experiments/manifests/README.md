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

위 목록은 production manifest 계약이다. discovery 보조 산출물은 stage 이름을 명시한 별도
파일로 둘 수 있으며, 현재 T4 ScreenQA 파일럿은 아래처럼 `pilot`/`draft` 상태로 관리한다.

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

## ScreenQA T4 파일럿 v2

최종 위치 질문은 `t4_pilot.jsonl`, 내용→위치 질문 쌍 초안은
`t4_pairs_draft.jsonl`, 전수 시각 판정은 `t4_visual_audit.jsonl`에 둔다. 시각 감사 파일은
생성기의 입력이며, 감사에서 실패한 문항은 교체하지 않고 post-generation 단계에서 제거한다.

기본 실행은 입력·해시·감사 조인만 확인하고 파일을 쓰지 않는다.

```bash
python vlm_diagnosis/scripts/gen_t4_pilot.py --dry-run
```

검증된 산출물을 명시적으로 갱신할 때만 다음 순서로 실행한다.

```bash
python vlm_diagnosis/scripts/gen_t4_pilot.py --write-in-place
python vlm_diagnosis/scripts/build_t4_pair_reviews.py
python vlm_diagnosis/scripts/validate_t4_pilot.py
```

`t4_pairs_review_A.xlsx`와 `t4_pairs_review_B.xlsx`는 draft label과 상대 검수자 정보를 제외한
독립·블라인드 검수 파일이다. T2↔T4 우선순위가 확정되기 전에는 same-evidence type-crossing
쌍의 `final_label`을 비워 두고 `PRECEDENCE_PENDING`으로 표시한다.
