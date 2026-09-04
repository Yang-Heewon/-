# M0 measurement report — m0-20260813T150604Z

- source: `1b25b75670facb42a072f910397aea3a881e6c1a`
- model: `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- samples: 40

## image_base
- grounding: mean score 0.60 (n=10)
- icon: mean score 1.00 (n=10)
- layout: mean score 1.00 (n=10)
- ocr: mean score 1.00 (n=10)
- masked-generate vs HF generate 일치: 33/40
- finite: 40/40

## mask_2d_4d
- max_abs_logit_diff (worst): 0
- prediction_equal: 40/40
- finite: 40/40

## keep100
- max_abs_logit_diff (worst): 0
- prediction_equal: 40/40
- mask_exactly_equal: 40/40
- finite: 40/40

## cache_identity_strict
- max_abs_logit_diff (worst): 0.155
- prediction_equal: 40/40
- serialize roundtrip bitexact: 40/40
- max prefix KV diff: 0.978
- finite: 40/40

## cache_equivalence_operational
- split_after_image: max diff worst 0.141, median 0.0593, prediction_equal 40/40
- split_late: max diff worst 0.133, median 0.0469, prediction_equal 40/40
- finite: 40/40

## full_mask
- task score drop mean: 0.78
- delta_logp mean: -11.38
- finite: 40/40

## v1_v2
- V1 mean task score: 0.50 (smuggling 경로)
- V2 mean task score: 0.12 (기본 semantics)
- finite: 40/40

## thresholds
- strict_cache_max_abs: null (측정 후 고정 필요)
- strict_logit_max_abs: null (측정 후 고정 필요)
- operational_logit_max_abs: null (측정 후 고정 필요)
- task_equivalence_percentage_point: null (측정 후 고정 필요)
