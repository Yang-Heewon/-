# Recurrent session summary

- Input: `results/smoke/recurrent_session_q25_checked.jsonl`
- Run/model: `session-20260904T233408Z` / `qwen25vl`
- Datasets: ScreenQA
- Samples/questions/rows: 1 / 4 / 12
- Initial shared image-prefill warm time: total 1644.1 ms; mean 1644.1 ms over 1 samples

> **n=1 is a smoke/validation run, not efficacy evidence.**

> Each condition follows its own generated-answer history. FULL is an independent own-history baseline, not a matched-history causal ablation.
> The CPU cold reservoir retains every K/V uncompressed; only GPU hot history is budgeted. This is not total-storage compression.
> The recurrent gate is a training-free heuristic, not a learned LSTM.
> Same-image paired bootstrap is intentionally omitted in this minimal analyzer. All values are descriptive; no statistical-significance claim is made.

## Overall task metrics

| Condition | n | EM | ANLS | Loyalty to FULL | Full-correct EM retention |
|---|---:|---:|---:|---:|---:|
| full | 4 | 1.000 | 1.000 | 1.000 | 1.000 (4/4) |
| image_static | 4 | 1.000 | 1.000 | 1.000 | 1.000 (4/4) |
| recurrent | 4 | 1.000 | 1.000 | 1.000 | 1.000 (4/4) |

## Metrics by turn

| Turn | Condition | n | EM | ANLS | Full-correct EM retention |
|---:|---|---:|---:|---:|---:|
| 1 | full | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 1 | image_static | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 1 | recurrent | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 2 | full | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 2 | image_static | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 2 | recurrent | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 3 | full | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 3 | image_static | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 3 | recurrent | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 4 | full | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 4 | image_static | 1 | 1.000 | 1.000 | 1.000 (1/1) |
| 4 | recurrent | 1 | 1.000 | 1.000 | 1.000 (1/1) |

## Recurrent working-set composition (means)

| Turn | n | selected image | selected history text | selected prefix control | kept count | entered mean/total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 231.00 | 13.00 | 4.00 | 248.00 | 13.00/13 |
| 2 | 1 | 205.00 | 39.00 | 4.00 | 248.00 | 26.00/26 |
| 3 | 1 | 178.00 | 66.00 | 4.00 | 248.00 | 28.00/28 |
| 4 | 1 | 155.00 | 89.00 | 4.00 | 248.00 | 28.00/28 |

## Selection, memory, and timing by turn (means)

| Turn | Condition | entered mean/total | history text after | image/history weight | CPU cold MiB | GPU hot MiB | active-KV peak MiB | process-GPU peak MiB | state KiB | combined MiB | load / TTFT / turn ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | full | 22.00/22 | 22.00 | —/— | 68.91 | 67.70 | 68.91 | 16028.07 | 13.54 | 137.83 | 22.4/87.0/293.3 |
| 1 | image_static | 0.00/0 | 0.00 | 1.000/0.000 | 68.91 | 13.56 | 14.77 | 15909.95 | 13.54 | 83.69 | 11.2/75.0/209.9 |
| 1 | recurrent | 13.00/13 | 13.00 | 0.675/0.325 | 68.91 | 13.56 | 14.77 | 15909.95 | 13.54 | 83.69 | 9.7/81.5/237.9 |
| 2 | full | 29.00/29 | 51.00 | —/— | 70.49 | 68.91 | 70.49 | 16028.80 | 13.85 | 141.00 | 22.0/85.9/341.2 |
| 2 | image_static | 0.00/0 | 0.00 | 1.000/0.000 | 70.49 | 13.56 | 15.15 | 15913.90 | 13.85 | 85.65 | 9.9/73.7/326.0 |
| 2 | recurrent | 26.00/26 | 39.00 | 0.567/0.433 | 70.49 | 13.56 | 15.15 | 15913.90 | 13.85 | 85.65 | 9.9/80.6/370.4 |
| 3 | full | 24.00/24 | 75.00 | —/— | 71.80 | 70.49 | 71.80 | 16030.48 | 14.10 | 143.62 | 22.4/87.1/221.5 |
| 3 | image_static | 0.00/0 | 0.00 | 1.000/0.000 | 71.80 | 13.56 | 14.88 | 15910.95 | 14.10 | 86.69 | 9.8/73.1/207.2 |
| 3 | recurrent | 28.00/28 | 66.00 | 0.512/0.488 | 71.80 | 13.56 | 14.88 | 15910.95 | 14.10 | 86.69 | 10.0/81.4/238.9 |
| 4 | full | 26.00/26 | 101.00 | —/— | 73.23 | 71.80 | 73.23 | 16062.33 | 14.38 | 146.47 | 22.6/86.4/283.4 |
| 4 | image_static | 0.00/0 | 0.00 | 1.000/0.000 | 73.23 | 13.56 | 14.98 | 15911.87 | 14.38 | 88.22 | 9.9/74.6/273.2 |
| 4 | recurrent | 28.00/28 | 89.00 | 0.480/0.520 | 73.23 | 13.56 | 14.98 | 15911.87 | 14.38 | 88.22 | 10.0/82.3/309.3 |
