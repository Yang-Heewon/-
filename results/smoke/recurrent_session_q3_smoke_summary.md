# Recurrent session summary

- Input: `results/smoke/recurrent_session_q3_smoke.jsonl`
- Run/model: `session-20260904T233158Z` / `qwen3vl`
- Datasets: ScreenQA
- Samples/questions/rows: 1 / 3 / 9
- Initial shared image-prefill warm time: total 1555.4 ms; mean 1555.4 ms over 1 samples

> **n=1 is a smoke/validation run, not efficacy evidence.**

> Each condition follows its own generated-answer history. FULL is an independent own-history baseline, not a matched-history causal ablation.
> The CPU cold reservoir retains every K/V uncompressed; only GPU hot history is budgeted. This is not total-storage compression.
> The recurrent gate is a training-free heuristic, not a learned LSTM.
> Same-image paired bootstrap is intentionally omitted in this minimal analyzer. All values are descriptive; no statistical-significance claim is made.

## Overall task metrics

| Condition | n | EM | ANLS | Loyalty to FULL | Full-correct EM retention |
|---|---:|---:|---:|---:|---:|
| full | 3 | 1.000 | 1.000 | 1.000 | 1.000 (3/3) |
| image_static | 3 | 1.000 | 1.000 | 1.000 | 1.000 (3/3) |
| recurrent | 3 | 1.000 | 1.000 | 1.000 | 1.000 (3/3) |

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

## Recurrent working-set composition (means)

| Turn | n | selected image | selected history text | selected prefix control | kept count | entered mean/total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 383.00 | 22.00 | 4.00 | 409.00 | 22.00/22 |
| 2 | 1 | 354.00 | 51.00 | 4.00 | 409.00 | 29.00/29 |
| 3 | 1 | 329.00 | 76.00 | 4.00 | 409.00 | 25.00/25 |

## Selection, memory, and timing by turn (means)

| Turn | Condition | entered mean/total | history text after | image/history weight | CPU cold MiB | GPU hot MiB | active-KV peak MiB | process-GPU peak MiB | state KiB | combined MiB | load / TTFT / turn ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | full | 22.00/22 | 22.00 | —/— | 290.67 | 287.58 | 290.67 | 17319.95 | 22.20 | 581.37 | 58.5/151.9/365.7 |
| 1 | image_static | 0.00/0 | 0.00 | 1.000/0.000 | 290.67 | 57.52 | 60.61 | 16859.05 | 22.20 | 351.30 | 39.8/128.1/334.1 |
| 1 | recurrent | 22.00/22 | 22.00 | 0.675/0.325 | 290.67 | 57.52 | 60.61 | 16859.05 | 22.20 | 351.30 | 22.8/120.1/352.0 |
| 2 | full | 29.00/29 | 51.00 | —/— | 294.75 | 290.67 | 294.75 | 17325.37 | 22.52 | 589.52 | 58.4/144.6/511.9 |
| 2 | image_static | 0.00/0 | 0.00 | 1.000/0.000 | 294.75 | 57.52 | 61.59 | 16859.05 | 22.52 | 356.37 | 22.6/111.1/468.7 |
| 2 | recurrent | 29.00/29 | 51.00 | 0.567/0.433 | 294.75 | 57.52 | 61.59 | 16859.05 | 22.52 | 356.37 | 22.5/119.0/526.0 |
| 3 | full | 25.00/25 | 76.00 | —/— | 298.27 | 294.75 | 298.27 | 17333.52 | 22.78 | 596.55 | 58.4/144.8/427.6 |
| 3 | image_static | 0.00/0 | 0.00 | 1.000/0.000 | 298.27 | 57.52 | 61.03 | 16859.05 | 22.78 | 359.32 | 22.9/111.2/393.1 |
| 3 | recurrent | 25.00/25 | 76.00 | 0.512/0.488 | 298.27 | 57.52 | 61.03 | 16859.05 | 22.78 | 359.32 | 22.6/119.6/436.3 |
