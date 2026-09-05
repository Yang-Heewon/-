# Recurrent session summary

- Input: `results/smoke/recurrent_session_q3_delete_checked.jsonl`
- Run/model: `session-20260905T000429Z` / `qwen3vl`
- Schema/storage: `1.1` / `delete`
- Datasets: ScreenQA
- Samples/questions/rows: 1 / 3 / 9
- Initial shared image-prefill warm time: total 1561.7 ms; mean 1561.7 ms over 1 samples
- Initial cache setup mean: full=58.0 ms, image_static=47.6 ms, recurrent=32.8 ms

> **n=1 is a smoke/validation run, not efficacy evidence.**

> Each condition follows its own generated-answer history. FULL is an independent own-history baseline, not a matched-history causal ablation.
> Delete mode physically retains only selected K/V for compressed conditions; evicted K/V and per-token state are irreversible. FULL retains its complete own history, and no CPU cold reservoir is kept.
> Process-GPU peak is an absolute process measurement: it includes the shared model and other conditions' resident caches, so it is not method-only memory.
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

| Turn | n | selected image | selected history text | selected prefix control | kept count | image/history weight | entered mean/total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 383.00 | 22.00 | 4.00 | 409.00 | 0.675/0.325 | 22.00/22 |
| 2 | 1 | 354.00 | 51.00 | 4.00 | 409.00 | 0.567/0.433 | 29.00/29 |
| 3 | 1 | 329.00 | 76.00 | 4.00 | 409.00 | 0.512/0.488 | 25.00/25 |

## Persistent storage and deletion by turn (means)

| Turn | Condition | logical tokens | retained KV tokens/MiB | image remaining | CPU cold MiB | resident GPU KV MiB | initial deleted | deleted this turn/image | persistent tensors MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | full | 2067.00 | 2067.00/290.67 | 2040.00 | 0.00 | 290.67 | 0.00 | 0.00/0.00 | 290.73 |
| 1 | image_static | 2067.00 | 409.00/57.52 | 405.00 | 0.00 | 57.52 | 1636.00 | 22.00/0.00 | 57.54 |
| 1 | recurrent | 2067.00 | 409.00/57.52 | 383.00 | 0.00 | 57.52 | 1636.00 | 22.00/22.00 | 57.54 |
| 2 | full | 2096.00 | 2096.00/294.75 | 2040.00 | 0.00 | 294.75 | 0.00 | 0.00/0.00 | 294.81 |
| 2 | image_static | 2096.00 | 409.00/57.52 | 405.00 | 0.00 | 57.52 | 1636.00 | 29.00/0.00 | 57.54 |
| 2 | recurrent | 2096.00 | 409.00/57.52 | 354.00 | 0.00 | 57.52 | 1636.00 | 29.00/29.00 | 57.54 |
| 3 | full | 2121.00 | 2121.00/298.27 | 2040.00 | 0.00 | 298.27 | 0.00 | 0.00/0.00 | 298.32 |
| 3 | image_static | 2121.00 | 409.00/57.52 | 405.00 | 0.00 | 57.52 | 1636.00 | 25.00/0.00 | 57.54 |
| 3 | recurrent | 2121.00 | 409.00/57.52 | 329.00 | 0.00 | 57.52 | 1636.00 | 25.00/25.00 | 57.54 |

## Transient memory and timing by turn (means)

| Turn | Condition | hot input MiB | active-KV peak MiB | compaction upper MiB | H2D/D2H MiB | process-GPU peak MiB* | selector/session metadata KiB | combined upper MiB | load / TTFT / turn ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | full | 287.58 | 290.67 | 290.67 | 0.00/0.00 | 17192.59 | 22.20/34.26 | 290.69 | 0.0/92.4/283.8 |
| 1 | image_static | 57.52 | 60.61 | 175.64 | 0.00/0.00 | 17267.85 | 4.39/21.31 | 175.64 | 0.0/87.9/281.6 |
| 1 | recurrent | 57.52 | 60.61 | 175.64 | 0.00/0.00 | 17267.85 | 4.39/21.31 | 175.64 | 0.0/97.8/305.8 |
| 2 | full | 290.67 | 294.75 | 294.75 | 0.00/0.00 | 17199.61 | 22.52/34.49 | 294.77 | 0.0/87.4/420.2 |
| 2 | image_static | 57.52 | 61.59 | 176.62 | 0.00/0.00 | 17272.91 | 4.39/21.31 | 176.63 | 0.0/86.1/428.1 |
| 2 | recurrent | 57.52 | 61.59 | 176.62 | 0.00/0.00 | 17272.91 | 4.39/21.31 | 176.63 | 0.0/95.7/482.8 |
| 3 | full | 294.75 | 298.27 | 298.27 | 0.00/0.00 | 17203.25 | 22.78/34.68 | 298.29 | 0.0/85.6/338.5 |
| 3 | image_static | 57.52 | 61.03 | 176.06 | 0.00/0.00 | 17276.63 | 4.39/21.31 | 176.07 | 0.0/86.5/348.3 |
| 3 | recurrent | 57.52 | 61.03 | 176.06 | 0.00/0.00 | 17276.63 | 4.39/21.31 | 176.07 | 0.0/95.1/390.6 |

\* Process-GPU peak has process scope, not per-method scope.
