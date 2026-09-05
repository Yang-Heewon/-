# Recurrent session summary

- Input: `results/smoke/recurrent_session_q3_adapter_refactor.jsonl`
- Run/model: `session-20260905T042733Z` / `qwen3vl`
- Schema/storage: `1.1` / `delete`
- Datasets: ScreenQA
- Samples/questions/rows: 1 / 3 / 9
- Initial shared image-prefill warm time: total 1582.4 ms; mean 1582.4 ms over 1 samples
- Initial cache setup mean: full=60.8 ms, image_static=53.4 ms, recurrent=50.4 ms

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
| 1 | full | 2067.00 | 2067.00/290.67 | 2040.00 | 0.00 | 290.67 | 0.00 | 0.00/0.00 | 290.74 |
| 1 | image_static | 2067.00 | 409.00/57.52 | 405.00 | 0.00 | 57.52 | 1636.00 | 22.00/0.00 | 57.54 |
| 1 | recurrent | 2067.00 | 409.00/57.52 | 383.00 | 0.00 | 57.52 | 1636.00 | 22.00/22.00 | 57.54 |
| 2 | full | 2096.00 | 2096.00/294.75 | 2040.00 | 0.00 | 294.75 | 0.00 | 0.00/0.00 | 294.82 |
| 2 | image_static | 2096.00 | 409.00/57.52 | 405.00 | 0.00 | 57.52 | 1636.00 | 29.00/0.00 | 57.54 |
| 2 | recurrent | 2096.00 | 409.00/57.52 | 354.00 | 0.00 | 57.52 | 1636.00 | 29.00/29.00 | 57.54 |
| 3 | full | 2121.00 | 2121.00/298.27 | 2040.00 | 0.00 | 298.27 | 0.00 | 0.00/0.00 | 298.33 |
| 3 | image_static | 2121.00 | 409.00/57.52 | 405.00 | 0.00 | 57.52 | 1636.00 | 25.00/0.00 | 57.54 |
| 3 | recurrent | 2121.00 | 409.00/57.52 | 329.00 | 0.00 | 57.52 | 1636.00 | 25.00/25.00 | 57.54 |

## Transient memory and timing by turn (means)

| Turn | Condition | hot input MiB | active-KV peak MiB | compaction upper MiB | H2D/D2H MiB | process-GPU peak MiB* | selector/session metadata KiB | combined upper MiB | load / TTFT / turn ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | full | 287.58 | 290.67 | 290.67 | 0.00/0.00 | 17192.59 | 36.33/32.27 | 290.71 | 0.0/94.7/288.6 |
| 1 | image_static | 57.52 | 60.61 | 175.64 | 0.00/0.00 | 17267.85 | 7.19/19.31 | 175.65 | 0.0/87.9/286.8 |
| 1 | recurrent | 57.52 | 60.61 | 175.64 | 0.00/0.00 | 17267.85 | 7.19/19.31 | 175.65 | 0.0/97.4/309.4 |
| 2 | full | 290.67 | 294.75 | 294.75 | 0.00/0.00 | 17199.61 | 36.84/32.49 | 294.79 | 0.0/88.2/429.1 |
| 2 | image_static | 57.52 | 61.59 | 176.62 | 0.00/0.00 | 17272.91 | 7.19/19.31 | 176.63 | 0.0/87.7/436.7 |
| 2 | recurrent | 57.52 | 61.59 | 176.62 | 0.00/0.00 | 17272.91 | 7.19/19.31 | 176.63 | 0.0/97.1/495.2 |
| 3 | full | 294.75 | 298.27 | 298.27 | 0.00/0.00 | 17203.25 | 37.28/32.69 | 298.30 | 0.0/87.8/346.5 |
| 3 | image_static | 57.52 | 61.03 | 176.06 | 0.00/0.00 | 17276.63 | 7.19/19.31 | 176.07 | 0.0/87.2/354.5 |
| 3 | recurrent | 57.52 | 61.03 | 176.06 | 0.00/0.00 | 17276.63 | 7.19/19.31 | 176.07 | 0.0/96.9/401.4 |

\* Process-GPU peak has process scope, not per-method scope.
