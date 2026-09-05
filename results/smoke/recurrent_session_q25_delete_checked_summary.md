# Recurrent session summary

- Input: `results/smoke/recurrent_session_q25_delete_checked.jsonl`
- Run/model: `session-20260905T000322Z` / `qwen25vl`
- Schema/storage: `1.1` / `delete`
- Datasets: ScreenQA
- Samples/questions/rows: 1 / 4 / 12
- Initial shared image-prefill warm time: total 1608.2 ms; mean 1608.2 ms over 1 samples
- Initial cache setup mean: full=22.0 ms, image_static=12.5 ms, recurrent=12.1 ms

> **n=1 is a smoke/validation run, not efficacy evidence.**

> Each condition follows its own generated-answer history. FULL is an independent own-history baseline, not a matched-history causal ablation.
> Delete mode physically retains only selected K/V for compressed conditions; evicted K/V and per-token state are irreversible. FULL retains its complete own history, and no CPU cold reservoir is kept.
> Process-GPU peak is an absolute process measurement: it includes the shared model and other conditions' resident caches, so it is not method-only memory.
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

| Turn | n | selected image | selected history text | selected prefix control | kept count | image/history weight | entered mean/total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 231.00 | 13.00 | 4.00 | 248.00 | 0.675/0.325 | 13.00/13 |
| 2 | 1 | 209.00 | 35.00 | 4.00 | 248.00 | 0.567/0.433 | 22.00/22 |
| 3 | 1 | 186.00 | 58.00 | 4.00 | 248.00 | 0.512/0.488 | 23.00/23 |
| 4 | 1 | 162.00 | 82.00 | 4.00 | 248.00 | 0.480/0.520 | 25.00/25 |

## Persistent storage and deletion by turn (means)

| Turn | Condition | logical tokens | retained KV tokens/MiB | image remaining | CPU cold MiB | resident GPU KV MiB | initial deleted | deleted this turn/image | persistent tensors MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | full | 1260.00 | 1260.00/68.91 | 1222.00 | 0.00 | 68.91 | 0.00 | 0.00/0.00 | 68.94 |
| 1 | image_static | 1260.00 | 248.00/13.56 | 244.00 | 0.00 | 13.56 | 990.00 | 22.00/0.00 | 13.58 |
| 1 | recurrent | 1260.00 | 248.00/13.56 | 231.00 | 0.00 | 13.56 | 990.00 | 22.00/13.00 | 13.58 |
| 2 | full | 1289.00 | 1289.00/70.49 | 1222.00 | 0.00 | 70.49 | 0.00 | 0.00/0.00 | 70.53 |
| 2 | image_static | 1289.00 | 248.00/13.56 | 244.00 | 0.00 | 13.56 | 990.00 | 29.00/0.00 | 13.58 |
| 2 | recurrent | 1289.00 | 248.00/13.56 | 209.00 | 0.00 | 13.56 | 990.00 | 29.00/22.00 | 13.58 |
| 3 | full | 1313.00 | 1313.00/71.80 | 1222.00 | 0.00 | 71.80 | 0.00 | 0.00/0.00 | 71.84 |
| 3 | image_static | 1313.00 | 248.00/13.56 | 244.00 | 0.00 | 13.56 | 990.00 | 24.00/0.00 | 13.58 |
| 3 | recurrent | 1313.00 | 248.00/13.56 | 186.00 | 0.00 | 13.56 | 990.00 | 24.00/23.00 | 13.58 |
| 4 | full | 1339.00 | 1339.00/73.23 | 1222.00 | 0.00 | 73.23 | 0.00 | 0.00/0.00 | 73.26 |
| 4 | image_static | 1339.00 | 248.00/13.56 | 244.00 | 0.00 | 13.56 | 990.00 | 26.00/0.00 | 13.58 |
| 4 | recurrent | 1339.00 | 248.00/13.56 | 162.00 | 0.00 | 13.56 | 990.00 | 26.00/24.00 | 13.58 |

## Transient memory and timing by turn (means)

| Turn | Condition | hot input MiB | active-KV peak MiB | compaction upper MiB | H2D/D2H MiB | process-GPU peak MiB* | selector/session metadata KiB | combined upper MiB | load / TTFT / turn ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | full | 67.70 | 68.91 | 68.91 | 0.00/0.00 | 16011.54 | 13.54/20.95 | 68.92 | 0.0/63.0/202.3 |
| 1 | image_static | 13.56 | 14.77 | 41.89 | 0.00/0.00 | 16013.85 | 2.66/13.04 | 41.89 | 0.0/62.5/195.8 |
| 1 | recurrent | 13.56 | 14.77 | 41.89 | 0.00/0.00 | 16013.85 | 2.66/13.04 | 41.89 | 0.0/69.1/217.1 |
| 2 | full | 68.91 | 70.49 | 70.49 | 0.00/0.00 | 16011.96 | 13.85/21.18 | 70.51 | 0.0/61.7/297.4 |
| 2 | image_static | 13.56 | 15.15 | 42.27 | 0.00/0.00 | 16014.26 | 2.66/13.04 | 42.28 | 0.0/60.0/303.5 |
| 2 | recurrent | 13.56 | 15.15 | 42.27 | 0.00/0.00 | 16014.26 | 2.66/13.04 | 42.28 | 0.0/67.8/344.6 |
| 3 | full | 70.49 | 71.80 | 71.80 | 0.00/0.00 | 16012.11 | 14.10/21.37 | 71.82 | 0.0/60.9/181.5 |
| 3 | image_static | 13.56 | 14.88 | 42.00 | 0.00/0.00 | 16015.00 | 2.66/13.04 | 42.00 | 0.0/61.3/187.8 |
| 3 | recurrent | 13.56 | 14.88 | 42.00 | 0.00/0.00 | 16015.00 | 2.66/13.04 | 42.00 | 0.0/68.2/215.6 |
| 4 | full | 71.80 | 73.23 | 73.23 | 0.00/0.00 | 16013.52 | 14.38/21.57 | 73.24 | 0.0/61.9/240.9 |
| 4 | image_static | 13.56 | 14.98 | 42.11 | 0.00/0.00 | 16015.11 | 2.66/13.04 | 42.11 | 0.0/60.9/248.2 |
| 4 | recurrent | 13.56 | 14.98 | 42.11 | 0.00/0.00 | 16015.11 | 2.66/13.04 | 42.11 | 0.0/69.2/282.1 |

\* Process-GPU peak has process scope, not per-method scope.
