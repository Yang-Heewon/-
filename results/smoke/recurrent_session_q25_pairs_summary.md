# Physical global KV-pair session validation

Model: qwen25vl; images: 1; paired turns: 4.
Unit: one (layer, KV head, logical token) K/V vector pair. Head quotas are not fixed.

| Condition | EM | ANLS | FULL-correct retention | Answer agreement | Mean stored KV MiB |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 1.000 | 71.107 |
| image_static | 1.000 | 1.000 | 1.000 | 1.000 | 13.541 |
| recurrent | 1.000 | 1.000 | 1.000 | 1.000 | 13.541 |

## Recurrent allocation by turn

| Image | Turn | Global B | Head min–max | Distinct token IDs | Image pairs | Text pairs | Deleted this turn | Persistent tensors MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ScreenQA:14107 | 1 | 27,731 | 28–559 | 1,000 | 25,550 | 1,733 | 2,464 | 14.661 |
| ScreenQA:14107 | 2 | 27,731 | 36–541 | 1,002 | 23,220 | 4,063 | 3,248 | 14.661 |
| ScreenQA:14107 | 3 | 27,731 | 51–538 | 986 | 21,041 | 6,242 | 2,688 | 14.661 |
| ScreenQA:14107 | 4 | 27,731 | 64–519 | 975 | 19,015 | 8,268 | 2,912 | 14.661 |

## Scope and limits

- Exact global budgets, head/modality totals, physical-byte accounting, zero cold KV and turn continuity pass validation.
- Logs contain counts, not every retained pair ID. Irreversible deletion and non-shared storage are checked by cache/session tests, not inferred from these counts.
- The retained fraction is relative to the initial full prefix, not the growing FULL cache. Current-turn KV and compaction copies require extra temporary storage; the initial prefill is full.
- Persistent tensors include KV, 34-byte per-pair selector state, 8-byte per-pair cache IDs and template tensors; not model weights, activations, Python objects or allocator reserve.
- GPU allocation peaks include the shared model and other condition caches. They are not isolated per-method memory measurements.
- All conditions use the same Python ragged eager reference backend. This is not a fused-kernel throughput claim.
- Each condition generates its own answer history. This is not a matched-history selector ablation or a KVZIP comparison.
- Small smoke results establish execution only, not general accuracy retention, statistical significance or novelty.
