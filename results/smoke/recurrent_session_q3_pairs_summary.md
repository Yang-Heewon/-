# Physical global KV-pair session validation

Model: qwen3vl; images: 1; paired turns: 3.
Unit: one (layer, KV head, logical token) K/V vector pair. Head quotas are not fixed.

| Condition | EM | ANLS | FULL-correct retention | Answer agreement | Mean stored KV MiB |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 1.000 | 294.562 |
| image_static | 1.000 | 1.000 | 1.000 | 1.000 | 57.516 |
| recurrent | 1.000 | 1.000 | 1.000 | 1.000 | 57.516 |

## Recurrent allocation by turn

| Image | Turn | Global B | Head min–max | Distinct token IDs | Image pairs | Text pairs | Deleted this turn | Persistent tensors MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ScreenQA:14107 | 1 | 117,792 | 29–1345 | 1,951 | 110,578 | 6,062 | 6,336 | 62.249 |
| ScreenQA:14107 | 2 | 117,792 | 54–1308 | 1,946 | 102,583 | 14,057 | 8,352 | 62.249 |
| ScreenQA:14107 | 3 | 117,792 | 75–1251 | 1,943 | 95,708 | 20,932 | 7,200 | 62.249 |

## Scope and limits

- Exact global budgets, head/modality totals, physical-byte accounting, zero cold KV and turn continuity pass validation.
- Logs contain counts, not every retained pair ID. Irreversible deletion and non-shared storage are checked by cache/session tests, not inferred from these counts.
- The retained fraction is relative to the initial full prefix, not the growing FULL cache. Current-turn KV and compaction copies require extra temporary storage; the initial prefill is full.
- Persistent tensors include KV, 34-byte per-pair selector state, 8-byte per-pair cache IDs and template tensors; not model weights, activations, Python objects or allocator reserve.
- GPU allocation peaks include the shared model and other condition caches. They are not isolated per-method memory measurements.
- All conditions use the same Python ragged eager reference backend. This is not a fused-kernel throughput claim.
- Each condition generates its own answer history. This is not a matched-history selector ablation or a KVZIP comparison.
- Small smoke results establish execution only, not general accuracy retention, statistical significance or novelty.
