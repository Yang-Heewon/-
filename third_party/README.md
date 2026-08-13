# Pinned baseline sources

These repositories are reference implementations, pinned as git checkouts.
The VLM adapters in `vlm_diagnosis/core/kv_baselines.py` document whenever an
algorithm is adapted rather than executed through the upstream runtime.

| Directory | Commit | License | Role |
|---|---|---|---|
| `kvpress` | `4e41f149018daf32644eca1e5b22b736da426586` | Apache-2.0 | Sparse presses, merge-on-evict, composed/quantized cache reference |
| `KIVI` | `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` | MIT | Asymmetric K-per-channel/V-per-token KV quantization reference |
| `HERMES` | `8d699b16a6bedb9086c1b39ec4253c6a1d1ce789` | MIT | Qwen2.5-VL-native hierarchical/streaming KV baseline for M5 |

Do not report upstream names without the implementation-status label:

- `upstream-runtime`: unmodified official execution path;
- `vlm-adaptation`: algorithm adapted to visual-token-only Qwen2.5-VL;
- `quality-simulation`: fake quantization or logical mask, no physical memory claim.
