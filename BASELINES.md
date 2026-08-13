# Visual KV compression baselines

## What is implemented

The comparison is split by compression mechanism. It never uses model-weight
quantization.

| Family | Baseline | Implementation status |
|---|---|---|
| FULL | FP16 visual KV | logical reference |
| SPARSE | Random and S1/SnapKV-style visual token eviction | visual-only quality simulation |
| QUANT | KIVI-style asymmetric K/V 2-bit or 4-bit | visual-only fake-quant quality simulation |
| TRANSFORMED | merge-on-evict | adaptation of NVIDIA kvpress `MergingPress` |
| HYBRID | S1 eviction plus KIVI-style quantization | local composition |
| QUANT physical | Transformers `QuantizedCache`, HQQ backend | Qwen2.5-VL/V100 smoke-tested |
| trajectory | HERMES | pinned official Qwen2.5-VL source; M5 integration target |

`quality simulation` means task/logp degradation can be measured, but the
execution does not physically shrink GPU memory. Only the physical cache path
may be used for latency and memory claims.

## Pinned upstream repositories

See `third_party/README.md`. Current pins:

- NVIDIA kvpress `4e41f149018daf32644eca1e5b22b736da426586` — Apache-2.0
- KIVI `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` — MIT
- HERMES `8d699b16a6bedb9086c1b39ec4253c6a1d1ce789` — MIT

## Byte matching

The storage estimator accounts for:

- packed K and V payload;
- quantization scales and zero-points;
- sparse token indices;
- mRoPE temporal/height/width position metadata.

These values are estimates until M6 writes and measures a real serialized artifact.

The Phase 1 protocol uses four primary target ratios relative to full visual-KV
serialized bytes: **20%, 40%, 60%, and 80% keep**. FULL is the uncompressed
100% reference. SPARSE/QUANT/TRANSFORMED/HYBRID must fit the same target bytes
at each operating point. If 20% is lossless, only the small diagnostic subset adds
5% and 10%; the main grid remains unchanged.

The current runner has not implemented that grid yet. Its target is the native
serialized size of all visual KV values under 4-bit KIVI-style quantization
(about 28% in the one-document smoke). SPARSE and TRANSFORMED keep the largest
number of FP16 tokens that fit those bytes, and HYBRID keeps the largest number
of 8-bit tokens. Treat this as a path smoke only. The required grid runner is
specified in `experiments/05_M2B_FAMILY_STRESS.md`.

Run:

```bash
python -m vlm_diagnosis.exps.m2_family_baselines \
  --limit 1 \
  --reference-bits 4 \
  --hybrid-bits 8 \
  --device cuda:0
```

The current runner reports teacher-forced answer logp and is a diagnostic
smoke path. It must gain dataset task metrics before producing paper results.

## Physical quantized-cache smoke

Install:

```bash
python -m pip install -r requirements-baselines.txt
```

Run the V100-compatible HQQ backend:

```bash
python -m vlm_diagnosis.scripts.smoke_quantized_cache \
  --backend hqq \
  --nbits 4 \
  --device cuda:0
```

The first generated token uses the unquantized prefill values because of the
Transformers cache update contract. The second token exercises dequantization;
therefore the smoke generates two tokens.

Quanto 0.2.7 installs but its CUDA extension uses `cp.async`/Marlin operations
requiring sm_80 or newer. It does not compile on the local V100 (sm_70), so its
physical runtime is an A100+ confirmation item.

## Interpretation guardrails

- Equal token retention is not an equal storage comparison; use serialized bytes.
- Fake quantization is a quality baseline, not a speed/memory measurement.
- The S1 selector is a Qwen2.5-VL adaptation, not an unmodified upstream SnapKV runtime.
  Because it uses the future question, label it as a read-time comparator unless a
  separate write-time construction is implemented. It is not persistent-storage
  compression when full KV must be retained until the question arrives.
- Merge-on-evict currently uses pre-RoPE visual keys in the adapter; label it as an adaptation.
- A single smoke record confirms execution only and is not a research result.
