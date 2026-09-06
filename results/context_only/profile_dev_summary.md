# Context-only KV — profile — results/context_only/profile_*_qwen25vl_dev.jsonl

model qwen25vl · code 429ee2dc (dirty) · split dev · protect: prefix first 4 + tokenizer special ids in prefix (image/video placeholders excluded) · NLL: answers[0] body tokens, EOS excluded, teacher forced

## 품질 (context 0, 질문 0, FULL EM —; FULL 참조 없는 질문 60개 제외)

### EM (행 = 방법, 열 = 유지율)

| 방법 |  |
|---|

### FULL-correct 보존 (행 = 방법, 열 = 유지율)

| 방법 |  |
|---|

### ΔNLL vs FULL (행 = 방법, 열 = 유지율)

| 방법 |  |
|---|

## 저장량·비용 (build 기록 평균)

| 조건 | 쌍 유지 비율 | KV bytes | metadata bytes | prefill s | score s | select s | prune s | peak GB |
|---|---|---|---|---|---|---|---|---|
| d|keep_high|global|k0.2|s0 | 0.200 | 13.5 MB | 227 KB | 0.99 | 0.000 | 0.033 | 0.031 | 15.60 |
| k_norm|keep_high|global|k0.2|s0 | 0.200 | 13.5 MB | 227 KB | 0.96 | 0.004 | 0.034 | 0.028 | 15.60 |
| mlp_norm|keep_high|global|k0.2|s0 | 0.200 | 13.5 MB | 227 KB | 1.00 | 0.000 | 0.032 | 0.024 | 15.60 |
| plain|keep_high|global|k0.2|s0 | 1.000 | 67.7 MB | 1093 KB | 0.96 | 0.000 | 0.000 | 0.060 | 15.65 |
| r|keep_high|global|k0.2|s0 | 0.200 | 13.5 MB | 227 KB | 0.99 | 0.000 | 0.032 | 0.033 | 15.60 |
