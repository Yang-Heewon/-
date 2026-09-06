# Context-only KV — full — results/context_only/smoke_full.jsonl

model qwen25vl · code 429ee2dc (dirty) · split dev · protect: prefix first 4 + tokenizer special ids in prefix · NLL: answers[0] body tokens, EOS excluded, teacher forced

## 단계 1 parity (keep=100% ragged 경로 vs 일반 FULL forward)

| context | 질문 | 위치 일치 | logits 최대 오차 | 평균 오차 | argmax 일치율 | 첫 답 token 일치 | NLL cached | NLL dense | NLL 차 |
|---|---|---|---|---|---|---|---|---|---|
| 14107 | sqa_val_01608 | True | 0.3281 | 0.01531 | 1.000 | True | 0.0107 | 0.0105 | 0.00021 |
| 14107 | sqa_val_01607 | True | 0.1094 | 0.01271 | 1.000 | True | 0.4293 | 0.4383 | 0.00904 |
| 2178 | sqa_val_00272 | True | 0.2969 | 0.01277 | 1.000 | True | 0.0412 | 0.0412 | 0.00002 |
| 2178 | sqa_val_00270 | True | 0.2812 | 0.01665 | 1.000 | True | 0.0004 | 0.0004 | 0.00001 |

## 저장량·비용 (build 기록 평균)

| 조건 | 쌍 유지 비율 | KV bytes | metadata bytes | prefill s | score s | select s | prune s | peak GB |
|---|---|---|---|---|---|---|---|---|
| full|none|none|k1|s0 | 1.000 | 67.7 MB | 1093 KB | 1.01 | 0.000 | 0.000 | 0.034 | 15.65 |
