# Context-only KV — full — results/context_only/full_qwen25vl_dev.jsonl

model qwen25vl · code 429ee2dc (dirty) · split dev · protect: prefix first 4 + tokenizer special ids in prefix (image/video placeholders excluded) · NLL: answers[0] body tokens, EOS excluded, teacher forced

## 단계 1 parity (keep=100% ragged 경로 vs 일반 FULL forward)

| context | 질문 | 위치 일치 | logits 최대 오차 | 평균 오차 | argmax 일치율 | 첫 답 token 일치 | NLL cached | NLL dense | NLL 차 |
|---|---|---|---|---|---|---|---|---|---|
| 14107 | sqa_val_01608 | True | 0.3281 | 0.01531 | 1.000 | True | 0.0107 | 0.0105 | 0.00021 |
| 14107 | sqa_val_01607 | True | 0.1094 | 0.01271 | 1.000 | True | 0.4293 | 0.4383 | 0.00904 |
| 2178 | sqa_val_00272 | True | 0.2969 | 0.01277 | 1.000 | True | 0.0412 | 0.0412 | 0.00002 |
| 2178 | sqa_val_00270 | True | 0.2812 | 0.01665 | 1.000 | True | 0.0004 | 0.0004 | 0.00001 |
| 37948 | sqa_val_04610 | True | 0.1133 | 0.01048 | 0.947 | True | 1.3257 | 1.3259 | 0.00025 |
| 37948 | sqa_val_04611 | True | 0.2344 | 0.01372 | 1.000 | True | 5.4799 | 5.4787 | 0.00114 |
| 3350 | sqa_val_00417 | True | 0.0781 | 0.00993 | 1.000 | True | 0.6929 | 0.6932 | 0.00025 |
| 3350 | sqa_val_00418 | True | 0.1328 | 0.00968 | 1.000 | True | 0.0011 | 0.0011 | 0.00001 |
| 49909 | sqa_val_06093 | True | 0.1074 | 0.01347 | 1.000 | True | 1.5488 | 1.5458 | 0.00294 |
| 49909 | sqa_val_06092 | True | 0.1133 | 0.01086 | 1.000 | True | 2.3292 | 2.3330 | 0.00379 |
| 5151 | sqa_val_00624 | True | 0.1299 | 0.01090 | 1.000 | True | 0.0543 | 0.0531 | 0.00123 |
| 5151 | sqa_val_00628 | True | 0.1875 | 0.01005 | 1.000 | True | 0.0084 | 0.0083 | 0.00013 |
| 60887 | sqa_val_07297 | True | 0.2500 | 0.01076 | 1.000 | True | 8.6994 | 8.7189 | 0.01953 |
| 60887 | sqa_val_07301 | True | 0.2344 | 0.01572 | 1.000 | True | 2.4073 | 2.4078 | 0.00044 |
| 48950 | sqa_val_05970 | True | 0.0957 | 0.01169 | 1.000 | True | 0.0028 | 0.0028 | 0.00000 |
| 48950 | sqa_val_05971 | True | 0.1914 | 0.01293 | 1.000 | True | 2.6745 | 2.6729 | 0.00158 |

## 저장량·비용 (build 기록 평균)

| 조건 | 쌍 유지 비율 | KV bytes | metadata bytes | prefill s | score s | select s | prune s | peak GB |
|---|---|---|---|---|---|---|---|---|
| full|none|none|k1|s0 | 1.000 | 67.7 MB | 1093 KB | 0.97 | 0.000 | 0.000 | 0.029 | 15.65 |
