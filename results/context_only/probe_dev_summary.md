# Context-only KV — probe — results/context_only/probe_qwen25vl_dev.jsonl

model qwen25vl · code 429ee2dc (dirty) · split dev · protect: prefix first 4 + tokenizer special ids in prefix (image/video placeholders excluded) · NLL: answers[0] body tokens, EOS excluded, teacher forced

## 단계 2 관측 (단일 prefill 통계)

| context | token | 시각 | 잔차 검사 최대 상대오차 | R 시각/비시각 평균 | D 시각/비시각 평균 | R–D token 순위 상관 | 상위 D token |
|---|---|---|---|---|---|---|---|
| 14107 | 1238 | 1222 | 4.88e-04 | 0.318/0.684 | 0.071/0.713 | 0.731 | Ċ@2, <|im_start|>@0, <|image_pad|>@282, <|image_pad|>@337, <|im_end|>@9 |
| 2178 | 1238 | 1222 | 4.88e-04 | 0.342/0.684 | 0.072/0.713 | 0.592 | Ċ@2, <|im_start|>@0, <|image_pad|>@947, <|image_pad|>@554, <|im_end|>@9 |
| 37948 | 1238 | 1222 | 4.88e-04 | 0.320/0.684 | 0.067/0.713 | 0.729 | Ċ@2, <|im_start|>@0, <|im_end|>@9, <|image_pad|>@312, <|image_pad|>@446 |
| 3350 | 1238 | 1222 | 4.88e-04 | 0.323/0.684 | 0.071/0.712 | 0.654 | Ċ@2, <|im_start|>@0, <|image_pad|>@162, <|im_end|>@9, <|image_pad|>@569 |
| 49909 | 1238 | 1222 | 4.88e-04 | 0.322/0.683 | 0.069/0.713 | 0.750 | Ċ@2, <|im_start|>@0, <|image_pad|>@572, <|image_pad|>@312, <|image_pad|>@313 |
| 5151 | 1238 | 1222 | 4.88e-04 | 0.318/0.683 | 0.072/0.713 | 0.738 | Ċ@2, <|im_start|>@0, <|image_pad|>@812, <|image_pad|>@968, <|image_pad|>@578 |
| 60887 | 1238 | 1222 | 4.88e-04 | 0.323/0.683 | 0.070/0.713 | 0.709 | Ċ@2, <|im_start|>@0, <|image_pad|>@493, <|im_end|>@9, <|image_pad|>@522 |
| 48950 | 1238 | 1222 | 4.88e-04 | 0.305/0.683 | 0.069/0.712 | 0.697 | Ċ@2, <|im_start|>@0, <|im_end|>@9, <|image_pad|>@751, <|image_pad|>@271 |
