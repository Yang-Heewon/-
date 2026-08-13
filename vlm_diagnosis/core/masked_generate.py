"""4D additive mask 아래에서의 greedy 생성 + cache resume 유틸.

- prefill은 4D causal mask(+선택적 column eviction)로 수행하고,
  step decode는 (1,1,1,kv_len) mask 행을 확장하며 진행한다.
- 생성 text 토큰의 mRoPE position은 세 축 모두 이전 최대값+1 (Qwen2.5-VL rope_delta
  semantics와 동일).
- M0 generation 검사와 M2-A 생성 기반 metric의 공용 부품.
"""
import torch

from .masked_eval import causal_mask_4d, evict_columns, mrope_position_ids


def _neg(dtype):
    return torch.finfo(dtype).min


@torch.no_grad()
def greedy_generate_masked(model, processor, ins, max_new_tokens=24,
                           evict_cols=None, row_start=None):
    """ins: signals.vlm_inputs(...) 출력 (프롬프트). greedy 생성 문자열 반환.

    evict_cols: 프롬프트 내 차단할 KV column 위치 (LongTensor), None이면 무차단.
    row_start: eviction이 적용되기 시작하는 query 행 (V2: vis_end+1, V1: P).
               생성 토큰 행에는 항상 적용된다.
    """
    device = ins["input_ids"].device
    P = ins["input_ids"].shape[1]
    attn2d = torch.ones(1, P, dtype=torch.long, device=device)
    pos = mrope_position_ids(model, ins["input_ids"], ins["image_grid_thw"], attn2d)

    m4 = causal_mask_4d(P, device)
    if evict_cols is not None and len(evict_cols) > 0:
        m4 = evict_columns(m4, evict_cols, P if row_start is None else row_start)

    out = model(input_ids=ins["input_ids"], pixel_values=ins["pixel_values"],
                image_grid_thw=ins["image_grid_thw"], attention_mask=m4,
                position_ids=pos, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[0, -1].argmax()
    next_pos = int(pos.max()) + 1

    generated = [int(next_id)]
    eos_ids = {model.config.eos_token_id} if isinstance(
        model.config.eos_token_id, int) else set(model.config.eos_token_id or [])
    for step in range(max_new_tokens - 1):
        if int(next_id) in eos_ids:
            break
        kv_len = P + len(generated)
        row = torch.zeros(1, 1, 1, kv_len, device=device, dtype=torch.float16)
        if evict_cols is not None and len(evict_cols) > 0:
            row[0, 0, 0, evict_cols] = _neg(torch.float16)
        p = torch.full((3, 1, 1), next_pos, device=device, dtype=pos.dtype)
        out = model(input_ids=next_id.view(1, 1), attention_mask=row,
                    position_ids=p, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[0, -1].argmax()
        next_pos += 1
        generated.append(int(next_id))

    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
