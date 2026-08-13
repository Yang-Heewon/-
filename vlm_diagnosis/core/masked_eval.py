"""4D attention mask 기반 KV 축출 에뮬레이션 + teacher-forced 정답 logp.

핵심 아이디어: 캐시 수술 대신 전체 시퀀스(프롬프트+정답) 1회 forward에
4D additive mask를 넣어 "특정 행(query) 이후부터 특정 열(KV)이 안 보이는" 상황을 만든다.
- 열 차단 시작 행 = 프롬프트 끝  → 디코딩 시점 축출 (질문은 시각 KV를 본 상태)
- 열 차단 시작 행 = 시각 토큰 끝 → 질문 도착 전 축출 (D4의 캐시 재사용 semantics)
mRoPE position_ids는 2D 마스크로 미리 계산해 명시적으로 전달한다
(4D 마스크를 넣으면 내부 get_rope_index가 2D를 기대하므로).
"""
import torch


def causal_mask_4d(L, device, dtype=torch.float16):
    neg = torch.finfo(dtype).min
    m = torch.zeros(L, L, device=device, dtype=dtype)
    m.masked_fill_(torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), 1), neg)
    return m[None, None]  # (1,1,L,L)


def evict_columns(mask4d, cols, row_start):
    """row_start 이후의 모든 query 행에서 cols 위치의 KV를 차단."""
    m = mask4d.clone()
    m[0, 0, row_start:, cols] = torch.finfo(m.dtype).min
    return m


def mrope_position_ids(model, input_ids, image_grid_thw, attention_mask_2d):
    core = model.model if hasattr(model.model, "get_rope_index") else model
    pos, _ = core.get_rope_index(
        input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask_2d)
    return pos  # (3, B, L)


@torch.no_grad()
def answer_logp(model, input_ids, pixel_values, image_grid_thw,
                answer_start, attention_mask, position_ids=None):
    """정답 구간(answer_start..L)의 토큰별 logp 합. attention_mask는 2D 또는 4D."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = out.logits.float()  # fp16 → fp32로 안정화
    labels = input_ids[0, answer_start:]
    pred = logits[0, answer_start - 1:-1]
    logp = torch.log_softmax(pred, dim=-1)
    tok_logp = logp[torch.arange(len(labels)), labels]
    return tok_logp.sum().item(), tok_logp
