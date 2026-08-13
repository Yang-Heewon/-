"""토큰 스팬 분해 — 시각/텍스트/싱크 인덱스."""
import torch


def token_spans(input_ids, config, n_sink=4):
    """input_ids: (1, L). 시각 토큰(image_pad) 위치와 주요 경계를 반환."""
    ids = input_ids[0]
    vis = (ids == config.image_token_id).nonzero(as_tuple=True)[0]
    assert len(vis) > 0, "시각 토큰이 없음 — image_token_id 확인"
    return {
        "visual": vis,                      # 시각 토큰 인덱스 (1D LongTensor)
        "vis_end": int(vis.max().item()),   # 마지막 시각 토큰 위치
        "sink": torch.arange(n_sink),       # 앞쪽 싱크 토큰
        "L": ids.shape[0],
    }
