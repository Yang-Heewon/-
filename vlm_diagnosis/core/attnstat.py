"""메모리 안전 attention 통계 — output_attentions 절대 금지 (V100 OOM, DIAGNOSIS §3.1).

apply_multimodal_rotary_pos_emb를 감싸 post-RoPE q,k만 층 순서로 캡처하고,
필요한 축약량(특정 query 행들이 각 key 열에 준 attention 총량)만 청크로 계산한다.
"""
import math
import torch
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen
import transformers.models.qwen3_vl.modeling_qwen3_vl as _qwen3


class QKCapture:
    """with QKCapture() as cap: model(...)  →  cap.qk = [(q,k)] 층 순서."""

    def __init__(self):
        self.qk = []

    def __enter__(self):
        self._orig = _qwen.apply_multimodal_rotary_pos_emb
        self._orig3 = _qwen3.apply_rotary_pos_emb

        def wrapped(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
            qr, kr = self._orig(q, k, cos, sin, mrope_section, unsqueeze_dim)
            self.qk.append((qr.detach(), kr.detach()))
            return qr, kr

        def wrapped3(q, k, cos, sin, *a, **kw):
            qr, kr = self._orig3(q, k, cos, sin, *a, **kw)
            self.qk.append((qr.detach(), kr.detach()))
            return qr, kr

        _qwen.apply_multimodal_rotary_pos_emb = wrapped
        _qwen3.apply_rotary_pos_emb = wrapped3
        return self

    def __exit__(self, *exc):
        _qwen.apply_multimodal_rotary_pos_emb = self._orig
        _qwen3.apply_rotary_pos_emb = self._orig3
        return False


@torch.no_grad()
def recv_column_mass(qk, row_start, row_end, chunk=256):
    """rows [row_start, row_end)의 query가 각 key 열에 준 attention 총량.
    전 층·전 헤드·행 평균, (L,) 반환. causal 마스크 적용."""
    total = None
    for q, k in qk:
        q = q[0].float()   # (H, L, d)
        k = k[0].float()   # (Hkv, L, d)
        H, L, d = q.shape
        k = k.repeat_interleave(H // k.shape[0], dim=0)
        recv = torch.zeros(L, device=q.device)
        cols = torch.arange(L, device=q.device)
        for s in range(row_start, row_end, chunk):
            e = min(s + chunk, row_end)
            w = q[:, s:e] @ k.transpose(-1, -2) / math.sqrt(d)     # (H, R, L)
            rows = torch.arange(s, e, device=q.device)
            w.masked_fill_(cols[None, None, :] > rows[None, :, None], float("-inf"))
            recv += w.softmax(-1).sum(dim=(0, 1))
            del w
        total = recv if total is None else total + recv
    return total / (len(qk) * max(row_end - row_start, 1))


@torch.no_grad()
def recv_column_stats(qk, row_start, row_end, chunk=256):
    """행 구간의 query가 각 key 열에 준 attention의 (합, 최대) 동시 계산.
    합 = recv_column_mass와 동일(h2o/s5 관례), 최대 = KVzip 원저의 max 집계.
    반환: (sum_LT, max_LT) 각 (L,), 전 층 집계(합은 합산, 최대는 최대)."""
    total = None
    peak = None
    for q, k in qk:
        q = q[0].float(); k = k[0].float()
        H, L, d = q.shape
        k = k.repeat_interleave(H // k.shape[0], dim=0)
        cols = torch.arange(L, device=q.device)
        recv = torch.zeros(L, device=q.device)
        rmax = torch.zeros(L, device=q.device)
        for s0 in range(row_start, row_end, chunk):
            e = min(s0 + chunk, row_end)
            w = q[:, s0:e] @ k.transpose(-1, -2) / math.sqrt(d)
            rows = torch.arange(s0, e, device=q.device)
            w.masked_fill_(cols[None, None, :] > rows[None, :, None], float("-inf"))
            p = w.softmax(-1)
            recv += p.sum(dim=(0, 1))
            rmax = torch.maximum(rmax, p.amax(dim=(0, 1)))
            del w, p
        total = recv if total is None else total + recv
        peak = rmax if peak is None else torch.maximum(peak, rmax)
    return total, peak


@torch.no_grad()
def recv_snapkv(qk, row_start, row_end, kernel=7, chunk=256):
    """SnapKV 원저 점수의 시각 적응: 관찰 창(row_start..row_end) attention을
    head별로 합산 → head 차원 max → 1D max-pooling(kernel)로 클러스터 스무딩.
    (원저의 per-head topk는 토큰 단위 eviction 의미상 불가 — head-max로 집계,
    편차는 문서화.) 반환 (L,)."""
    import torch.nn.functional as F
    total = None
    for q, k in qk:
        q = q[0].float(); k = k[0].float()
        H, L, d = q.shape
        k = k.repeat_interleave(H // k.shape[0], dim=0)
        cols = torch.arange(L, device=q.device)
        per_head = torch.zeros(H, L, device=q.device)
        for s0 in range(row_start, row_end, chunk):
            e = min(s0 + chunk, row_end)
            w = q[:, s0:e] @ k.transpose(-1, -2) / math.sqrt(d)
            rows = torch.arange(s0, e, device=q.device)
            w.masked_fill_(cols[None, None, :] > rows[None, :, None], float("-inf"))
            per_head += w.softmax(-1).sum(dim=1)
            del w
        hmax = per_head.amax(dim=0)                       # head 차원 max
        sm = F.max_pool1d(hmax[None, None], kernel, 1, kernel // 2)[0, 0]
        total = sm if total is None else total + sm       # 층은 합산
    return total
