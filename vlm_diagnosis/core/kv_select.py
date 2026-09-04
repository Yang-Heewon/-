"""전체 KV 선택 (VLM_idea.md §7) — 후보 집합을 (층, KV head, token 위치) 세 짝으로 잡는다.

시각 token만이 아니라 프롬프트의 모든 token(system 문구, vision 경계, 시각, 질문,
assistant 머리말)이 후보이며, 같은 token 위치라도 층·KV head마다 남길지 여부가 다를 수
있다. GQA(grouped-query attention)에서는 query head가 아니라 **KV head** 단위로 비용을
센다 (한 KV head를 지우면 그 그룹의 query head 전부가 못 본다).

구성 요소
  per_head_column_stats  캡처한 (q, k)로 (층, KV head, 열)별 attention 평균·최대 계산
  select_triples         고정 예산(세 짝 개수) 아래 core–delta 결합 (core_delta_keep 재사용)
  select_dual_prefill_*  image-only/joint-prefill 독립 top-k의 예산 고정 합집합
  build_eviction_mask    (1,1,Lq,Lk) 인과 마스크 → (1,Hq,Lq,Lk) head별 열 차단 마스크
  PerHeadEviction        HF eager attention을 층별로 가로채 위 마스크를 적용하는 컨텍스트
  greedy_generate_perhead  마스크 아래 greedy 생성 (masked_generate 재사용)

시뮬레이션 주의: head 단위 선택은 HF 캐시(층마다 (B,H,L,d) 직사각형)로는 물리적으로
표현되지 않는다(ragged). byte는 회계로만 기록하며, token 단위(공통 마스크) 선택만
캐시 잘라내기로 물리 검증이 가능하다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q25
import transformers.models.qwen3_vl.modeling_qwen3_vl as _q3

from .core_delta import (
    core_delta_keep, dual_prefill_union_keep, _stable_topk, _sanitize,
)
from .masked_generate import greedy_generate_masked


# ----------------------------------------------------------------------------- 점수
@torch.no_grad()
def per_head_column_stats(qk, row_start: int, row_end: int, chunk: int = 256):
    """rows [row_start, row_end)의 query가 각 열에 준 attention을 (층, KV head, 열)로 집계.

    반환 (mean_mass, peak): 각 (n_layers, H_kv, L) CPU float32.
      mean_mass = 행 평균 후 그룹 내 query head 평균  (S1/h2o 관례의 head별 분해)
      peak      = 행·그룹 query head에 대한 최대     (KVzip 원저 max 집계의 head별 분해)
    층·head를 다시 평균/최대로 합치면 recv_column_mass / recv_column_stats 와 순위가 같다.
    """
    sums, peaks = [], []
    for q, k in qk:
        q = q[0].float()                       # (Hq, L, d)
        k = k[0].float()                       # (Hkv, L, d)
        Hq, L, d = q.shape
        Hkv = k.shape[0]
        g = Hq // Hkv
        kk = k.repeat_interleave(g, dim=0)     # (Hq, L, d)
        cols = torch.arange(L, device=q.device)
        s_acc = torch.zeros(Hq, L, device=q.device)
        m_acc = torch.zeros(Hq, L, device=q.device)
        for s0 in range(row_start, row_end, chunk):
            e = min(s0 + chunk, row_end)
            w = q[:, s0:e] @ kk.transpose(-1, -2) / math.sqrt(d)      # (Hq, R, L)
            rows = torch.arange(s0, e, device=q.device)
            w.masked_fill_(cols[None, None, :] > rows[None, :, None], float("-inf"))
            p = w.softmax(-1)
            s_acc += p.sum(1)
            m_acc = torch.maximum(m_acc, p.amax(1))
            del w, p
        n_rows = max(row_end - row_start, 1)
        sums.append((s_acc / n_rows).view(Hkv, g, L).mean(1).cpu())
        peaks.append(m_acc.view(Hkv, g, L).amax(1).cpu())
    return torch.stack(sums), torch.stack(peaks)


# ----------------------------------------------------------------------------- 선택
@dataclass
class TripleSelection:
    granularity: str          # "head" | "token"
    budget_triples: int       # 목표 (층×head×token 세 짝 수)
    kept_triples: int         # 실제 보존 수
    forced_triples: int       # 보호로 강제 보존된 수 (예산에 포함)
    core_count: int
    query_count: int
    alpha: float

    def as_dict(self):
        return asdict(self)


@dataclass
class DualPrefillSelection:
    granularity: str
    budget_triples: int
    kept_triples: int
    forced_triples: int
    image_fraction: float
    image_quota: int
    joint_quota: int
    image_count: int
    joint_count: int
    initial_overlap: int
    joint_backfill: int

    def as_dict(self):
        return asdict(self)


def _as_bool(x, shape):
    if x is None:
        return torch.zeros(shape, dtype=torch.bool)
    return torch.as_tensor(x).bool().reshape(shape)


def select_triples(core, query, budget_triples: int, alpha: float,
                   forced=None) -> tuple[torch.Tensor, TripleSelection]:
    """head 단위 core–delta 선택. core/query: (n_layers, H_kv, P) 점수 (core는 정의되지
    않은 위치에 -inf). forced: 같은 모양 bool — 항상 보존하며 예산에 **포함**한다.
    반환 keep bool (n_layers, H_kv, P)."""
    query = torch.as_tensor(query).float()
    core = torch.as_tensor(core).float()
    assert core.shape == query.shape, (core.shape, query.shape)
    shape = query.shape
    n = query.numel()
    forced = _as_bool(forced, shape)
    B = max(0, min(int(budget_triples), n))
    keep = forced.clone()
    n_forced = int(forced.sum())
    remaining = B - n_forced
    cand = (~forced).flatten().nonzero(as_tuple=True)[0]
    cc = qc = 0
    if remaining > 0 and cand.numel() > 0:
        sub, info = core_delta_keep(core.flatten()[cand], query.flatten()[cand], remaining, alpha)
        chosen = cand[torch.tensor(sorted(sub), dtype=torch.long)]
        keep.view(-1)[chosen] = True
        cc, qc = info.core_count, info.query_count
    sel = TripleSelection("head", B, int(keep.sum()), n_forced, cc, qc, float(alpha))
    return keep, sel


def select_tokens(core_tok, query_tok, budget_tokens: int, alpha: float, n_layers: int,
                  n_kv_heads: int, forced_tok=None) -> tuple[torch.Tensor, TripleSelection]:
    """token 단위(모든 층·head 공통 마스크) core–delta 선택. 점수는 (P,) 벡터.
    반환 keep bool (n_layers, H_kv, P) — token 결정을 층·head에 복제."""
    query_tok = torch.as_tensor(query_tok).float().flatten()
    core_tok = torch.as_tensor(core_tok).float().flatten()
    P = query_tok.shape[0]
    forced_tok = _as_bool(forced_tok, (P,))
    B = max(0, min(int(budget_tokens), P))
    keep_tok = forced_tok.clone()
    n_forced = int(forced_tok.sum())
    remaining = B - n_forced
    cand = (~forced_tok).nonzero(as_tuple=True)[0]
    cc = qc = 0
    if remaining > 0 and cand.numel() > 0:
        sub, info = core_delta_keep(core_tok[cand], query_tok[cand], remaining, alpha)
        keep_tok[cand[torch.tensor(sorted(sub), dtype=torch.long)]] = True
        cc, qc = info.core_count, info.query_count
    keep = keep_tok[None, None, :].expand(n_layers, n_kv_heads, P).clone()
    per = n_layers * n_kv_heads
    sel = TripleSelection("token", B * per, int(keep.sum()), n_forced * per, cc * per, qc * per,
                          float(alpha))
    return keep, sel


def select_dual_prefill_triples(
    image_score,
    joint_score,
    budget_triples: int,
    image_fraction: float,
    forced=None,
    image_eligible=None,
) -> tuple[torch.Tensor, DualPrefillSelection]:
    """Select an exact triple budget from two independent prefill rankings.

    ``image_eligible`` marks the coordinates that exist in the image-only
    prefill.  The joint prefill is defined over every target-cache coordinate.
    Forced entries (for example attention sinks) are included in the budget
    before the remaining budget is split between the two rankings.
    """
    if not 0.0 <= image_fraction <= 1.0:
        raise ValueError("image_fraction must be in [0,1]")
    image_score = torch.as_tensor(image_score).float()
    joint_score = torch.as_tensor(joint_score).float()
    if image_score.shape != joint_score.shape:
        raise ValueError(
            f"score shape mismatch: image {tuple(image_score.shape)} vs "
            f"joint {tuple(joint_score.shape)}"
        )
    shape = joint_score.shape
    n = joint_score.numel()
    budget = max(0, min(int(budget_triples), n))
    forced = _as_bool(forced, shape)
    n_forced = int(forced.sum())
    if n_forced > budget:
        raise ValueError(
            f"forced entries ({n_forced}) exceed triple budget ({budget})"
        )
    if image_eligible is None:
        image_eligible = torch.ones(shape, dtype=torch.bool)
    else:
        image_eligible = _as_bool(image_eligible, shape)
    candidates = ~forced
    remaining = budget - n_forced
    image_quota = int(round(image_fraction * remaining))
    chosen, info = dual_prefill_union_keep(
        image_score,
        joint_score,
        remaining,
        image_quota,
        image_eligible=image_eligible & candidates,
        joint_eligible=candidates,
    )
    keep = forced.clone()
    if chosen:
        keep.view(-1)[torch.tensor(sorted(chosen), dtype=torch.long)] = True
    sel = DualPrefillSelection(
        granularity="head",
        budget_triples=budget,
        kept_triples=int(keep.sum()),
        forced_triples=n_forced,
        image_fraction=float(image_fraction),
        image_quota=info.image_quota,
        joint_quota=info.joint_quota,
        image_count=info.image_count,
        joint_count=info.joint_count,
        initial_overlap=info.initial_overlap,
        joint_backfill=info.joint_backfill,
    )
    return keep, sel


def select_dual_prefill_tokens(
    image_score,
    joint_score,
    budget_tokens: int,
    image_fraction: float,
    n_layers: int,
    n_kv_heads: int,
    forced_tok=None,
    image_eligible_tok=None,
) -> tuple[torch.Tensor, DualPrefillSelection]:
    """Token-mask counterpart of :func:`select_dual_prefill_triples`."""
    image_score = torch.as_tensor(image_score).float().flatten()
    joint_score = torch.as_tensor(joint_score).float().flatten()
    if image_score.shape != joint_score.shape:
        raise ValueError(
            f"score shape mismatch: image {tuple(image_score.shape)} vs "
            f"joint {tuple(joint_score.shape)}"
        )
    prompt_len = joint_score.numel()
    forced_tok = _as_bool(forced_tok, (prompt_len,))
    image_eligible_tok = (_as_bool(image_eligible_tok, (prompt_len,))
                          if image_eligible_tok is not None
                          else torch.ones(prompt_len, dtype=torch.bool))
    keep_one, token_sel = select_dual_prefill_triples(
        image_score.view(1, 1, prompt_len),
        joint_score.view(1, 1, prompt_len),
        budget_tokens,
        image_fraction,
        forced=forced_tok.view(1, 1, prompt_len),
        image_eligible=image_eligible_tok.view(1, 1, prompt_len),
    )
    keep_tok = keep_one[0, 0]
    keep = keep_tok[None, None, :].expand(n_layers, n_kv_heads, prompt_len).clone()
    per_token = n_layers * n_kv_heads
    sel = DualPrefillSelection(
        granularity="token",
        budget_triples=token_sel.budget_triples * per_token,
        kept_triples=int(keep.sum()),
        forced_triples=token_sel.forced_triples * per_token,
        image_fraction=token_sel.image_fraction,
        image_quota=token_sel.image_quota * per_token,
        joint_quota=token_sel.joint_quota * per_token,
        image_count=token_sel.image_count * per_token,
        joint_count=token_sel.joint_count * per_token,
        initial_overlap=token_sel.initial_overlap * per_token,
        joint_backfill=token_sel.joint_backfill * per_token,
    )
    return keep, sel


def uniform_token_keep(P: int, budget_tokens: int, n_layers: int, n_kv_heads: int,
                       forced_tok=None) -> torch.Tensor:
    """row-major 등간격 token 선택(공간·의미 무관 통제군)을 층·head에 복제."""
    forced_tok = _as_bool(forced_tok, (P,))
    B = max(0, min(int(budget_tokens), P))
    keep_tok = forced_tok.clone()
    k = B - int(forced_tok.sum())
    if k > 0:
        cand = (~forced_tok).nonzero(as_tuple=True)[0]
        pick = torch.linspace(0, cand.numel() - 1, steps=min(k, cand.numel())).round().long()
        keep_tok[cand[pick]] = True
    return keep_tok[None, None, :].expand(n_layers, n_kv_heads, P).clone()


# ----------------------------------------------------------------------------- 마스크
def build_eviction_mask(attention_mask: torch.Tensor, evict_heads: torch.Tensor,
                        row_start: int, groups: int) -> torch.Tensor:
    """attention_mask (1,1,Lq,Lk) 가산 마스크에 head별 열 차단을 얹어 (1,Hq,Lq,Lk)로 반환.

    evict_heads: (H_kv, P) bool — 프롬프트 열 중 지울 것. P 이후 열(생성 token)은 지우지 않음.
    row_start: prefill(Lq>1, 행 index=절대 위치)에서 차단이 시작되는 행. decode(Lq==1)
               행은 생성 token이므로 항상 차단한다.
    """
    _, _, Lq, Lk = attention_mask.shape
    Hkv, P = evict_heads.shape
    Hq = Hkv * groups
    dev = attention_mask.device
    ev = evict_heads.to(dev).repeat_interleave(groups, dim=0)               # (Hq, P)
    col = torch.zeros(Hq, Lk, dtype=torch.bool, device=dev)
    n = min(P, Lk)
    col[:, :n] = ev[:, :n]
    if Lq > 1:
        assert Lq == Lk, "prefill은 전체 시퀀스를 한 번에 넣는 경우만 지원 (Lq == Lk)"
        rows = torch.arange(Lq, device=dev) >= row_start                    # (Lq,)
        block = col[None, :, None, :] & rows[None, None, :, None]           # (1,Hq,Lq,Lk)
    else:
        block = col[None, :, None, :]
    m = attention_mask.expand(1, Hq, Lq, Lk).clone()
    m.masked_fill_(block, torch.finfo(m.dtype).min)
    return m


class PerHeadEviction:
    """with PerHeadEviction(keep, row_start): ...  — 층별 (H_kv, P) keep 마스크를
    HF eager attention 호출에 주입한다. keep: (n_layers, H_kv, P) bool.
    vision encoder의 attention(layer_idx 없음)과 4D 마스크가 없는 호출은 그대로 통과."""

    def __init__(self, keep: torch.Tensor, row_start: int):
        self.evict = (~keep.bool())
        self.row_start = int(row_start)
        self.calls = 0

    def __enter__(self):
        self._o25 = _q25.eager_attention_forward
        self._o3 = _q3.eager_attention_forward
        ev_all, row_start, self_ref = self.evict, self.row_start, self

        def make(orig):
            def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
                li = getattr(module, "layer_idx", None)
                if li is None or attention_mask is None or attention_mask.dim() != 4:
                    return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
                Lk = key.shape[2]
                m = build_eviction_mask(attention_mask[:, :, :, :Lk], ev_all[li], row_start,
                                        module.num_key_value_groups)
                self_ref.calls += 1
                return orig(module, query, key, value, m, scaling, dropout, **kw)
            return wrapped

        _q25.eager_attention_forward = make(self._o25)
        _q3.eager_attention_forward = make(self._o3)
        return self

    def __exit__(self, *exc):
        _q25.eager_attention_forward = self._o25
        _q3.eager_attention_forward = self._o3
        return False


@torch.no_grad()
def greedy_generate_perhead(model, processor, ins, keep: torch.Tensor, row_start: int,
                            max_new_tokens: int = 32) -> str:
    """keep (n_layers, H_kv, P) bool 아래 greedy 생성. row_start=P 면 'prefill 후 잘라내기'
    (질문 행은 전부 보고 생성 행만 제한), row_start=vis_end+1 이면 '질문 도착 전 잘라내기'."""
    with PerHeadEviction(keep.to(ins["input_ids"].device), row_start) as ctx:
        out = greedy_generate_masked(model, processor, ins, max_new_tokens=max_new_tokens)
    assert ctx.calls > 0, "eviction 마스크가 한 번도 적용되지 않음 — attention 경로 확인"
    return out


# ----------------------------------------------------------------------------- 회계
def kept_composition(keep: torch.Tensor, visual_idx: torch.Tensor, n_sink: int = 4) -> dict:
    """보존 세 짝의 구성: 시각/텍스트 비율, sink(앞 n_sink token) 보존 비율, 층별 개수."""
    n_layers, Hkv, P = keep.shape
    is_vis = torch.zeros(P, dtype=torch.bool)
    is_vis[visual_idx.cpu()] = True
    kept = keep.sum().item()
    vis_kept = keep[:, :, is_vis].sum().item()
    sink_kept = keep[:, :, :n_sink].float().mean().item() if n_sink > 0 else float("nan")
    per_layer = keep.sum(dim=(1, 2)).tolist()
    return {"kept_triples": int(kept),
            "keep_frac_visual": (vis_kept / kept) if kept else float("nan"),
            "keep_frac_text": ((kept - vis_kept) / kept) if kept else float("nan"),
            "visual_keep_ratio": vis_kept / max(1, int(is_vis.sum()) * n_layers * Hkv),
            "text_keep_ratio": (kept - vis_kept) / max(1, int((~is_vis).sum()) * n_layers * Hkv),
            "sink_kept_frac": sink_kept,
            "per_layer_kept": per_layer}


def kv_bytes(kept_triples: int, head_dim: int, element_bytes: int = 2) -> int:
    """K와 V 본체 byte (세 짝 하나 = K와 V 각 head_dim 원소)."""
    return int(kept_triples) * 2 * head_dim * element_bytes


def index_bytes(n_layers: int, n_kv_heads: int, P: int, granularity: str) -> int:
    """보존 위치 색인 byte — bitmask 기준 (head: 세 짝마다 1bit, token: token마다 1bit)."""
    bits = n_layers * n_kv_heads * P if granularity == "head" else P
    return (bits + 7) // 8
