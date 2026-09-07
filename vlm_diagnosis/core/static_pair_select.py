"""정적 (layer, KV head, token) 쌍 선택 — 명시적 mapping, 정확한 전역 예산, 고정 seed 동점 처리
(docs/CONTEXT-ONLY-KV-COMPRESSION.md §4.3, §5, 단계 4 대조군).

점수 텐서는 항상 (L, H, T). token 신호 (L, T) 또는 (L-1, T) 를 쌍 점수로 바꾸는 규칙은 이름으로 고정한다:
    mlp_norm_same / r_same / hidden_rel_same / hidden_cos_same / r_std_same : score[l,h,i] = signal[l,i] (head 공통)
    d_same_zero0 : score[0,h,i] = 0, score[l,h,i] = D[l-1,i]  (l>=1)   — 문서 기본
    d_shift_prev : score[0:2,h,i] = 0, score[l,h,i] = D[l-2,i] (l>=2)
                   — same-layer D proxy를 한 layer 늦춤; 마지막 D 관측은 사용하지 않음
선택: 보호 쌍은 항상 포함(예산에 포함), 나머지는 전역 top-(B−보호). 동점은 seed 순열로 해소.
"""
from __future__ import annotations

import hashlib

import torch

TOKEN_MAPPINGS = ("mlp_norm_same", "r_same", "hidden_rel_same", "hidden_cos_same", "r_std_same",
                  "d_same_zero0", "d_shift_prev")


def map_token_signal(signal: torch.Tensor, mapping: str, n_layers: int, n_heads: int) -> torch.Tensor:
    for value, name in ((n_layers, "n_layers"), (n_heads, "n_heads")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if mapping not in TOKEN_MAPPINGS:
        raise ValueError(f"unknown mapping {mapping}")
    signal = torch.as_tensor(signal).float()
    expected_ndim = 1 if mapping == "r_std_same" else 2
    if signal.ndim != expected_ndim or signal.shape[-1] < 1:
        raise ValueError(f"{mapping} expects a nonempty {'(T,)' if expected_ndim == 1 else '2D'} signal")
    if not torch.isfinite(signal).all():
        raise ValueError("signal must contain finite values")
    if mapping in ("mlp_norm_same", "r_same", "hidden_rel_same", "hidden_cos_same"):
        if signal.shape[0] != n_layers:
            raise ValueError(f"{mapping} expects an (L, T) signal")
        return signal[:, None, :].expand(n_layers, n_heads, -1).clone()
    if mapping == "r_std_same":
        if signal.ndim != 1:
            raise ValueError("r_std_same expects a (T,) per-token standard deviation")
        return signal[None, None, :].expand(n_layers, n_heads, -1).clone()
    if mapping in ("d_same_zero0", "d_shift_prev"):
        if n_layers < 2:
            raise ValueError("D mappings require at least two layers")
        if signal.shape[0] != n_layers - 1:
            raise ValueError(f"{mapping} expects an (L-1, T) signal")
        zero = torch.zeros_like(signal[:1])
        full = torch.cat([zero, signal], dim=0)
        if mapping == "d_shift_prev":
            full = torch.cat([zero, full[:-1]], dim=0)
        return full[:, None, :].expand(n_layers, n_heads, -1).clone()
    raise ValueError(f"unknown mapping {mapping}")


def head_norm_scores(kv, kind: str) -> torch.Tensor:
    """K 또는 V 벡터 크기, head 별. kv: [(k, v)] with (1, H, T, d). 반환 (L, H, T) fp32 CPU."""
    idx = {"k": 0, "v": 1}[kind]
    return torch.stack([pair[idx][0].float().norm(dim=-1).cpu() for pair in kv])


def protected_positions(prefix_ids: torch.Tensor, special_ids, n_prefix: int = 4) -> torch.Tensor:
    """보호 위치 (T,) bool: 앞 min(n_prefix, T) + prefix 안의 tokenizer special token 위치의 합집합."""
    ids = prefix_ids[0] if prefix_ids.ndim == 2 else prefix_ids
    T = ids.numel()
    mask = torch.zeros(T, dtype=torch.bool)
    mask[:min(n_prefix, T)] = True
    if len(special_ids):
        mask |= torch.isin(ids.cpu(), torch.as_tensor(sorted(int(s) for s in special_ids)))
    return mask


def budget_pairs(keep_ratio: float, n_layers: int, n_heads: int, n_tokens: int) -> int:
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must be in (0, 1]")
    return int(round(keep_ratio * n_layers * n_heads * n_tokens))


def _tie_permutation(shape, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    n = 1
    for s in shape:
        n *= s
    return torch.randperm(n, generator=g).view(shape)


def select_pairs(score: torch.Tensor, protected: torch.Tensor, budget: int, seed: int,
                 candidates: torch.Tensor | None = None) -> torch.Tensor:
    """전역 top-budget 선택. score (L,H,T); protected (T,) 또는 (L,H,T) bool — 항상 보존, 예산에 포함.
    candidates: 선택 가능한 쌍 (None = 전부). 반환 keep (L,H,T) bool, keep.sum() == budget."""
    score = torch.as_tensor(score).float().cpu()
    L, H, T = score.shape
    protected = torch.as_tensor(protected, dtype=torch.bool).cpu()
    if protected.ndim == 1:
        protected = protected[None, None, :].expand(L, H, T)
    if protected.shape != score.shape:
        raise ValueError("protected mask shape mismatch")
    n_prot = int(protected.sum())
    if budget < n_prot:
        raise ValueError(f"budget {budget} smaller than protected pairs {n_prot}")
    if budget > score.numel():
        raise ValueError("budget exceeds pair count")
    eligible = ~protected if candidates is None else (~protected) & torch.as_tensor(candidates, dtype=torch.bool).cpu()
    n_free = budget - n_prot
    if n_free > int(eligible.sum()):
        raise ValueError("not enough eligible pairs for the budget")
    keep = protected.clone()
    if n_free:
        if not torch.isfinite(score[eligible]).all():
            raise ValueError("nonfinite scores among eligible pairs")
        # Preserve exact ordering even when pair ranks exceed FP32's 2**24
        # consecutive-integer range.
        tie = _tie_permutation(score.shape, seed)
        # 1차: 점수 내림차순, 2차: seed 순열 (동점 해소) — 안정 정렬 두 번
        flat_s, flat_t = score.flatten(), tie.flatten()
        idx = torch.nonzero(eligible.flatten(), as_tuple=True)[0]
        order = idx[torch.argsort(flat_t[idx], stable=True)]
        order = order[torch.argsort(flat_s[order], descending=True, stable=True)]
        keep.view(-1)[order[:n_free]] = True
    if int(keep.sum()) != budget or not bool(keep[protected].all()):
        raise RuntimeError("selection violated budget or protection")
    return keep


def random_scores(shape, seed: int) -> torch.Tensor:
    return torch.rand(shape, generator=torch.Generator().manual_seed(int(seed)))


def recent_scores(n_layers: int, n_heads: int, n_tokens: int) -> torch.Tensor:
    return torch.arange(n_tokens).float()[None, None, :].expand(n_layers, n_heads, n_tokens).clone()


def layer_matched_select(score, protected, budget, seed):
    """층별 삭제 수를 맞춘 대조군: 층마다 같은 유지 수 (B/L, 나머지는 앞 층부터 1개씩), 층 안에서 방법 순위."""
    score = torch.as_tensor(score).float().cpu()
    L, H, T = score.shape
    protected = torch.as_tensor(protected, dtype=torch.bool).cpu()
    if protected.ndim == 1:
        protected = protected[None, None, :].expand(L, H, T)
    per = [budget // L + (1 if l < budget % L else 0) for l in range(L)]
    keep = torch.zeros_like(protected)
    for l in range(L):
        keep[l] = select_pairs(score[l:l+1], protected[l:l+1], per[l], seed + l)[0]
    return keep


def boundary_control_select(score, protected, budget, seed, boundary_seed):
    """첫 layer 선택을 모든 방법에서 공유하는 대조군.
    B0 = max(P0, min(N0, B − Pother, round(B/L))); layer 0 = 보호분 + boundary_seed 로 뽑은 무작위 비보호 쌍,
    나머지 layer 에서 B−B0 개를 방법 점수로 선택."""
    score = torch.as_tensor(score).float().cpu()
    L, H, T = score.shape
    protected = torch.as_tensor(protected, dtype=torch.bool).cpu()
    if protected.ndim == 1:
        protected = protected[None, None, :].expand(L, H, T)
    N0, P0 = H * T, int(protected[0].sum())
    P_other = int(protected[1:].sum())
    B0 = max(P0, min(N0, budget - P_other, int(round(budget / L))))
    keep = torch.zeros_like(protected)
    keep[0] = select_pairs(random_scores((1, H, T), boundary_seed), protected[:1], B0, boundary_seed)[0]
    keep[1:] = select_pairs(score[1:], protected[1:], budget - B0, seed)
    return keep


def shuffled_D(R: torch.Tensor, seed: int):
    """층 순서를 고정 seed 순열 pi 로 섞은 뒤 D_shuffle[l-1] = |R[pi[l]] − R[pi[l-1]]|. (수집한 R 의 행만 섞음)"""
    _validate_R(R)
    L = R.shape[0]
    pi = torch.randperm(L, generator=torch.Generator().manual_seed(int(seed)))
    Rs = R[pi]
    return (Rs[1:] - Rs[:-1]).abs(), pi.tolist()


def anchor_D(R: torch.Tensor, seed: int):
    """anchor 층 l 을 유지하고 비교 층 j_l != l 을 고정 seed 로 뽑아 |R[l] − R[j_l]|, l = 1..L-1 (layer 0 경계 동일)."""
    _validate_R(R)
    L = R.shape[0]
    g = torch.Generator().manual_seed(int(seed))
    js = []
    for l in range(1, L):
        j = int(torch.randint(0, L - 1, (1,), generator=g))
        j = j + 1 if j >= l else j          # j != l
        js.append(j)
    D = torch.stack([(R[l] - R[js[l - 1]]).abs() for l in range(1, L)])
    return D, js


def _validate_R(R: torch.Tensor):
    if not isinstance(R, torch.Tensor) or R.ndim != 2 or R.shape[0] < 2 or R.shape[1] < 1:
        raise ValueError("R must be (L, T) with L >= 2 and T >= 1")
    if not torch.isfinite(R).all():
        raise ValueError("R must contain finite values")


def average_rank(x: torch.Tensor) -> torch.Tensor:
    """동점에 평균 순위를 부여하는 1D 순위 (1 부터)."""
    x = x.flatten().double()
    order = torch.argsort(x, stable=True)
    sorted_x = x[order]
    ranks = torch.empty_like(x)
    n = x.numel()
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_avg_rank(a: torch.Tensor, b: torch.Tensor, max_n: int = 50000, seed: int = 0):
    """평균 순위 Spearman. 상수 점수는 None (정의 불가)."""
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() > max_n:
        idx = torch.randperm(a.numel(), generator=torch.Generator().manual_seed(seed))[:max_n]
        a, b = a[idx], b[idx]
    if a.numel() < 2 or a.max() == a.min() or b.max() == b.min():
        return None
    ra, rb = average_rank(a), average_rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def keep_ids_per_head(keep: torch.Tensor):
    """(L,H,T) bool → RaggedKVCache 용 head 별 정렬된 logical token ID 목록."""
    L, H, T = keep.shape
    return [torch.nonzero(keep[l, h], as_tuple=True)[0].long() for l in range(L) for h in range(H)]


def selection_digest(keep: torch.Tensor) -> str:
    return hashlib.sha256(keep.cpu().numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# 2026-09-07 추가: 다양성 채우기, K 공간 farthest-point 순서, MLP 스파이크 기반 보호
# ---------------------------------------------------------------------------------------------
def massive_activation_positions(R: torch.Tensor, factor: float = 10.0) -> torch.Tensor:
    """(T,) bool: 어느 층에서든 R[l,i] 가 그 층 중앙값의 factor 배를 넘는 token (massive activation / sink 후보).
    R: (L, T). factor 는 개발 세트에서 고정 (화면 표본: 10 → 화면당 약 1개, 5 → 약 30개로 시각 token 까지 포함)."""
    _validate_R(R)
    med = R.median(dim=1, keepdim=True).values.clamp(min=1e-12)
    return ((R / med) > factor).any(0).cpu()


def farthest_point_order(keys: torch.Tensor, n_steps: int, start: int = 0, seed_mask: torch.Tensor | None = None) -> torch.Tensor:
    """keys (G, T, d): 그룹(층·head)마다 코사인 거리 farthest-point 순서. 반환 score (G, T):
    뽑힌 순번이 빠를수록 큼(T − 순번), 미선택은 남은 최소거리 × 0.5 (항상 선택분 아래).
    seed_mask (G, T) bool 이 있으면 그 조각들을 이미 뽑힌 대표로 두고 시작한다(다양성 채우기용)."""
    if keys.ndim != 3 or keys.shape[1] < 1:
        raise ValueError("keys must be (G, T, d)")
    G, T, _ = keys.shape
    K = keys.float()
    K = K / K.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dev = K.device
    score = torch.zeros(G, T, device=dev)
    mind = torch.full((G, T), float("inf"), device=dev)
    ar = torch.arange(G, device=dev)
    if seed_mask is not None:
        sm = seed_mask.to(dev)
        for g in range(G):
            idx = torch.nonzero(sm[g], as_tuple=True)[0]
            if idx.numel():
                d = 1.0 - K[g] @ K[g, idx].T                       # (T, n_seed)
                mind[g] = d.min(1).values
                mind[g, idx] = -1.0
        cur = mind.argmax(-1)
    else:
        cur = torch.full((G,), min(start, T - 1), dtype=torch.long, device=dev)
    for step in range(min(n_steps, T)):
        if seed_mask is not None and bool((mind[ar, cur] < 0).all()):
            break                                                   # 남은 후보 없음
        score[ar, cur] = float(T - step)
        c = K[ar, cur]
        dist = 1.0 - torch.einsum("bd,bnd->bn", c, K)
        mind = torch.minimum(mind, dist)
        mind[ar, cur] = -1.0
        cur = mind.argmax(-1)
    rest = score == 0
    score[rest] = (mind.clamp(min=0) * 0.5)[rest]
    return score.cpu()


def importance_diversity_select(score: torch.Tensor, keys: torch.Tensor, protected: torch.Tensor, budget: int,
                                seed: int, div_frac: float) -> torch.Tensor:
    """중요도 + 다양성. score (L,H,T); keys (L,H,T,d) 또는 [(k,v)] 형태의 kv.
    1) 보호 + 전역 top-(round((1−div_frac)·(B−보호))) 를 score 로 선택
    2) 나머지 예산을 (층, head) 그룹에 균등 배분해, 각 그룹에서 이미 뽑힌 조각들과 K 공간(코사인)에서
       가장 먼 조각부터 채운다 (farthest-point, 이미 뽑힌 것들을 시작 대표로 사용).
    반환 keep (L,H,T), keep.sum() == budget."""
    if not (0.0 <= div_frac <= 1.0):
        raise ValueError("div_frac must be in [0, 1]")
    score = torch.as_tensor(score).float().cpu()
    L, H, T = score.shape
    if not isinstance(keys, torch.Tensor):
        keys = torch.stack([pair[0][0].float() for pair in keys])          # (L,H,T,d)
    keys = keys[:, :, :T]
    if tuple(keys.shape[:3]) != (L, H, T):
        raise ValueError("keys shape disagrees with score")
    protected = torch.as_tensor(protected, dtype=torch.bool).cpu()
    if protected.ndim == 1:
        protected = protected[None, None, :].expand(L, H, T)
    n_prot = int(protected.sum())
    if budget < n_prot:
        raise ValueError(f"budget {budget} smaller than protected pairs {n_prot}")
    n_free = budget - n_prot
    n_div = int(round(div_frac * n_free))
    n_imp = n_free - n_div
    keep = select_pairs(score, protected, n_prot + n_imp, seed)
    if n_div:
        G = L * H
        per = [n_div // G + (1 if g < n_div % G else 0) for g in range(G)]
        order = farthest_point_order(keys.reshape(G, T, -1), max(per), seed_mask=keep.reshape(G, T))
        kf = keep.reshape(G, T).clone()
        for g in range(G):
            if per[g] == 0:
                continue
            cand = torch.nonzero(~kf[g], as_tuple=True)[0]
            if cand.numel() < per[g]:
                raise ValueError("not enough unselected pairs for the diversity fill")
            top = cand[torch.argsort(order[g, cand], descending=True, stable=True)[:per[g]]]
            kf[g, top] = True
        keep = kf.view(L, H, T)
    if int(keep.sum()) != budget or not bool(keep[protected].all()):
        raise RuntimeError("importance+diversity selection violated budget or protection")
    return keep
