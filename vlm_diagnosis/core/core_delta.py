"""Fixed-budget KV-selection combiners.

``core_delta_keep`` preserves the original reconstruction/query experiment.
``dual_prefill_union_keep`` implements the current method: rank importance in
one image-only prefill and one image+existing-text prefill, independently take
their quota, union and de-duplicate the selections, then backfill from the
joint-prefill ranking to keep the exact byte-equivalent item budget.  These are
selection masks over a canonical cache; K and V values are never spliced from
incompatible forward passes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class CoreDeltaInfo:
    keep_count: int
    core_count: int          # core branch가 실제로 채운 개수
    query_count: int         # query branch가 실제로 채운 개수
    alpha: float
    n_tokens: int
    core_query_overlap: int  # core top-B와 query top-B(각각 예산 전체)의 교집합 크기
    nan_core: int
    nan_query: int

    def as_dict(self):
        return asdict(self)


@dataclass
class DualPrefillInfo:
    """Selection accounting for the two-prefill union policy.

    ``image_quota`` and ``joint_quota`` describe the two independently ranked
    initial sets.  An overlap between them is de-duplicated and then filled
    from the remaining joint-prefill ranking so that ``keep_count`` stays an
    exact budget.  ``image_count`` gives the protected image-prefill items in
    the final set; ``joint_count`` is the rest of the final set.
    """

    keep_count: int
    image_quota: int
    joint_quota: int
    image_count: int
    joint_count: int
    initial_overlap: int
    joint_backfill: int
    n_items: int
    nan_image: int
    nan_joint: int

    def as_dict(self):
        return asdict(self)


def _sanitize(score: torch.Tensor) -> tuple[torch.Tensor, int]:
    """NaN/inf → -inf (선택 최하위). float64 1D로 통일. NaN 개수 반환."""
    s = torch.as_tensor(score).detach().flatten().to(torch.float64)
    bad = ~torch.isfinite(s)
    n_bad = int(bad.sum())
    if n_bad:
        s = s.clone()
        s[bad] = float("-inf")
    return s, n_bad


def _stable_topk(score: torch.Tensor, k: int, allowed: torch.Tensor | None = None) -> list[int]:
    """점수 내림차순 상위 k개 index. 동점은 index 오름차순(결정적).
    allowed: bool mask (False인 위치는 후보 제외)."""
    n = score.shape[0]
    if k <= 0 or n == 0:
        return []
    cand = (torch.arange(n, device=score.device) if allowed is None
            else allowed.to(score.device).nonzero(as_tuple=True)[0])
    if cand.numel() == 0:
        return []
    # 후보 부분집합만 정렬: 제외된 index가 -inf 동점으로 섞여 들어오는 일을 원천 차단.
    # stable=True → 동점은 index 오름차순 유지 (결정적 tie-breaking).
    order = torch.argsort(score[cand], descending=True, stable=True)
    k = min(k, int(cand.numel()))
    return [int(cand[i]) for i in order[:k]]


def _eligible_mask(value, n: int, name: str) -> torch.Tensor:
    if value is None:
        return torch.ones(n, dtype=torch.bool)
    mask = torch.as_tensor(value, dtype=torch.bool).flatten().cpu()
    if mask.numel() != n:
        raise ValueError(f"{name} length mismatch: expected {n}, got {mask.numel()}")
    return mask


def dual_prefill_union_keep(
    image_score,
    joint_score,
    keep_count: int,
    image_quota: int,
    image_eligible=None,
    joint_eligible=None,
) -> tuple[set[int], DualPrefillInfo]:
    """Union independently important items from image-only and joint prefills.

    The two score tensors must already use the same coordinate system (for the
    VLM runners this is flattened ``layer x KV-head x prompt-position``).
    ``image_quota`` items are first ranked by the image-only prefill and the
    remaining quota by the image+text prefill.  The two initial top-k sets are
    formed independently; overlap is removed, then the joint ranking fills any
    resulting hole.  Consequently the final set is deterministic and contains
    exactly ``min(max(keep_count, 0), n_items)`` entries.

    Eligibility is explicit because text suffix positions do not exist in the
    image-only prefill.  Invalid numeric scores (NaN/Inf) are ranked last, but
    remain eligible when an exact budget requires them.
    """
    image_raw = torch.as_tensor(image_score)
    joint_raw = torch.as_tensor(joint_score)
    if image_raw.shape != joint_raw.shape:
        raise ValueError(
            f"score shape mismatch: image {tuple(image_raw.shape)} vs joint {tuple(joint_raw.shape)}"
        )
    image, nan_image = _sanitize(image_raw)
    joint, nan_joint = _sanitize(joint_raw)
    n = int(image.numel())
    budget = max(0, min(int(keep_count), n))
    requested_image = int(image_quota)
    if requested_image < 0 or requested_image > budget:
        raise ValueError(
            f"image_quota must be in [0, {budget}], got {requested_image}"
        )

    image_allowed = _eligible_mask(image_eligible, n, "image_eligible")
    joint_allowed = _eligible_mask(joint_eligible, n, "joint_eligible")
    if budget > int((image_allowed | joint_allowed).sum()):
        raise ValueError("combined eligible items cannot satisfy keep_count")

    # If the image-only coordinate space is smaller than its requested quota,
    # the missing slots are deliberately handed to the joint pass.
    actual_image_quota = min(requested_image, int(image_allowed.sum()))
    requested_joint = budget - requested_image
    initial_image = _stable_topk(image, actual_image_quota, image_allowed)
    initial_joint = _stable_topk(joint, requested_joint, joint_allowed)
    image_set, joint_set = set(initial_image), set(initial_joint)
    overlap = len(image_set & joint_set)
    keep = image_set | joint_set

    # De-duplication (and an undersized image-only domain) may leave holes.
    # Prefer the joint/full-prefix ranking for backfill because it is defined
    # over the complete target cache, including text-only positions.
    need = budget - len(keep)
    joint_backfill = 0
    if need > 0:
        allowed = joint_allowed.clone()
        if keep:
            allowed[torch.tensor(sorted(keep), dtype=torch.long)] = False
        extra = _stable_topk(joint, need, allowed)
        keep.update(extra)
        joint_backfill += len(extra)

    # A restricted joint domain is supported for unit use.  Fall back to any
    # remaining image-prefill item before declaring the exact budget impossible.
    need = budget - len(keep)
    if need > 0:
        allowed = image_allowed.clone()
        if keep:
            allowed[torch.tensor(sorted(keep), dtype=torch.long)] = False
        keep.update(_stable_topk(image, need, allowed))
    if len(keep) != budget:
        raise ValueError("eligible items cannot satisfy exact keep_count after de-duplication")

    # Overlapping initial picks are attributed to the protected image branch;
    # every other final item was supplied by the joint branch or its backfill.
    info = DualPrefillInfo(
        keep_count=budget,
        image_quota=requested_image,
        joint_quota=requested_joint,
        image_count=len(image_set),
        joint_count=len(keep - image_set),
        initial_overlap=overlap,
        joint_backfill=joint_backfill,
        n_items=n,
        nan_image=nan_image,
        nan_joint=nan_joint,
    )
    return keep, info


def core_delta_keep(core_score, query_score, keep_count: int, alpha: float,
                    ) -> tuple[set[int], CoreDeltaInfo]:
    """VLM_idea.md §10 Step 1의 실제 구현.

    core_score, query_score: 같은 시각 토큰 순서의 1D 점수 (클수록 보존 우선).
    keep_count: 최종 보존 토큰 수 B (byte 예산에서 kv_baselines.max_keep_for_budget로 유도).
    alpha: core 비율 ∈ [0, 1].

    반환: (시각 순서 index 집합 — 크기 == min(keep_count, n), 진단 정보).
    keep_count<=0 → 빈 집합. NaN 점수는 최하위. 중복 없음이 보장된다.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    c, nan_c = _sanitize(core_score)
    q, nan_q = _sanitize(query_score)
    if c.shape != q.shape:
        raise ValueError(f"score length mismatch: core {tuple(c.shape)} vs query {tuple(q.shape)}")
    n = int(c.shape[0])
    B = max(0, min(int(keep_count), n))
    B_C = int(round(alpha * B))
    B_C = max(0, min(B_C, B))

    core = _stable_topk(c, B_C)
    allowed = torch.ones(n, dtype=torch.bool)
    if core:
        allowed[torch.tensor(core, dtype=torch.long)] = False
    B_Q = B - len(core)
    query = _stable_topk(q, B_Q, allowed=allowed)

    keep = set(core) | set(query)
    # 불변식: 정확히 B개, 중복 없음
    assert len(keep) == len(core) + len(query) == B, (len(keep), len(core), len(query), B)

    overlap = len(set(_stable_topk(c, B)) & set(_stable_topk(q, B))) if B > 0 else 0
    info = CoreDeltaInfo(keep_count=B, core_count=len(core), query_count=len(query),
                         alpha=float(alpha), n_tokens=n, core_query_overlap=overlap,
                         nan_core=nan_c, nan_query=nan_q)
    return keep, info


def rank_normalize(score) -> torch.Tensor:
    """점수를 [0,1] rank 백분위로 변환 (스케일 불변 결합용). NaN은 0."""
    s, _ = _sanitize(score)
    n = s.shape[0]
    if n == 0:
        return s
    order = torch.argsort(s, descending=False, stable=True)
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = torch.arange(n, dtype=torch.float64)
    r = ranks / max(n - 1, 1)
    r[~torch.isfinite(s)] = 0.0
    return r


def weighted_sum_keep(core_score, query_score, keep_count: int, w_core: float) -> set[int]:
    """약한 형태 ablation (§9): rank 정규화 점수의 가중합 top-k.
    core_delta_keep와 같은 예산 B를 쓰므로 byte는 동일. 결합 방식의 차이만 격리한다."""
    if not (0.0 <= w_core <= 1.0):
        raise ValueError("w_core must be in [0,1]")
    fused = w_core * rank_normalize(core_score) + (1.0 - w_core) * rank_normalize(query_score)
    n = fused.shape[0]
    B = max(0, min(int(keep_count), n))
    return set(_stable_topk(fused, B))


@torch.no_grad()
def visual_kv_invariance(model, processor, img, text_a: str, text_b: str, device):
    """§3의 전제를 수치로 확인: 이미지 뒤에 서로 다른 텍스트를 붙여도 시각 위치의
    K/V가 동일한가. 두 prefill의 past_key_values를 시각 position에서 비교한다.

    반환 dict: 층별 최대 |ΔK|, |ΔV|의 전체 최대와, 비교 기준으로 K/V 절대값 최대.
    (fp16에서 위치 무관 연산은 bit-동일해야 하며, 차이가 0이 아니면 kernel 비결정성
    수준인지 실제 의존성인지 크기로 판별한다.)
    """
    from .signals import vlm_inputs
    from .spans import token_spans

    def _kv(text):
        ins = vlm_inputs(processor, img, text, device)
        out = model(input_ids=ins["input_ids"], attention_mask=torch.ones_like(ins["input_ids"]),
                    pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"],
                    use_cache=True)
        sp = token_spans(ins["input_ids"], model.config)
        pk = out.past_key_values
        layers = []
        n_layers = len(pk.layers) if hasattr(pk, "layers") else len(pk)
        for li in range(n_layers):
            if hasattr(pk, "layers"):
                k, v = pk.layers[li].keys, pk.layers[li].values
            elif hasattr(pk, "key_cache"):
                k, v = pk.key_cache[li], pk.value_cache[li]
            else:
                k, v = pk[li]
            vis = sp["visual"].to(k.device)
            layers.append((k[0, :, vis].float().cpu(), v[0, :, vis].float().cpu()))
        return layers, int(sp["visual"].numel())

    def _cmp(A, B):
        dk = max(float((ka - kb).abs().max()) for (ka, _), (kb, _) in zip(A, B))
        dv = max(float((va - vb).abs().max()) for (_, va), (_, vb) in zip(A, B))
        nk = sum(float((ka - kb).norm() ** 2) for (ka, _), (kb, _) in zip(A, B)) ** 0.5
        nv = sum(float((va - vb).norm() ** 2) for (_, va), (_, vb) in zip(A, B)) ** 0.5
        kf = sum(float(ka.norm() ** 2) for ka, _ in A) ** 0.5
        vf = sum(float(va.norm() ** 2) for _, va in A) ** 0.5
        return {"max_abs_dK": dk, "max_abs_dV": dv,
                "rel_fro_dK": nk / max(kf, 1e-12), "rel_fro_dV": nv / max(vf, 1e-12)}

    A, n_a = _kv(text_a)
    B, n_b = _kv(text_b)
    A2, _ = _kv(text_a)            # 결정성 대조군: 같은 텍스트 두 번
    assert n_a == n_b, (n_a, n_b)
    main = _cmp(A, B)
    ctrl = _cmp(A, A2)
    kmax = max(float(ka.abs().max()) for ka, _ in A)
    vmax = max(float(va.abs().max()) for _, va in A)
    # 해석: 시각 위치 K/V는 뒤 텍스트에 (인과 마스크로) 의존할 수 없다. 남는 차이는
    # 시퀀스 길이가 달라 GEMM/softmax 커널 타일링이 바뀌며 생기는 fp16 반올림 잡음이며,
    # 상대 Frobenius 오차가 ~1e-3 이하이고 값 크기 대비 ulp 수준이면 "동일"로 본다.
    return {"n_visual": n_a, "n_layers": len(A), **main,
            "K_abs_max": kmax, "V_abs_max": vmax,
            "same_text_control": ctrl,
            "identical_bitwise": bool(main["max_abs_dK"] == 0.0 and main["max_abs_dV"] == 0.0),
            "identical_up_to_fp16_noise": bool(main["rel_fro_dK"] < 1e-2 and main["rel_fro_dV"] < 1e-2)}
