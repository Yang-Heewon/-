"""단일 prefill MLP·hidden-state 통계 수집기 (docs/CONTEXT-ONLY-KV-COMPRESSION.md §4).

pre-norm decoder 블록 하나의 관측 위치:
    x[l]    = 블록 입력 잔차                      (decoder layer pre-hook)
    r[l]    = x[l] + attention 출력               (post_attention_layernorm pre-hook 입력)
    m[l]    = MLP(Norm2(r[l]))                    (mlp forward hook 출력)
    x[l+1]  = r[l] + m[l]                         (decoder layer output hook)

통계 (batch 1, L 층, T token, 모두 FP32로 축약):
    MLP_norm[l,i]      = ‖m[l,i]‖₂
    Residual_norm[l,i] = ‖r[l,i]‖₂
    R[l,i]             = MLP_norm / (Residual_norm + eps)
    D[l-1,i]           = |R[l,i] − R[l-1,i]|,  l = 1..L-1        → shape (L-1, T)
    hidden_rel[l,i]    = ‖x[l+1,i] − x[l,i]‖ / (‖x[l,i]‖ + eps)   (attention + MLP 변화, 대조군)
    hidden_cos[l,i]    = 1 − cos(x[l,i], x[l+1,i])                (방향 변화, 대조군)
필수 검사: x[l+1] ≈ r[l] + m[l]. 층마다 최대 상대 오차를 기록하고 허용오차를 넘으면 실패.
stats 모드에서는 d 차원을 hook 안에서 즉시 축약하고 raw activation 을 보관하지 않는다.
층 hook 호출은 forward 당 정확히 한 번이어야 하며, 재호출은 조용히 덮어쓰지 않고 실패한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def language_layers(model):
    core = getattr(model, "model", model)
    return (core.language_model if hasattr(core, "language_model") else core).layers


@dataclass
class MLPDynamics:
    mlp_norm: torch.Tensor          # (L, T)
    residual_norm: torch.Tensor     # (L, T)
    R: torch.Tensor                 # (L, T)
    D: torch.Tensor                 # (L-1, T)
    hidden_rel: torch.Tensor        # (L, T)
    hidden_cos: torch.Tensor        # (L, T)
    residual_max_abs_err: torch.Tensor   # (L,)
    residual_max_rel_err: torch.Tensor   # (L,)
    eps: float
    n_layers: int
    n_tokens: int
    dtype: str
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return {"mlp_norm": self.mlp_norm, "residual_norm": self.residual_norm, "R": self.R, "D": self.D,
                "hidden_rel": self.hidden_rel, "hidden_cos": self.hidden_cos}


class MLPDynamicsCollector:
    """with MLPDynamicsCollector(model) as col: model(...); stats = col.result()

    mode="stats": norm 만 유지 (배포 비용 경로). mode="debug": 짧은 context 에서 raw r, m, x 도 보관.
    """

    def __init__(self, model, mode="stats", eps=1e-6, rel_tol=None):
        if mode not in ("stats", "debug"):
            raise ValueError("mode must be 'stats' or 'debug'")
        self.layers = language_layers(model)
        if not len(self.layers):
            raise ValueError("no decoder layers found")
        self.mode, self.eps = mode, float(eps)
        self.rel_tol = rel_tol
        self.handles = []
        self.reset()

    def reset(self):
        self._x, self._r, self._m = {}, {}, {}
        self.mlp_norm, self.res_norm = {}, {}
        self.hidden_rel, self.hidden_cos = {}, {}
        self.abs_err, self.rel_err = {}, {}
        self.calls = {}
        self.raw = {} if self.mode == "debug" else None
        self.dtype = None

    # ---- hooks -------------------------------------------------------------------------
    def _layer_pre(self, li, module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if h.ndim != 3 or h.shape[0] != 1:
            raise ValueError("collector requires batch-one (1, T, d) hidden states")
        n = self.calls.get(li, 0)
        if n:
            raise RuntimeError(f"decoder layer {li} was called {n+1} times in one collection; call reset() between forwards")
        self.calls[li] = 1
        self.dtype = str(h.dtype)
        self._x[li] = h[0]

    def _norm2_pre(self, li, module, args, kwargs):
        r = args[0] if args else kwargs["hidden_states"]
        if li in self._r:
            raise RuntimeError(f"post_attention_layernorm of layer {li} called twice")
        self._r[li] = r[0]

    def _mlp_post(self, li, module, args, out):
        if li in self._m:
            raise RuntimeError(f"mlp of layer {li} called twice")
        self._m[li] = out[0]

    def _layer_post(self, li, module, args, out):
        x_next = (out[0] if isinstance(out, (tuple, list)) else out)[0]
        x, r, m = self._x.pop(li), self._r.pop(li), self._m.pop(li)
        rf, mf, xf, xnf = r.float(), m.float(), x.float(), x_next.float()
        recon = rf + mf
        diff = (xnf - recon).abs().amax(dim=-1)                       # (T,)
        self.abs_err[li] = float(diff.max())
        self.rel_err[li] = float((diff / (xnf.abs().amax(dim=-1) + self.eps)).max())
        if self.rel_tol is not None and self.rel_err[li] > self.rel_tol:
            raise RuntimeError(f"layer {li}: x_next != r + m (rel err {self.rel_err[li]:.3e} > {self.rel_tol})")
        self.mlp_norm[li] = mf.norm(dim=-1).cpu()
        self.res_norm[li] = rf.norm(dim=-1).cpu()
        self.hidden_rel[li] = ((xnf - xf).norm(dim=-1) / (xf.norm(dim=-1) + self.eps)).cpu()
        self.hidden_cos[li] = (1.0 - torch.nn.functional.cosine_similarity(xf, xnf, dim=-1)).cpu()
        if self.raw is not None:
            self.raw[li] = {"x": xf.cpu(), "r": rf.cpu(), "m": mf.cpu(), "x_next": xnf.cpu()}

    # ---- context manager -----------------------------------------------------------------
    def __enter__(self):
        try:
            for li, layer in enumerate(self.layers):
                self.handles.append(layer.register_forward_pre_hook(
                    lambda mod, a, kw, li=li: self._layer_pre(li, mod, a, kw), with_kwargs=True))
                self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                    lambda mod, a, kw, li=li: self._norm2_pre(li, mod, a, kw), with_kwargs=True))
                self.handles.append(layer.mlp.register_forward_hook(
                    lambda mod, a, o, li=li: self._mlp_post(li, mod, a, o)))
                self.handles.append(layer.register_forward_hook(
                    lambda mod, a, o, li=li: self._layer_post(li, mod, a, o)))
        except Exception:
            self._remove()
            raise
        return self

    def _remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self._x.clear(); self._r.clear(); self._m.clear()

    def __exit__(self, *exc):
        self._remove()
        return False

    # ---- result --------------------------------------------------------------------------
    def result(self) -> MLPDynamics:
        L = len(self.layers)
        if sorted(self.calls) != list(range(L)) or len(self.mlp_norm) != L:
            raise RuntimeError(f"incomplete collection: {len(self.mlp_norm)}/{L} layers observed")
        mlp = torch.stack([self.mlp_norm[l] for l in range(L)])
        res = torch.stack([self.res_norm[l] for l in range(L)])
        R = mlp / (res + self.eps)
        if L < 2:
            raise ValueError("D requires at least two decoder layers")
        D = (R[1:] - R[:-1]).abs()
        stats = MLPDynamics(
            mlp_norm=mlp, residual_norm=res, R=R, D=D,
            hidden_rel=torch.stack([self.hidden_rel[l] for l in range(L)]),
            hidden_cos=torch.stack([self.hidden_cos[l] for l in range(L)]),
            residual_max_abs_err=torch.tensor([self.abs_err[l] for l in range(L)]),
            residual_max_rel_err=torch.tensor([self.rel_err[l] for l in range(L)]),
            eps=self.eps, n_layers=L, n_tokens=int(mlp.shape[1]), dtype=self.dtype or "")
        for t in stats.as_dict().values():
            if not torch.isfinite(t).all():
                raise RuntimeError("nonfinite MLP dynamics statistic")
        if self.raw is not None:
            stats.extra["raw"] = self.raw
        return stats


def d_topk_mean(D: torch.Tensor, k: int) -> torch.Tensor:
    """token 별 D 상위 min(k, L-1) 평균. D: (L-1, T) → (T,)"""
    if D.ndim != 2 or D.shape[0] < 1:
        raise ValueError("D must be (L-1, T) with L >= 2")
    k = min(int(k), D.shape[0])
    return D.topk(k, dim=0).values.mean(0)


def token_table(stats: MLPDynamics, input_ids: torch.Tensor, tokenizer=None, special_ids=()):
    """token 표: index, id, 읽을 수 있는 조각, special 여부, MLP norm 평균, R 평균/최대, D 평균/최대/top-3, R 표준편차."""
    ids = input_ids[0].tolist() if input_ids.ndim == 2 else input_ids.tolist()
    if len(ids) != stats.n_tokens:
        raise ValueError("input ids and statistics disagree on token count")
    special = set(int(s) for s in special_ids)
    top3 = d_topk_mean(stats.D, 3)
    rows = []
    for i, tid in enumerate(ids):
        piece = tokenizer.convert_ids_to_tokens(tid) if tokenizer is not None else ""
        rows.append({"index": i, "token_id": int(tid), "piece": piece, "special": int(tid) in special,
                     "mlp_norm_mean": float(stats.mlp_norm[:, i].mean()),
                     "R_mean": float(stats.R[:, i].mean()), "R_max": float(stats.R[:, i].max()),
                     "R_std": float(stats.R[:, i].std(unbiased=False)),
                     "D_mean": float(stats.D[:, i].mean()), "D_max": float(stats.D[:, i].max()),
                     "D_top3_mean": float(top3[i])})
    return rows
