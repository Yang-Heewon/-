"""Byte-matched visual KV compression baselines.

The quality path uses fake quantization and mask-based eviction so it can run on
Qwen2.5-VL/eager attention on V100.  It measures numerical/task degradation, not
physical memory or latency.  Physical-cache measurements are a separate M6 gate.

Algorithm provenance:
* KIVI-style asymmetric quantization (key per-channel, value per-token):
  https://github.com/jy-yuan/KIVI (MIT), pinned in ``third_party/KIVI``.
* Merge-on-evict follows NVIDIA kvpress ``MergingPress``:
  https://github.com/NVIDIA/kvpress (Apache-2.0), pinned in
  ``third_party/kvpress``.  This file is an adaptation for visual-token-only,
  mask-based evaluation and is therefore not the upstream runtime/kernel.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import math
from typing import Iterable, Optional

import torch


@dataclass(frozen=True)
class KVShape:
    """Logical visual KV shape used for storage accounting."""

    layers: int
    batch: int
    kv_heads: int
    tokens: int
    head_dim: int
    position_dims: int = 3  # Qwen2.5-VL mRoPE: temporal, height, width

    def with_tokens(self, tokens: int) -> "KVShape":
        return KVShape(
            layers=self.layers,
            batch=self.batch,
            kv_heads=self.kv_heads,
            tokens=tokens,
            head_dim=self.head_dim,
            position_dims=self.position_dims,
        )


@dataclass(frozen=True)
class StorageEstimate:
    payload_bytes: int
    metadata_bytes: int
    position_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes + self.position_bytes


def shape_from_config(config, visual_tokens: int, batch: int = 1) -> KVShape:
    """Build a storage shape from a Qwen-style multimodal config."""
    text = getattr(config, "text_config", config)
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    return KVShape(
        layers=int(text.num_hidden_layers),
        batch=batch,
        kv_heads=int(text.num_key_value_heads),
        tokens=int(visual_tokens),
        head_dim=int(head_dim),
    )


def _position_bytes(shape: KVShape, tokens: Optional[int] = None) -> int:
    # int32 is sufficient for the offsets used in these experiments.
    n = shape.tokens if tokens is None else tokens
    return shape.batch * n * shape.position_dims * 4


def dense_storage(shape: KVShape, element_bits: int = 16) -> StorageEstimate:
    values = shape.layers * shape.batch * shape.kv_heads * shape.tokens * shape.head_dim * 2
    return StorageEstimate(
        payload_bytes=math.ceil(values * element_bits / 8),
        metadata_bytes=0,
        position_bytes=_position_bytes(shape),
    )


def sparse_storage(
    shape: KVShape,
    keep_tokens: int,
    element_bits: int = 16,
    index_bytes: int = 4,
    shared_indices: bool = True,
) -> StorageEstimate:
    """Storage for hard eviction or merge-on-evict.

    The current VLM adapter uses one visual keep set shared across layers/heads.
    Set ``shared_indices=False`` for head/layer-specific physical eviction.
    """
    if not 1 <= keep_tokens <= shape.tokens:
        raise ValueError(f"keep_tokens must be in [1, {shape.tokens}], got {keep_tokens}")
    kept = shape.with_tokens(keep_tokens)
    payload = dense_storage(kept, element_bits=element_bits).payload_bytes
    copies = 1 if shared_indices else shape.layers * shape.batch * shape.kv_heads
    metadata = keep_tokens * index_bytes * copies
    return StorageEstimate(payload, metadata, _position_bytes(kept))


def quantized_storage(
    shape: KVShape,
    nbits: int,
    key_group_size: int = 64,
    value_group_size: int = 64,
    scale_bytes: int = 2,
    zero_bytes: int = 2,
) -> StorageEstimate:
    """KIVI-style group quantization storage including affine metadata."""
    if nbits not in (2, 3, 4, 8):
        raise ValueError("nbits must be one of 2, 3, 4, 8")
    per_tensor = shape.layers * shape.batch * shape.kv_heads * shape.tokens * shape.head_dim
    payload = math.ceil((2 * per_tensor * nbits) / 8)

    key_groups = (
        shape.layers
        * shape.batch
        * shape.kv_heads
        * shape.head_dim
        * math.ceil(shape.tokens / key_group_size)
    )
    value_groups = (
        shape.layers
        * shape.batch
        * shape.kv_heads
        * shape.tokens
        * math.ceil(shape.head_dim / value_group_size)
    )
    metadata = (key_groups + value_groups) * (scale_bytes + zero_bytes)
    return StorageEstimate(payload, metadata, _position_bytes(shape))


def hybrid_storage(
    shape: KVShape,
    keep_tokens: int,
    nbits: int,
    key_group_size: int = 64,
    value_group_size: int = 64,
    index_bytes: int = 4,
) -> StorageEstimate:
    kept = shape.with_tokens(keep_tokens)
    quant = quantized_storage(
        kept,
        nbits=nbits,
        key_group_size=key_group_size,
        value_group_size=value_group_size,
    )
    return StorageEstimate(
        payload_bytes=quant.payload_bytes,
        metadata_bytes=quant.metadata_bytes + keep_tokens * index_bytes,
        position_bytes=quant.position_bytes,
    )


def max_keep_for_budget(
    shape: KVShape,
    target_bytes: int,
    kind: str,
    nbits: int = 4,
    key_group_size: int = 64,
    value_group_size: int = 64,
) -> int:
    """Largest token count fitting a physical byte budget."""
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    lo, hi, best = 1, shape.tokens, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if kind in ("sparse", "merge"):
            used = sparse_storage(shape, mid).total_bytes
        elif kind == "hybrid":
            used = hybrid_storage(
                shape,
                mid,
                nbits=nbits,
                key_group_size=key_group_size,
                value_group_size=value_group_size,
            ).total_bytes
        else:
            raise ValueError(f"unknown kind: {kind}")
        if used <= target_bytes:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return max(1, best)


def _groupwise_affine_fake_quantize_last(x: torch.Tensor, nbits: int, group_size: int) -> torch.Tensor:
    """Affine quantize/dequantize groups along the final dimension."""
    if nbits not in (2, 3, 4, 8):
        raise ValueError("nbits must be one of 2, 3, 4, 8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    dtype = x.dtype
    n = x.shape[-1]
    pad = (-n) % group_size
    work = x.float()
    if pad:
        work = torch.nn.functional.pad(work, (0, pad))
    grouped = work.reshape(*work.shape[:-1], -1, group_size)
    xmin = grouped.amin(dim=-1, keepdim=True)
    xmax = grouped.amax(dim=-1, keepdim=True)
    qmax = (1 << nbits) - 1
    scale = (xmax - xmin) / qmax
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round((grouped - xmin) / safe_scale).clamp_(0, qmax)
    dequant = q * safe_scale + xmin
    dequant = torch.where(scale > 0, dequant, xmin)
    return dequant.reshape(*work.shape[:-1], -1)[..., :n].to(dtype)


def fake_quantize_keys(keys: torch.Tensor, nbits: int = 4, group_size: int = 64) -> torch.Tensor:
    """KIVI-style per-channel keys: group along the token dimension."""
    if keys.ndim != 4:
        raise ValueError("keys must have shape (batch, kv_heads, tokens, head_dim)")
    transposed = keys.permute(0, 1, 3, 2)
    return _groupwise_affine_fake_quantize_last(transposed, nbits, group_size).permute(0, 1, 3, 2)


def fake_quantize_values(values: torch.Tensor, nbits: int = 4, group_size: int = 64) -> torch.Tensor:
    """KIVI-style per-token values: group along the head dimension."""
    if values.ndim != 4:
        raise ValueError("values must have shape (batch, kv_heads, tokens, head_dim)")
    return _groupwise_affine_fake_quantize_last(values, nbits, group_size)


def merge_evicted_into_kept(
    keys: torch.Tensor,
    values: torch.Tensor,
    keep_indices: torch.Tensor,
    similarity_threshold: float = 0.0,
    merge_fraction: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge-on-evict adapted from NVIDIA kvpress ``MergingPress``.

    Evicted values are blended into their most cosine-similar surviving key.
    Keys remain unchanged.  Returned tensors contain only kept positions.
    """
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("keys and values must share shape (B, H, T, D)")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [0, 1]")
    if not 0.0 < merge_fraction <= 1.0:
        raise ValueError("merge_fraction must be in (0, 1]")

    bsz, heads, seq_len, head_dim = keys.shape
    keep_indices = torch.as_tensor(keep_indices, device=keys.device, dtype=torch.long).flatten()
    keep_indices = torch.unique(keep_indices, sorted=True)
    if keep_indices.numel() == 0 or keep_indices.min() < 0 or keep_indices.max() >= seq_len:
        raise ValueError("keep_indices must be non-empty and inside the token range")
    n_kept = keep_indices.numel()
    expanded_keep = keep_indices.view(1, 1, -1).expand(bsz, heads, -1)
    gather_keep = expanded_keep.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kept_keys = keys.gather(2, gather_keep)
    kept_values = values.gather(2, gather_keep)
    if n_kept == seq_len:
        return kept_keys, kept_values

    evict_mask = torch.ones(seq_len, device=keys.device, dtype=torch.bool)
    evict_mask[keep_indices] = False
    evict_indices = evict_mask.nonzero(as_tuple=False).flatten()
    expanded_evict = evict_indices.view(1, 1, -1).expand(bsz, heads, -1)
    gather_evict = expanded_evict.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    evict_keys = keys.gather(2, gather_evict)
    evict_values = values.gather(2, gather_evict)

    eps = 1e-6
    kept_normed = kept_keys.float() / kept_keys.float().norm(dim=-1, keepdim=True).clamp(min=eps)
    evict_normed = evict_keys.float() / evict_keys.float().norm(dim=-1, keepdim=True).clamp(min=eps)
    max_sim, target = (evict_normed @ kept_normed.transpose(-2, -1)).max(dim=-1)
    merge_ok = max_sim >= similarity_threshold
    if merge_fraction < 1.0 and merge_ok.any():
        cutoff = max_sim.masked_fill(~merge_ok, float("-inf")).quantile(
            1.0 - merge_fraction, dim=-1, keepdim=True
        )
        merge_ok &= max_sim >= cutoff

    weights = max_sim.clamp(min=0) * merge_ok.float()
    target_norm = kept_values.float().norm(dim=-1).gather(2, target)
    evict_norm = evict_values.float().norm(dim=-1)
    weights = weights * evict_norm / (evict_norm + target_norm + eps)

    target_expanded = target.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    value_accum = torch.zeros_like(kept_values, dtype=torch.float32)
    value_accum.scatter_add_(2, target_expanded, weights.unsqueeze(-1) * evict_values.float())
    weight_accum = torch.zeros(bsz, heads, n_kept, device=keys.device, dtype=torch.float32)
    weight_accum.scatter_add_(2, target, weights)
    merged = torch.where(
        (weight_accum > 0).unsqueeze(-1),
        (kept_values.float() + value_accum) / (1.0 + weight_accum).unsqueeze(-1),
        kept_values.float(),
    ).to(values.dtype)
    return kept_keys, merged


class VisualKVTransform(AbstractContextManager):
    """Apply fake quantization and/or merge to visual K/V projections.

    This context manager modifies only the language-model K/V projection outputs
    at ``visual_positions``.  Eviction itself is still supplied by the caller's
    4D attention mask.  Thus it is a quality baseline, not a physical cache.
    """

    def __init__(
        self,
        model,
        visual_positions: Iterable[int] | torch.Tensor,
        *,
        keep_indices: Optional[Iterable[int] | torch.Tensor] = None,
        nbits: Optional[int] = None,
        key_group_size: int = 64,
        value_group_size: int = 64,
        merge: bool = False,
        similarity_threshold: float = 0.0,
        merge_fraction: float = 1.0,
    ):
        self.model = model
        self.visual_positions = torch.as_tensor(list(visual_positions), dtype=torch.long)
        self.keep_indices = None
        if keep_indices is not None:
            # merge_evicted_into_kept returns values in sorted keep-index order.
            # Canonicalize here too so the hook writes those values back to the
            # corresponding sorted visual positions for every caller.
            raw_keep = torch.as_tensor(list(keep_indices), dtype=torch.long)
            self.keep_indices = torch.unique(raw_keep.flatten(), sorted=True)
        self.nbits = nbits
        self.key_group_size = key_group_size
        self.value_group_size = value_group_size
        self.merge = merge
        self.similarity_threshold = similarity_threshold
        self.merge_fraction = merge_fraction
        self._hooks = []
        self._keys: dict[int, torch.Tensor] = {}

        if self.visual_positions.numel() == 0:
            raise ValueError("visual_positions cannot be empty")
        if merge and self.keep_indices is None:
            raise ValueError("merge=True requires keep_indices")
        if nbits is None and not merge:
            raise ValueError("at least one transform must be enabled")

    @staticmethod
    def _as_bhtd(output: torch.Tensor, attn) -> torch.Tensor:
        bsz, seq_len, _ = output.shape
        return output.view(bsz, seq_len, attn.config.num_key_value_heads, attn.head_dim).transpose(1, 2)

    @staticmethod
    def _as_projection(states: torch.Tensor) -> torch.Tensor:
        return states.transpose(1, 2).reshape(states.shape[0], states.shape[2], -1)

    def _key_hook(self, layer_idx: int, attn):
        def hook(_module, _inputs, output):
            states = self._as_bhtd(output, attn)
            pos = self.visual_positions.to(states.device)
            visual = states.index_select(2, pos)
            if self.nbits is not None:
                visual = fake_quantize_keys(visual, self.nbits, self.key_group_size)
                states = states.clone()
                states.index_copy_(2, pos, visual)
            self._keys[layer_idx] = visual
            return self._as_projection(states)

        return hook

    def _value_hook(self, layer_idx: int, attn):
        def hook(_module, _inputs, output):
            states = self._as_bhtd(output, attn)
            pos = self.visual_positions.to(states.device)
            visual = states.index_select(2, pos)
            states = states.clone()
            if self.merge:
                keep = self.keep_indices.to(states.device)
                _, visual = merge_evicted_into_kept(
                    self._keys[layer_idx],
                    visual,
                    keep,
                    similarity_threshold=self.similarity_threshold,
                    merge_fraction=self.merge_fraction,
                )
                if self.nbits is not None:
                    visual = fake_quantize_values(visual, self.nbits, self.value_group_size)
                states.index_copy_(2, pos.index_select(0, keep), visual)
            else:
                visual = fake_quantize_values(visual, self.nbits, self.value_group_size)
                states.index_copy_(2, pos, visual)
            return self._as_projection(states)

        return hook

    def __enter__(self):
        language_model = (
            self.model.model.language_model
            if hasattr(self.model.model, "language_model")
            else self.model.model
        )
        for layer_idx, layer in enumerate(language_model.layers):
            attn = layer.self_attn
            self._hooks.append(attn.k_proj.register_forward_hook(self._key_hook(layer_idx, attn)))
            self._hooks.append(attn.v_proj.register_forward_hook(self._value_hook(layer_idx, attn)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._keys.clear()
        return False
