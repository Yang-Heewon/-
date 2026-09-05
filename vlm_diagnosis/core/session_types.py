"""Owned, token-aligned data at the modality adapter / cache engine boundary.

These types describe one shared decoder cache, not interchangeable K/V from
different models. Features are small per-token metadata (coordinates, time,
etc.), never a hidden raw-media or evicted-KV reservoir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch


DEFAULT_MODALITY_NAMES = {
    0: "control", 1: "text", 2: "image", 3: "audio", 4: "video", 5: "sensor",
}


def validate_modalities(ids, length, names=None):
    names = dict(DEFAULT_MODALITY_NAMES if names is None else names)
    if not names or any(isinstance(k, bool) or not isinstance(k, int) or k < 0
                        or not isinstance(v, str) or not v for k, v in names.items()):
        raise ValueError("modality_names must map nonnegative integer IDs to names")
    if len(set(names.values())) != len(names):
        raise ValueError("modality names must be unique")
    ids = torch.as_tensor(ids)
    if ids.ndim != 1 or ids.numel() != length or ids.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("modality_ids must be a token-aligned 1D integer tensor")
    if not set(ids.tolist()) <= set(names):
        raise ValueError("unknown modality ID")
    return ids.detach().to(device="cpu", dtype=torch.long).clone(), names


def owned_features(features, length):
    result = {}
    for name, tensor in features.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise ValueError("token features must map names to tensors")
        if tensor.ndim < 1 or tensor.shape[0] != length or tensor.layout != torch.strided:
            raise ValueError(f"feature {name} must have token count as its leading dimension")
        result[name] = tensor.detach().cpu().clone()
    return result


def _ids(ids):
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
            or ids.shape[1] < 1 or ids.dtype != torch.long):
        raise ValueError("input_ids must be a nonempty batch-one int64 tensor")
    return ids.detach().cpu().clone()


def _prior(scores, length):
    scores = torch.as_tensor(scores)
    if (scores.ndim != 1 or scores.numel() != length or scores.dtype == torch.bool
            or scores.is_complex() or not torch.isfinite(scores).all() or (scores < 0).any()):
        raise ValueError("prior_scores must be finite nonnegative token-aligned scores")
    converted = scores.detach().to(device="cpu", dtype=torch.float32).clone()
    if not torch.isfinite(converted).all():
        raise ValueError("prior_scores exceed float32 range")
    return converted


@dataclass
class SessionSeed:
    kv: tuple
    prefix_ids: torch.Tensor
    prior_scores: torch.Tensor
    modality_ids: torch.Tensor
    next_position: int
    prefill_seconds: float
    modality_names: dict[int, str] = field(default_factory=lambda: dict(DEFAULT_MODALITY_NAMES))
    token_features: dict[str, torch.Tensor] = field(default_factory=dict)
    adapter_id: str = "qwen_image_v1"
    pair_prior_scores: torch.Tensor | None = None

    def __post_init__(self):
        self.prefix_ids = _ids(self.prefix_ids)
        n = self.prefix_ids.shape[1]
        self.modality_ids, self.modality_names = validate_modalities(self.modality_ids, n, self.modality_names)
        self.prior_scores = _prior(self.prior_scores, n)
        self.token_features = owned_features(self.token_features, n)
        if not self.kv or any(k.ndim != 4 or k.shape != v.shape or k.shape[0] != 1
                              or k.shape[-2] != n for k, v in self.kv):
            raise ValueError("seed requires aligned batch-one K/V at every decoder layer")
        if self.pair_prior_scores is not None:
            expected = (len(self.kv), self.kv[0][0].shape[1], n)
            if tuple(self.pair_prior_scores.shape) != expected:
                raise ValueError("pair_prior_scores must align with layer, KV head, and token")
            self.pair_prior_scores = _prior(self.pair_prior_scores.reshape(-1),
                                            len(self.kv)*self.kv[0][0].shape[1]*n).reshape(expected)
        if isinstance(self.next_position, bool) or not isinstance(self.next_position, int) or self.next_position < 0:
            raise ValueError("next_position must be a nonnegative logical position")
        if (isinstance(self.prefill_seconds, bool) or not isinstance(self.prefill_seconds, (int, float))
                or not math.isfinite(self.prefill_seconds) or self.prefill_seconds < 0):
            raise ValueError("prefill_seconds must be finite and nonnegative")
        if not isinstance(self.adapter_id, str) or not self.adapter_id:
            raise ValueError("seed requires an adapter_id")

    @property
    def image_mask(self):
        image_ids = [i for i, name in self.modality_names.items() if name == "image"]
        return torch.isin(self.modality_ids, torch.tensor(image_ids, dtype=torch.long))

    @property
    def image_score(self):
        """Legacy image-only API; new code uses prior_scores."""
        return self.prior_scores


@dataclass
class SessionInput:
    input_ids: torch.Tensor
    modality_ids: torch.Tensor
    prior_scores: torch.Tensor | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    position_ids: torch.Tensor | None = None
    next_position: int | None = None
    token_features: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        self.input_ids = _ids(self.input_ids)
        n = self.input_ids.shape[1]
        # IDs are checked against the seed's registry by the session. An
        # adapter may introduce custom modalities through that registry.
        ids = torch.as_tensor(self.modality_ids)
        if ids.ndim != 1 or ids.numel() != n or ids.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("input modality_ids must match input_ids")
        self.modality_ids = ids.detach().to(device="cpu", dtype=torch.long).clone()
        self.prior_scores = _prior(self.prior_scores, n) if self.prior_scores is not None else None
        self.token_features = owned_features(self.token_features, n)
        forbidden = {"past_key_values", "cache_position", "attention_mask", "use_cache",
                     "input_ids", "position_ids", "output_attentions"}
        if forbidden.intersection(self.model_kwargs):
            raise ValueError("model_kwargs cannot override cache/session-owned arguments")
        if self.position_ids is not None:
            if (not isinstance(self.position_ids, torch.Tensor) or self.position_ids.ndim < 1
                    or self.position_ids.shape[-1] != n or self.position_ids.dtype != torch.long):
                raise ValueError("position_ids must align with new tokens")
            self.position_ids = self.position_ids.detach().cpu().clone()
        if self.next_position is not None and (isinstance(self.next_position, bool)
                                               or not isinstance(self.next_position, int)
                                               or self.next_position < 0):
            raise ValueError("invalid next_position")


class TokenFeatures:
    """Token-aligned metadata compacted on exactly the same slots as K/V.

Schemas are fixed by the seed. Missing new-token features receive NaN for
floating tensors, -1 for signed integers and False for bools (unknown).
"""

    def __init__(self, features, length):
        self.tensors = owned_features(features, length)
        if any(t.dtype == torch.uint8 or t.is_complex() for t in self.tensors.values()):
            raise ValueError("features require floating, signed integer, or bool dtype")
        self.length = length

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def validate_append(self, features, length):
        supplied = owned_features(features, length)
        if not set(supplied) <= set(self.tensors):
            raise ValueError("new features must be declared by the seed schema")
        for name, value in supplied.items():
            old = self.tensors[name]
            if value.shape[1:] != old.shape[1:] or value.dtype != old.dtype:
                raise ValueError(f"feature {name} schema changed within the session")
        return supplied

    def append(self, features, length):
        supplied = self.validate_append(features, length)
        appended = {}
        for name, old in self.tensors.items():
            fill = float("nan") if old.is_floating_point() else False if old.dtype == torch.bool else -1
            value = supplied.get(name)
            if value is None:
                value = torch.full((length, *old.shape[1:]), fill, dtype=old.dtype)
            appended[name] = torch.cat((old, value))
        self.tensors = appended
        self.length += length

    def retain(self, positions):
        positions = torch.as_tensor(positions, dtype=torch.long, device="cpu")
        if positions.ndim != 1 or (positions.numel() and (positions.min() < 0 or positions.max() >= self.length)):
            raise ValueError("invalid feature selection")
        self.tensors = {name: t.index_select(0, positions) for name, t in self.tensors.items()}
        self.length = positions.numel()
