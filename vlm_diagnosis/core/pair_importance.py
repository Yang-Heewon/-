"""Global recurrent importance for layer/head/token KV pairs.

``PairImportance`` keeps the selection unit explicit: one cache column in one
decoder layer and one KV head.  It delegates score normalization and recurrent
updates to :class:`MultimodalImportance`, while owning the immutable mapping
from each physical score slot to ``(layer, head, logical_token)``.  Selection
uses one global pair budget; it intentionally imposes no per-layer, per-head,
per-token, or per-modality quota.
"""
from __future__ import annotations

import math
import numbers
from typing import Any

import torch

from .recurrent_importance import MultimodalImportance


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _modality_vector(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.long:
        raise ValueError(f"{name} must be a 1D int64 tensor")
    return value.detach().cpu().clone()


def _score_tensor(
    value: Any,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype == torch.bool or value.is_complex():
        raise ValueError(f"{name} must contain real nonnegative scores")
    try:
        result = value.detach().to(device="cpu", dtype=torch.float32).clone()
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"{name} must contain numeric scores") from exc
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError(f"{name} must contain finite nonnegative scores")
    return result


def _mask(value: Any, length: int, name: str = "keep") -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a 1D bool tensor")
    if value.numel() != length:
        raise ValueError(f"{name} length mismatch: expected {length}, got {value.numel()}")
    return value.detach().cpu().clone()


class PairImportance:
    """Recurrent state for a globally budgeted ragged KV cache.

    Initial priors have shape ``(layers, kv_heads, tokens)`` and flatten in
    layer-major, then head-major, then chronological token order.  Sink tokens
    are protected independently in every layer/head group and consume the same
    global pair budget as every other entry.
    """

    def __init__(
        self,
        pair_prior_scores: torch.Tensor,
        modality_ids: torch.Tensor,
        budget_pairs: int,
        n_sink: int = 4,
        prior_floor: float = 0.35,
        decay: float = 0.9,
        modality_names: dict[int, str] | None = None,
    ) -> None:
        if not isinstance(pair_prior_scores, torch.Tensor) or pair_prior_scores.ndim != 3:
            raise ValueError("pair_prior_scores must be a (layers, heads, tokens) tensor")
        n_layers, n_heads, n_tokens = map(int, pair_prior_scores.shape)
        if n_layers < 1 or n_heads < 1:
            raise ValueError("pair importance requires at least one layer and one KV head")
        scores = _score_tensor(
            pair_prior_scores,
            "pair_prior_scores",
            (n_layers, n_heads, n_tokens),
        )
        token_modalities = _modality_vector(modality_ids, "modality_ids")
        if token_modalities.numel() != n_tokens:
            raise ValueError(
                "modality_ids must align with the token axis of pair_prior_scores"
            )
        sink_count = _nonnegative_integer(n_sink, "n_sink")

        groups = n_layers * n_heads
        group_ids = torch.arange(groups, dtype=torch.long).repeat_interleave(n_tokens)
        token_ids = torch.arange(n_tokens, dtype=torch.long).repeat(groups)
        pair_modalities = token_modalities.repeat(groups)
        protected = token_ids < sink_count

        self.engine = MultimodalImportance(
            prior_scores=scores.reshape(-1),
            modality_ids=pair_modalities,
            budget=budget_pairs,
            protected=protected,
            prior_floor=prior_floor,
            decay=decay,
            modality_names=modality_names,
        )
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._n_sink = sink_count
        self._group_ids = group_ids
        self._token_ids = token_ids
        # This high-water mark is independent of retained IDs: physical
        # deletion must never make an old logical token number reusable.
        self._next_token_floor = n_tokens

    @property
    def groups(self) -> int:
        return self._n_layers * self._n_heads

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def n_heads(self) -> int:
        return self._n_heads

    @property
    def n_pairs(self) -> int:
        return int(self._group_ids.numel())

    @property
    def group_ids(self) -> torch.Tensor:
        return self._group_ids.clone()

    @property
    def token_ids(self) -> torch.Tensor:
        return self._token_ids.clone()

    @property
    def nbytes(self) -> int:
        index_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._group_ids, self._token_ids)
        )
        return int(self.engine._state_bytes() + index_bytes)

    def _validate_selection(self, keep: torch.Tensor) -> torch.Tensor:
        keep_cpu = _mask(keep, self.n_pairs)
        selected = int(keep_cpu.sum())
        if selected != self.engine.budget:
            raise ValueError(
                f"keep must select exactly budget_pairs ({self.engine.budget}), "
                f"got {selected}"
            )
        protected = self._token_ids < self._n_sink
        if bool((protected & ~keep_cpu).any()):
            raise ValueError("keep must retain every protected sink pair")
        return keep_cpu

    def snapshot(self, keep: torch.Tensor | None = None) -> dict[str, Any]:
        """Return JSON-safe pair topology and exact persistent state bytes."""
        counted = (
            torch.ones(self.n_pairs, dtype=torch.bool)
            if keep is None
            else _mask(keep, self.n_pairs)
        )
        group_counts = [
            int((counted & (self._group_ids == group)).sum())
            for group in range(self.groups)
        ]
        layer_head_counts = [
            group_counts[layer * self._n_heads : (layer + 1) * self._n_heads]
            for layer in range(self._n_layers)
        ]
        layer_counts = [sum(row) for row in layer_head_counts]
        head_counts = [
            sum(layer_head_counts[layer][head] for layer in range(self._n_layers))
            for head in range(self._n_heads)
        ]
        selected_tokens = self._token_ids[counted]
        modalities = self.engine.modality_ids
        names = self.engine.modality_names
        modality_counts = {
            names[modality_id]: int(
                (counted & (modalities == modality_id)).sum()
            )
            for modality_id in sorted(names)
        }
        return {
            "n_layers": self._n_layers,
            "n_heads": self._n_heads,
            "groups": self.groups,
            "resident_pairs": self.n_pairs,
            "counted_pairs": int(counted.sum()),
            "budget_pairs": self.engine.budget,
            "pairs_by_group": group_counts,
            "pairs_by_layer_head": layer_head_counts,
            "pairs_by_layer": layer_counts,
            "pairs_by_head": head_counts,
            "distinct_logical_tokens": int(selected_tokens.unique().numel()),
            "modality_pair_counts": modality_counts,
            "state_bytes": self.nbytes,
        }

    def select(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Globally select exactly ``budget_pairs`` entries."""
        keep, diagnostics = self.engine.select()
        result = dict(diagnostics)
        result["importance_state_bytes"] = result["state_bytes"]
        result.update(self.snapshot(keep))
        return keep, result

    def retain(self, keep: torch.Tensor) -> None:
        """Physically delete unselected score and pair-mapping state."""
        keep_cpu = self._validate_selection(keep)
        positions = keep_cpu.nonzero(as_tuple=True)[0]
        compact_group_ids = self._group_ids.index_select(0, positions).clone()
        compact_token_ids = self._token_ids.index_select(0, positions).clone()
        self.engine.retain(keep_cpu)
        self._group_ids = compact_group_ids
        self._token_ids = compact_token_ids

    def append(
        self,
        new_modality_ids: torch.Tensor,
        start_token: int,
        prior_scores: torch.Tensor | None = None,
    ) -> None:
        """Append new logical tokens to every layer/head pair group.

        ``start_token`` may leave a gap, but may never overlap any token range
        previously appended, including ranges that have since been deleted.
        """
        modalities = _modality_vector(new_modality_ids, "new_modality_ids")
        token_count = int(modalities.numel())
        start = _nonnegative_integer(start_token, "start_token")
        if start < self._next_token_floor:
            raise ValueError(
                "start_token overlaps a logical token range seen earlier in this session"
            )
        unknown = sorted(set(modalities.tolist()) - set(self.engine.modality_names))
        if unknown:
            raise ValueError(
                f"new_modality_ids contains IDs missing from modality_names: {unknown}"
            )
        shape = (self._n_layers, self._n_heads, token_count)
        raw_prior = (
            torch.zeros(shape, dtype=torch.float32)
            if prior_scores is None
            else _score_tensor(prior_scores, "prior_scores", shape)
        )
        if self.engine.prior_scale == 0.0 and bool((raw_prior > 0).any()):
            raise ValueError(
                "cannot append positive prior_scores when initial prior scale is zero"
            )
        if token_count == 0:
            return

        new_group_ids = torch.arange(self.groups, dtype=torch.long).repeat_interleave(
            token_count
        )
        logical = torch.arange(start, start + token_count, dtype=torch.long)
        new_token_ids = logical.repeat(self.groups)
        pair_modalities = modalities.repeat(self.groups)
        new_protected = new_token_ids < self._n_sink
        protected_count = int(self.engine._protected.sum()) + int(new_protected.sum())
        if protected_count > self.engine.budget:
            raise ValueError(
                f"protected sink pairs ({protected_count}) exceed budget_pairs "
                f"({self.engine.budget})"
            )

        # Allocate mapping outputs before the engine commit so all validation
        # and potentially fallible shape work precede persistent mutation.
        appended_group_ids = torch.cat((self._group_ids, new_group_ids))
        appended_token_ids = torch.cat((self._token_ids, new_token_ids))
        self.engine.append(
            self.groups * token_count,
            modality_ids=pair_modalities,
            prior_scores=raw_prior.reshape(-1),
        )
        if bool(new_protected.any()):
            self.engine._protected[-new_protected.numel():] = new_protected
        self._group_ids = appended_group_ids
        self._token_ids = appended_token_ids
        self._next_token_floor = start + token_count

    def observe(self, masses: list[torch.Tensor]) -> dict[str, Any]:
        """Update every resident pair from per-group attention mass vectors."""
        if not isinstance(masses, (list, tuple)) or len(masses) != self.groups:
            raise ValueError(
                f"masses must contain one vector for each of {self.groups} groups"
            )
        attention = torch.zeros(self.n_pairs, dtype=torch.float32)
        observed = torch.ones(self.n_pairs, dtype=torch.bool)
        for group, mass in enumerate(masses):
            positions = (self._group_ids == group).nonzero(as_tuple=True)[0]
            expected = int(positions.numel())
            vector = _score_tensor(mass, f"masses[{group}]", (expected,))
            tokens = self._token_ids.index_select(0, positions)
            if tokens.numel() > 1 and not bool((tokens[1:] > tokens[:-1]).all()):
                raise RuntimeError("pair mappings must remain chronological within each group")
            attention[positions] = vector

        diagnostics = self.engine.update(attention, observed)
        keep, _ = self.engine.select()
        result = dict(diagnostics)
        result["importance_state_bytes"] = result["state_bytes"]
        result.update(self.snapshot(keep))
        return result

    def selected_ids(self, keep: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Map a global selection to chronological token IDs per cache group."""
        if keep is None:
            keep = self.engine.select()[0]
        keep_cpu = self._validate_selection(keep)
        selected: list[torch.Tensor] = []
        for group in range(self.groups):
            ids = self._token_ids[keep_cpu & (self._group_ids == group)]
            ids = torch.sort(ids).values.to(dtype=torch.long, device="cpu").clone()
            selected.append(ids)
        return selected

