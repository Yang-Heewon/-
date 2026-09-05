r"""Training-free multimodal importance for a fixed-budget KV working set.

This module deliberately implements a *heuristic gated state*, not a learned
LSTM.  It consumes only attention from a completed interaction; there is no
API for a future question.  For token ``i`` the state is

.. math::

   p_i &= s_i / \max_j s_j, \qquad h_{i,0}=0,\\
   x_{i,t} &= a_{i,t} / \max_{j\in O_t\setminus P} a_{j,t},\\
   g^\mathrm{in}_{i,t} &= x_{i,t}/(x_{i,t}+h_{i,t-1}),\\
   g^\mathrm{forget}_{i,t} &= d(1-g^\mathrm{in}_{i,t}),\\
   h_{i,t} &= g^\mathrm{forget}_{i,t}h_{i,t-1}
              +g^\mathrm{in}_{i,t}x_{i,t}.

The input gate is zero when its denominator is zero; the forget gate then
equals ``d`` but multiplies a zero old state.  The update is applied only to
the observed, selectable set (observed tokens outside ``P``). Protected set
``P`` is excluded because its items never compete in ranking and a
high-attention sink must not compress the useful score range. An unobserved
cold token keeps its previous
state and is not confused with an observed token receiving zero attention. If
all observed selectable attention is zero, no positive state is fabricated
(existing observed state may still be forgotten by ``d``).

Prior and interaction scores use max normalization over their complete
available candidate domain, never separate per-modality normalization.  This
makes their scale comparable without granting a modality a quota. Consequently,
an unprotected special-token outlier can still set the common scale; callers
should mark a truly mandatory sink as protected rather than rely on a
data-tuned clipping threshold. The selection score after ``T`` completed
interactions with at least one observed selectable token is

.. math::

   w_P(T) &= f_P + (1-f_P)/(T+1),\\
   w_H(T) &= 1-w_P(T),\\
   r_{i,T} &= w_P(T)p_i+w_H(T)h_{i,T}.

Thus history influence grows with observed completed interactions while the
immutable prior always retains weight at least ``prior_floor``. Appended raw
priors use the original fixed global scale and never renormalize old values.
Protected tokens count inside the same exact budget. All persistent tensors
live on CPU, and ``state_bytes`` is their actual tensor-byte count.

``MultimodalImportance`` is the modality-neutral engine. ``RecurrentImportance``
is its image/non-image compatibility wrapper and preserves legacy diagnostic
aliases.  Neither class accepts a future question.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping
from typing import Any

import torch

from .core_delta import _stable_topk
from .session_types import DEFAULT_MODALITY_NAMES


def _one_dimensional(value: Any, name: str) -> torch.Tensor:
    """Convert an input to a detached CPU tensor without hiding shape errors."""
    try:
        tensor = torch.as_tensor(value)
    except Exception as exc:  # make the public validation error consistent
        raise ValueError(f"{name} must be convertible to a 1D tensor") from exc
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}")
    return tensor.detach().cpu().clone()


def _score_vector(value: Any, name: str, length: int | None = None) -> torch.Tensor:
    tensor = _one_dimensional(value, name)
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise ValueError(f"{name} must contain real nonnegative numeric values")
    try:
        tensor = tensor.to(torch.float32)
    except Exception as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if length is not None and tensor.numel() != length:
        raise ValueError(
            f"{name} length mismatch: expected {length}, got {tensor.numel()}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((tensor < 0).any()):
        raise ValueError(f"{name} must contain only nonnegative values")
    return tensor


def _bool_vector(value: Any, name: str, length: int) -> torch.Tensor:
    tensor = _one_dimensional(value, name)
    if tensor.dtype != torch.bool:
        raise ValueError(f"{name} must be a bool tensor")
    if tensor.numel() != length:
        raise ValueError(
            f"{name} length mismatch: expected {length}, got {tensor.numel()}"
        )
    return tensor


def _modality_name_map(value: Mapping[int, str] | None) -> dict[int, str]:
    """Validate and copy the complete ID-to-name vocabulary."""
    if value is None:
        return dict(DEFAULT_MODALITY_NAMES)
    if not isinstance(value, dict):
        raise ValueError("modality_names must be a dict[int, str]")
    result: dict[int, str] = {}
    for key, name in value.items():
        if isinstance(key, bool) or not isinstance(key, numbers.Integral) or int(key) < 0:
            raise ValueError("modality_names keys must be nonnegative integers")
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("modality_names values must be nonempty trimmed strings")
        result[int(key)] = name
    if len(set(result.values())) != len(result):
        raise ValueError("modality_names values must be unique")
    return result


def _modality_vector(
    value: Any,
    name: str,
    length: int,
    modality_names: Mapping[int, str],
) -> torch.Tensor:
    tensor = _one_dimensional(value, name)
    if tensor.dtype != torch.long:
        raise ValueError(f"{name} must be a long tensor")
    if tensor.numel() != length:
        raise ValueError(
            f"{name} length mismatch: expected {length}, got {tensor.numel()}"
        )
    if bool((tensor < 0).any()):
        raise ValueError(f"{name} must contain only nonnegative IDs")
    unknown = sorted(set(tensor.tolist()) - set(modality_names))
    if unknown:
        raise ValueError(f"{name} contains IDs missing from modality_names: {unknown}")
    return tensor


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


class MultimodalImportance:
    """Modality-neutral, interaction-adaptive importance for one session.

    Parameters are copied into private CPU state. ``budget`` is fixed and
    includes every protected entry. The default selector globally ranks all
    selectable tokens, with no modality quota or per-modality normalization.
    A custom selector must be a pure callable accepting keyword arguments
    ``scores``, ``protected``, ``budget``, and ``modality_ids`` and returning
    an exact-budget boolean mask.

    ``modality_names`` is the complete ID vocabulary when supplied. Omitting
    it uses :data:`DEFAULT_MODALITY_NAMES`; an ID outside the chosen vocabulary
    is rejected rather than silently reported as an unknown bucket.
    """

    _policy = "training_free_gated_multimodal_v1"

    def __init__(
        self,
        prior_scores: torch.Tensor,
        modality_ids: torch.Tensor,
        budget: int,
        protected: torch.Tensor | None = None,
        prior_floor: float = 0.35,
        decay: float = 0.9,
        modality_names: dict[int, str] | None = None,
        selector: Callable[..., Any] | None = None,
    ) -> None:
        raw_score = _score_vector(prior_scores, "prior_scores")
        n_items = int(raw_score.numel())
        names = _modality_name_map(modality_names)
        modality_ids_cpu = _modality_vector(
            modality_ids, "modality_ids", n_items, names
        )
        budget_int = _nonnegative_integer(budget, "budget")
        if budget_int > n_items:
            raise ValueError(
                f"budget ({budget_int}) exceeds initial token count ({n_items})"
            )
        if protected is None:
            protected_cpu = torch.zeros(n_items, dtype=torch.bool)
        else:
            protected_cpu = _bool_vector(protected, "protected", n_items)
        n_protected = int(protected_cpu.sum())
        if n_protected > budget_int:
            raise ValueError(
                f"protected tokens ({n_protected}) exceed budget ({budget_int})"
            )

        if selector is not None and not callable(selector):
            raise ValueError("selector must be callable or None")

        self._budget = budget_int
        self._prior_floor = _unit_interval(prior_floor, "prior_floor")
        self._decay = _unit_interval(decay, "decay")
        self._modality_names = names
        self._selector = selector

        max_score = float(raw_score.max()) if n_items else 0.0
        self._prior_scale = max_score
        self._prior_scores = (
            raw_score / max_score if max_score > 0.0 else torch.zeros_like(raw_score)
        )
        self._history_state = torch.zeros(n_items, dtype=torch.float32)
        self._modality_ids = modality_ids_cpu
        self._protected = protected_cpu
        self._ever_observed = torch.zeros(n_items, dtype=torch.bool)

        self._update_calls = 0
        self._observed_turns = 0
        self._last_transition = self._zero_transition()

    @property
    def budget(self) -> int:
        """The immutable token budget, including protected tokens."""
        return self._budget

    @property
    def n_tokens(self) -> int:
        return int(self._prior_scores.numel())

    @property
    def prior_floor(self) -> float:
        return self._prior_floor

    @property
    def prior_scale(self) -> float:
        return self._prior_scale

    @property
    def prior_scores(self) -> torch.Tensor:
        """A copy of normalized priors in the current physical slot space."""
        return self._prior_scores.clone()

    @property
    def modality_ids(self) -> torch.Tensor:
        """A copy of current CPU int64 modality IDs."""
        return self._modality_ids.clone()

    @property
    def modality_names(self) -> dict[int, str]:
        return dict(self._modality_names)

    @property
    def decay(self) -> float:
        return self._decay

    def _zero_transition(self) -> dict[str, int | float]:
        return {
            "retained_count": self._budget,
            "entered_count": 0,
            "evicted_count": 0,
            "turnover_count": 0,
            "symmetric_difference_count": 0,
            "turnover_fraction": 0.0,
        }

    def _weights(self, observed_turns: int | None = None) -> tuple[float, float]:
        turns = self._observed_turns if observed_turns is None else observed_turns
        prior_weight = self._prior_floor + (1.0 - self._prior_floor) / (turns + 1.0)
        return prior_weight, 1.0 - prior_weight

    def _scores(
        self,
        history_state: torch.Tensor | None = None,
        observed_turns: int | None = None,
    ) -> torch.Tensor:
        history = self._history_state if history_state is None else history_state
        prior_weight, history_weight = self._weights(observed_turns)
        return prior_weight * self._prior_scores + history_weight * history

    def _compute_keep(
        self,
        history_state: torch.Tensor | None = None,
        observed_turns: int | None = None,
    ) -> torch.Tensor:
        scores = self._scores(history_state, observed_turns)
        if self._selector is None:
            keep = self._protected.clone()
            remaining = self._budget - int(keep.sum())
            if remaining:
                allowed = ~self._protected
                chosen = _stable_topk(
                    scores.to(torch.float64), remaining, allowed=allowed
                )
                if chosen:
                    keep[torch.tensor(chosen, dtype=torch.long)] = True
        else:
            result = self._selector(
                scores=scores.clone(),
                protected=self._protected.clone(),
                budget=self._budget,
                modality_ids=self._modality_ids.clone(),
            )
            keep = _bool_vector(result, "selector result", self.n_tokens)
        selected = int(keep.sum())
        if selected != self._budget:
            raise ValueError(
                f"selector must select exactly budget ({self._budget}) tokens, "
                f"got {selected}"
            )
        if bool((self._protected & ~keep).any()):
            raise ValueError("selector must retain every protected token")
        return keep

    @staticmethod
    def _transition(before: torch.Tensor, after: torch.Tensor) -> dict[str, int | float]:
        entered = int((after & ~before).sum())
        evicted = int((before & ~after).sum())
        retained = int((before & after).sum())
        budget = int(after.sum())
        return {
            "retained_count": retained,
            "entered_count": entered,
            "evicted_count": evicted,
            # Conventional fixed-working-set turnover counts entering items;
            # symmetric_difference_count is also reported to remove ambiguity.
            "turnover_count": entered,
            "symmetric_difference_count": entered + evicted,
            "turnover_fraction": float(entered / budget) if budget else 0.0,
        }

    def _state_bytes(self) -> int:
        tensors = (
            self._prior_scores,
            self._history_state,
            self._modality_ids,
            self._protected,
            self._ever_observed,
        )
        return int(sum(t.numel() * t.element_size() for t in tensors))

    def _modality_counts(self, mask: torch.Tensor) -> dict[str, int]:
        counts: dict[str, int] = {}
        for modality_id in sorted(self._modality_names):
            name = self._modality_names[modality_id]
            counts[name] = int((mask & (self._modality_ids == modality_id)).sum())
        return counts

    def _named_count(self, mask: torch.Tensor, name: str) -> int:
        modality_id = next(
            (key for key, value in self._modality_names.items() if value == name), None
        )
        if modality_id is None:
            return 0
        return int((mask & (self._modality_ids == modality_id)).sum())

    def _selection_diagnostics(self, keep: torch.Tensor) -> dict[str, Any]:
        prior_weight, history_weight = self._weights()
        selected_image = self._named_count(keep, "image")
        selected_text = self._named_count(keep, "text")
        selected_total = int(keep.sum())
        scores_prior = prior_weight * self._prior_scores
        scores_history = history_weight * self._history_state
        diagnostics: dict[str, Any] = {
            "policy": self._policy,
            "normalization": "global_max_over_observed_unprotected_tokens",
            "tie_break": "lower_token_index",
            "token_count": self.n_tokens,
            "budget": self._budget,
            "kept_count": selected_total,
            "protected_count": int(self._protected.sum()),
            "selected_protected_tokens": int((keep & self._protected).sum()),
            "tokens_by_modality": self._modality_counts(
                torch.ones(self.n_tokens, dtype=torch.bool)
            ),
            "selected_tokens_by_modality": self._modality_counts(keep),
            "selected_text_tokens": selected_text,
            "prior_weight": float(prior_weight),
            "prior_floor": self._prior_floor,
            "prior_scale": self._prior_scale,
            "selected_prior_component_sum": float(scores_prior[keep].sum()),
            # Legacy image-oriented aliases retained for existing logs.
            "selected_image_tokens": selected_image,
            "image_weight": float(prior_weight),
            "history_weight": float(history_weight),
            "image_floor": self._prior_floor,
            "decay": self._decay,
            "image_prior_scale": self._prior_scale,
            "observed_turns": self._observed_turns,
            "update_calls": self._update_calls,
            "ever_observed_tokens": int(self._ever_observed.sum()),
            "history_nonzero_tokens": int((self._history_state > 0).sum()),
            "history_l1": float(self._history_state.sum()),
            "history_max": (
                float(self._history_state.max()) if self.n_tokens else 0.0
            ),
            "selected_image_component_sum": float(scores_prior[keep].sum()),
            "selected_history_component_sum": float(scores_history[keep].sum()),
            "state_bytes": self._state_bytes(),
            "turnover_reference": "selection_immediately_before_last_update",
        }
        diagnostics.update(self._last_transition)
        return diagnostics

    def append(
        self,
        n_tokens: int,
        *,
        modality_ids: torch.Tensor | None = None,
        prior_scores: torch.Tensor | None = None,
    ) -> None:
        """Append unprotected cold positions without changing old priors.

        The default modality is text (ID 1), which must exist in the selected
        vocabulary. Explicit ``prior_scores`` are raw scores normalized by the
        constructor's fixed global scale. If that scale was zero, a positive
        appended prior is undefined and rejected rather than silently changing
        the scale.
        """
        count = _nonnegative_integer(n_tokens, "n_tokens")
        if modality_ids is None:
            appended_modalities = torch.full((count,), 1, dtype=torch.long)
            if count and 1 not in self._modality_names:
                raise ValueError(
                    "default text modality ID 1 is missing from modality_names"
                )
        else:
            appended_modalities = _modality_vector(
                modality_ids, "modality_ids", count, self._modality_names
            )
        raw_append = (
            torch.zeros(count, dtype=torch.float32)
            if prior_scores is None
            else _score_vector(prior_scores, "prior_scores", count)
        )
        if self._prior_scale > 0.0:
            appended_prior = raw_append / self._prior_scale
        else:
            if bool((raw_append > 0).any()):
                raise ValueError(
                    "cannot append positive prior_scores when initial prior scale is zero"
                )
            appended_prior = torch.zeros_like(raw_append)
        if count == 0:
            return

        # Prepare every allocation before committing any state field.
        appended = (
            torch.cat((self._prior_scores, appended_prior)),
            torch.cat((self._history_state, torch.zeros(count, dtype=torch.float32))),
            torch.cat((self._modality_ids, appended_modalities)),
            torch.cat((self._protected, torch.zeros(count, dtype=torch.bool))),
            torch.cat((self._ever_observed, torch.zeros(count, dtype=torch.bool))),
        )
        (self._prior_scores, self._history_state, self._modality_ids,
         self._protected, self._ever_observed) = appended

    def retain(self, keep: torch.Tensor) -> None:
        """Irreversibly compact state to exactly the selected budget slots.

        Every persistent per-token tensor is copied into a new contiguous,
        rebased slot space. Deleted priors and recurrent history are not kept
        in a hidden reservoir and cannot be recalled by this object. The fixed
        budget, interaction schedule, original prior normalization scale, and
        most recent pre-compaction turnover diagnostics remain unchanged.

        ``keep`` must preserve every protected slot and contain exactly
        :attr:`budget` true entries. Validation and all slicing finish before
        object state is committed, so a failed call is atomic.
        """
        keep_cpu = _bool_vector(keep, "keep", self.n_tokens)
        selected = int(keep_cpu.sum())
        if selected != self._budget:
            raise ValueError(
                f"keep must select exactly budget ({self._budget}) tokens, got {selected}"
            )
        if bool((self._protected & ~keep_cpu).any()):
            raise ValueError("keep must retain every protected token")
        indices = keep_cpu.nonzero(as_tuple=True)[0]
        compacted = tuple(
            tensor.index_select(0, indices).clone()
            for tensor in (
                self._prior_scores,
                self._history_state,
                self._modality_ids,
                self._protected,
                self._ever_observed,
            )
        )
        (self._prior_scores, self._history_state, self._modality_ids,
         self._protected, self._ever_observed) = compacted

    def select(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return an exact-budget mask and JSON-serializable diagnostics.

        This method has no side effects.  Turnover describes the most recent
        completed :meth:`update`, whose before/after masks were both computed
        under this same fixed budget.
        """
        keep = self._compute_keep()
        return keep, self._selection_diagnostics(keep)

    def update(
        self,
        attention_mass: torch.Tensor,
        observed: torch.Tensor,
    ) -> dict[str, Any]:
        """Update from one completed interaction and return diagnostics.

        ``attention_mass`` must be finite and nonnegative.  Only unprotected
        positions for which ``observed`` is true participate in global
        normalization or the gated recurrence.  Validation completes before
        any state is mutated.
        """
        n_items = self.n_tokens
        attention = _score_vector(attention_mass, "attention_mass", n_items)
        observed_cpu = _bool_vector(observed, "observed", n_items)

        before_keep = self._compute_keep()
        new_history = self._history_state.clone()
        observed_count = int(observed_cpu.sum())
        update_domain = observed_cpu & ~self._protected
        updated_count = int(update_domain.sum())
        positive = update_domain & (attention > 0)
        positive_count = int(positive.sum())
        evidence_scale = (
            float(attention[update_domain].max()) if updated_count else 0.0
        )

        if updated_count:
            x = torch.zeros(n_items, dtype=torch.float32)
            if evidence_scale > 0.0:
                x[update_domain] = attention[update_domain] / evidence_scale
            old_observed = self._history_state[update_domain]
            x_observed = x[update_domain]
            denominator = x_observed + old_observed
            input_gate = torch.zeros_like(denominator)
            has_support = denominator > 0
            input_gate[has_support] = x_observed[has_support] / denominator[has_support]
            forget_gate = self._decay * (1.0 - input_gate)
            updated = forget_gate * old_observed + input_gate * x_observed
            new_history[update_domain] = updated.clamp_(0.0, 1.0)
            new_observed_turns = self._observed_turns + 1
        else:
            input_gate = torch.empty(0, dtype=torch.float32)
            forget_gate = torch.empty(0, dtype=torch.float32)
            new_observed_turns = self._observed_turns

        # Compute every fallible derived value before committing the new state.
        after_keep = self._compute_keep(new_history, new_observed_turns)
        transition = self._transition(before_keep, after_keep)
        mean_abs_delta = (
            float((new_history - self._history_state).abs().mean())
            if n_items else 0.0
        )

        self._history_state = new_history
        self._ever_observed |= observed_cpu
        self._observed_turns = new_observed_turns
        self._update_calls += 1
        self._last_transition = transition

        diagnostics = self._selection_diagnostics(after_keep)
        diagnostics.update(
            {
                "observed_count": observed_count,
                "observed_fraction": (
                    float(observed_count / n_items) if n_items else 0.0
                ),
                "updated_unprotected_count": updated_count,
                "observed_protected_count": observed_count - updated_count,
                "positive_evidence_count": positive_count,
                "evidence_scale": evidence_scale,
                "ignored_unobserved_positive_count": int(
                    ((~observed_cpu) & (attention > 0)).sum()
                ),
                "ignored_protected_positive_count": int(
                    (observed_cpu & self._protected & (attention > 0)).sum()
                ),
                "input_gate_min": float(input_gate.min()) if updated_count else 0.0,
                "input_gate_mean": float(input_gate.mean()) if updated_count else 0.0,
                "input_gate_max": float(input_gate.max()) if updated_count else 0.0,
                "forget_gate_min": float(forget_gate.min()) if updated_count else 0.0,
                "forget_gate_mean": float(forget_gate.mean()) if updated_count else 0.0,
                "forget_gate_max": float(forget_gate.max()) if updated_count else 0.0,
                "mean_abs_state_delta": mean_abs_delta,
            }
        )
        return diagnostics


class RecurrentImportance(MultimodalImportance):
    """Backward-compatible image/non-image view of the canonical engine.

    Image positions use modality ID 2 and every non-image position uses text
    ID 1. This intentionally keeps the legacy meaning of
    ``selected_text_tokens`` as *all non-image tokens*, including control and
    system-prefix positions. The supplied prior remains defined on every
    position, exactly as in the original implementation.
    """

    _policy = "training_free_gated_recurrent_v1"

    def __init__(
        self,
        image_score: torch.Tensor,
        image_mask: torch.Tensor,
        budget: int,
        protected: torch.Tensor | None = None,
        image_floor: float = 0.35,
        decay: float = 0.9,
    ) -> None:
        raw_score = _score_vector(image_score, "image_score")
        mask = _bool_vector(image_mask, "image_mask", int(raw_score.numel()))
        modality_ids = torch.where(
            mask,
            torch.tensor(2, dtype=torch.long),
            torch.tensor(1, dtype=torch.long),
        )
        super().__init__(
            prior_scores=raw_score,
            modality_ids=modality_ids,
            budget=budget,
            protected=protected,
            prior_floor=image_floor,
            decay=decay,
        )

    @property
    def image_floor(self) -> float:
        return self.prior_floor

    @property
    def _image_prior(self) -> torch.Tensor:
        """Legacy mutable view; new code should use copied ``prior_scores``."""
        return self._prior_scores

    @property
    def _image_mask(self) -> torch.Tensor:
        """Legacy computed image mask; it is not hidden persistent state."""
        return self._modality_ids == 2

    @property
    def _image_scale(self) -> float:
        return self._prior_scale

    def append(self, n_tokens: int) -> None:
        """Append zero-prior non-image positions using legacy semantics."""
        super().append(n_tokens)
