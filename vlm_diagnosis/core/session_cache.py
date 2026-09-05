"""Modality-neutral, physical token-common session KV compression.

Default ``delete`` storage keeps only selected K/V and their importance state.
Eviction is irreversible: candidates are survivors plus the current turn's new
tokens. ``offload`` explicitly restores the older uncompressed CPU reservoir
experiment. New tokens and compaction copies use additional transient memory.
Original post-RoPE keys are preserved without rerotation in either mode.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time

import torch
from transformers import DynamicCache

from .loader import assert_finite_logits
from .recurrent_importance import MultimodalImportance
from .session_types import SessionSeed, SessionInput, TokenFeatures, validate_modalities
from .session_adapters import QwenImageAdapter, QwenImageTemplate as SessionTemplate


def language_layers(model):
    core = getattr(model, "model", model)
    return (core.language_model if hasattr(core, "language_model") else core).layers


class AttentionMass:
    """Reduce actual eager attention weights in-place in the execution pipeline.

    No QK recomputation, extra scoring forward, or full-layer attention dump.
    A CPU vector accumulates the mass received by each physical cache slot;
    all decoder layers and query heads are averaged. Different forward lengths
    are supported because the active cache only appends within a turn.
    """

    def __init__(self, model, row_start=0, row_end=None):
        self.layers = language_layers(model)
        if not len(self.layers):
            raise ValueError("attention capture needs decoder layers")
        self.root = model if hasattr(model, "register_forward_pre_hook") else model.model
        self.layer_indices = {id(layer.self_attn): i for i, layer in enumerate(self.layers)}
        self.row_start, self.row_end = row_start, row_end
        self.mass = torch.zeros(0, dtype=torch.float32)
        self.row_count = 0
        self.calls = 0
        self.handles = []
        self.errors = []
        self.forward_calls = 0
        self.forward_shape = None
        self.in_forward = False

    def _forward_start(self, module, args):
        if self.in_forward:
            self.errors.append("nested model forward during attention capture")
        self.in_forward = True
        self.forward_calls = 0
        self.forward_shape = None

    def _forward_end(self, module, args, output):
        if self.forward_calls != len(self.layers):
            self.errors.append("incomplete decoder layers within a model forward")
        self.in_forward = False

    def _hook(self, module, args, output):
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if weights is None or weights.ndim != 4 or weights.shape[0] != 1:
            raise RuntimeError("session scoring requires batch-one eager decoder attention")
        if not self.in_forward or self.layer_indices[id(module)] != self.forward_calls:
            self.errors.append("decoder layers were not observed in forward order")
        shape = tuple(weights.shape[-2:])
        if self.forward_shape is not None and shape != self.forward_shape:
            self.errors.append("decoder attention shapes differ within one model forward")
        self.forward_shape = shape
        self.forward_calls += 1
        end = weights.shape[-2] if self.row_end is None else self.row_end
        if not 0 <= self.row_start < end <= weights.shape[-2]:
            raise ValueError("invalid attention observation row range")
        # Float32 reduction avoids fp16 summation overflow without casting the
        # full attention matrix or retaining it beyond this layer.
        mass = weights[0, :, self.row_start:end].sum(dim=1, dtype=torch.float32).mean(0).detach().cpu()
        if not torch.isfinite(mass).all():
            raise RuntimeError("non-finite attention mass")
        if mass.numel() < self.mass.numel():
            raise RuntimeError("physical cache shrank within an observation window")
        if mass.numel() > self.mass.numel():
            self.mass = torch.nn.functional.pad(self.mass, (0, mass.numel() - self.mass.numel()))
        self.mass += mass
        self.row_count += end - self.row_start
        self.calls += 1

    def __enter__(self):
        self.handles = [self.root.register_forward_pre_hook(self._forward_start),
                        self.root.register_forward_hook(self._forward_end)]
        self.handles.extend(layer.self_attn.register_forward_hook(self._hook) for layer in self.layers)
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def mean(self):
        if self.errors:
            raise RuntimeError("; ".join(self.errors))
        if not self.row_count or self.calls % len(self.layers):
            raise RuntimeError("incomplete decoder attention capture")
        return self.mass / self.row_count


class ColdKV:
    """Chronologically indexed immutable old KV plus append-only session KV."""

    def __init__(self, kv):
        self.kv = tuple((k.detach().cpu(), v.detach().cpu()) for k, v in kv)
        if not self.kv:
            raise ValueError("empty KV reservoir")
        lengths = {k.shape[-2] for pair in self.kv for k in pair}
        if len(lengths) != 1 or any(k.ndim != 4 or k.shape != v.shape or k.shape[0] != 1
                                    for k, v in self.kv):
            raise ValueError("expected equal-length batch-one K/V pairs")
        self.length = lengths.pop()

    @property
    def token_bytes(self):
        return sum(k.shape[0] * k.shape[1] * k.shape[-1] * k.element_size()
                   for pair in self.kv for k in pair)

    @property
    def nbytes(self):
        return self.length * self.token_bytes

    def gather(self, indices, device):
        indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        if indices.ndim != 1 or not indices.numel():
            raise ValueError("active cache must contain tokens")
        if int(indices[0]) < 0 or int(indices[-1]) >= self.length or not bool((indices[1:] > indices[:-1]).all()):
            raise ValueError("active indices must be sorted, unique, and in range")
        return DynamicCache.from_legacy_cache(tuple(
            (k.index_select(-2, indices).to(device), v.index_select(-2, indices).to(device))
            for k, v in self.kv))

    def append_from_active(self, cache, old_active_length):
        legacy = cache.to_legacy_cache()
        new_count = cache.get_seq_length() - old_active_length
        if new_count < 0 or len(legacy) != len(self.kv):
            raise ValueError("active cache and reservoir disagree")
        if new_count:
            self.kv = tuple(
                (torch.cat((old_k, k[..., old_active_length:, :].detach().cpu()), dim=-2),
                 torch.cat((old_v, v[..., old_active_length:, :].detach().cpu()), dim=-2))
                for (old_k, old_v), (k, v) in zip(self.kv, legacy))
            self.length += new_count
        return new_count


def compact_cache(cache, positions):
    """Copy selected physical columns into owning storage, never a slice view.

    The caller must release the old cache/output after this returns. The old
    cache, gathered tuple, and DynamicCache's own constructor copies temporarily
    coexist. Unselected columns are not retained by the returned DynamicCache.
    """
    positions = torch.as_tensor(positions, dtype=torch.long, device="cpu")
    length = cache.get_seq_length()
    if positions.ndim != 1 or not positions.numel():
        raise ValueError("compressed cache must retain tokens")
    if int(positions[0]) < 0 or int(positions[-1]) >= length or not bool((positions[1:] > positions[:-1]).all()):
        raise ValueError("compact positions must be sorted, unique, and in range")
    return DynamicCache.from_legacy_cache(tuple(
        (k.index_select(-2, positions.to(k.device)), v.index_select(-2, positions.to(v.device)))
        for k, v in cache.to_legacy_cache()))


@dataclass
class ImageSeed:
    """Compatibility constructor for the original single-image API."""
    kv: tuple
    prefix_ids: torch.Tensor
    image_mask: torch.Tensor
    image_score: torch.Tensor
    next_position: int
    prefill_seconds: float

    def as_session_seed(self):
        return SessionSeed(
            self.kv, self.prefix_ids, self.image_score,
            torch.where(self.image_mask, 2, 0).long(),
            self.next_position, self.prefill_seconds,
            adapter_id=QwenImageAdapter.adapter_id)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def prefill_image(model, processor, image, device):
    """Legacy image entry point; actual preprocessing belongs to the adapter."""
    return QwenImageAdapter().prefill(model, processor, image, device)


class MultimodalSession:
    """One shared decoder cache with an explicit model/modality adapter.

    Future requests never enter the selection made at the previous boundary.
    Adapters own input encoding, position semantics and attention observation;
    this engine owns cache eviction and token-aligned recurrent memory.
    Native audio/video support requires a compatible concrete adapter/model.
    """

    def __init__(self, model, processor, seed, device, budget, condition="recurrent",
                 prior_floor=0.35, decay=0.9, n_sink=4, storage="delete", *,
                 adapter, template=None, selector=None):
        if condition not in {"recurrent", "full", "static", "image_static"}:
            raise ValueError("unknown session condition")
        if storage not in {"delete", "offload"}:
            raise ValueError("storage must be delete or offload")
        if not isinstance(seed, SessionSeed):
            raise TypeError("MultimodalSession requires a SessionSeed")
        if seed.adapter_id != adapter.adapter_id:
            raise ValueError("seed and session adapter disagree")
        names = seed.modality_names
        present = {names[int(i)] for i in seed.modality_ids.unique()}
        if not present <= set(adapter.supported_modalities):
            raise ValueError("seed contains modalities unsupported by the adapter")
        self.model, self.processor, self.device = model, processor, device
        self.adapter = adapter
        self.condition, self.storage = condition, storage
        self.modality_names = dict(names)
        self.cold = self.cache = None
        self.initial_length = seed.prefix_ids.shape[1]
        self.total_seen = self.initial_length
        self.position, self.turn = seed.next_position, 0
        self.template = template if template is not None else adapter.make_template(processor, seed)
        self.features = TokenFeatures(seed.token_features, self.initial_length)
        protected = torch.zeros(self.initial_length, dtype=torch.bool)
        if not 0 <= n_sink <= self.initial_length:
            raise ValueError("invalid protected prefix length")
        protected[:n_sink] = True
        self.state = MultimodalImportance(
            seed.prior_scores, seed.modality_ids, budget, protected,
            prior_floor=prior_floor, decay=decay, modality_names=names, selector=selector)
        keep, self.selection = self.state.select()
        self.active_indices = (torch.arange(self.initial_length) if condition == "full"
                               else keep.nonzero(as_tuple=True)[0])
        initial_counts = self._counts(seed.modality_ids)
        _sync(device)
        started = time.perf_counter()
        source = ColdKV(seed.kv)
        self.token_bytes = source.token_bytes
        if storage == "offload":
            self.cold = source
        else:
            self.cache = source.gather(self.active_indices, device)
            if condition != "full":
                self.features.retain(self.active_indices)
                self.state.retain(keep)
                _, self.selection = self.state.select()
        _sync(device)
        self.initial_cache_setup_seconds = time.perf_counter() - started
        self.initial_deleted_tokens = (self.initial_length - self.active_indices.numel()
                                       if storage == "delete" else 0)
        retained = self._selection_snapshot()["selected_tokens_by_modality"]
        self.initial_deleted_tokens_by_modality = {
            name: initial_counts[name] - retained[name] if storage == "delete" else 0
            for name in initial_counts}

    def _counts(self, modality_ids):
        return {name: int((modality_ids == code).sum())
                for code, name in self.modality_names.items()}

    def _selected_modalities(self):
        modalities = self.state.modality_ids
        return modalities[self.active_indices] if self.storage == "offload" else modalities

    def _selection_snapshot(self):
        indices = self.active_indices
        modalities = self._selected_modalities()
        counts = self._counts(modalities)
        prefix = indices < self.initial_length
        names_to_ids = {name: code for code, name in self.modality_names.items()}
        text = modalities == names_to_ids.get("text", -1)
        control = modalities == names_to_ids.get("control", -1)
        info = dict(self.selection) if self.condition == "recurrent" else {
            "policy": self.condition,
            "image_weight": 1.0 if self.condition != "full" else None,
            "prior_weight": 1.0 if self.condition != "full" else None,
            "history_weight": 0.0 if self.condition != "full" else None,
        }
        info.update({
            "kept_count": indices.numel(),
            "selected_tokens_by_modality": counts,
            "selected_image_tokens": counts.get("image", 0),
            "selected_prefix_control_tokens": int((prefix & control).sum()),
            "selected_history_text_tokens": int((~prefix & text).sum()),
        })
        return info

    def _validate_input(self, prepared):
        if not isinstance(prepared, SessionInput):
            raise TypeError("adapter must return a SessionInput")
        ids, _ = validate_modalities(prepared.modality_ids, prepared.input_ids.shape[1],
                                     self.modality_names)
        present = {self.modality_names[int(i)] for i in ids.unique()}
        if not present <= set(self.adapter.supported_modalities):
            raise ValueError("new input contains unsupported modalities")
        self.features.validate_append(prepared.token_features, ids.numel())
        if (self.state.prior_scale == 0 and prepared.prior_scores is not None
                and bool((prepared.prior_scores > 0).any())):
            raise ValueError("positive new priors require a nonzero initial calibration scale")

    def _forward(self, prepared, cache):
        self._validate_input(prepared)
        old = cache.get_seq_length()
        output, next_position = self.adapter.forward(
            self.model, prepared, cache, self.position, self.device)
        assert_finite_logits(output.logits, f"recurrent_session_turn_{self.turn+1}")
        if output.past_key_values.get_seq_length() != old + prepared.input_ids.shape[1]:
            raise RuntimeError("adapter KV growth does not match token metadata")
        if not isinstance(next_position, int) or next_position <= self.position:
            raise RuntimeError("adapter must advance the logical position")
        self.position = next_position
        return output

    @torch.no_grad()
    def answer(self, request, max_new_tokens=32):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        # Reject incompatible requests before releasing/mutating the live KV.
        suffix = self.adapter.prepare_turn(self.template, request, first=self.turn == 0)
        ending = self.adapter.text_input(self.template.ending_ids, kind="ending")
        self._validate_input(suffix)
        self._validate_input(ending)
        eos_ids = self.adapter.stop_token_ids(self.model)
        old_indices = self.active_indices.clone()
        old_length, old_active = self.total_seen, old_indices.numel()
        selection_before = self._selection_snapshot()
        _sync(self.device)
        started = time.perf_counter()
        if self.storage == "offload":
            cache = self.cold.gather(old_indices, self.device)
        else:
            if self.cache is None:
                raise RuntimeError("session has no usable cache after an interrupted turn")
            cache, self.cache = self.cache, None
        _sync(self.device)
        load_seconds = time.perf_counter() - started
        generated, chunks = [], []
        predicted_stop = None

        def run(prepared, active):
            output = self._forward(prepared, active)
            # Keep only compactable token descriptors, never pixel/audio input.
            chunks.append((prepared.modality_ids, prepared.prior_scores, prepared.token_features))
            return output

        observer = self.adapter.observe(self.model) if self.condition == "recurrent" else nullcontext()
        with observer as capture:
            output = run(suffix, cache)
            del suffix
            _sync(self.device)
            ttft = time.perf_counter() - started
            for _ in range(max_new_tokens):
                token = int(output.logits[0, -1].argmax())
                if token in eos_ids:
                    predicted_stop = token
                    break
                generated.append(token)
                output = run(self.adapter.text_input(torch.tensor([[token]])), output.past_key_values)
            output = run(ending, output.past_key_values)
        cache = output.past_key_values
        new_count = cache.get_seq_length() - old_active
        if sum(ids.numel() for ids, _, _ in chunks) != new_count:
            raise RuntimeError("new KV and modality descriptor counts disagree")
        self.total_seen += new_count
        candidate_ids = torch.cat((old_indices, torch.arange(old_length, self.total_seen)))
        if self.storage == "offload":
            self.cold.append_from_active(cache, old_active)
        for modality_ids, priors, features in chunks:
            self.state.append(modality_ids.numel(), modality_ids=modality_ids, prior_scores=priors)
            self.features.append(features, modality_ids.numel())
        del chunks
        update = {}
        if capture is not None:
            mean = capture.mean()
            if mean.numel() != candidate_ids.numel():
                raise RuntimeError("attention slots and saved cache disagree")
            if self.storage == "offload":
                mass = torch.zeros(self.cold.length)
                observed = torch.zeros(self.cold.length, dtype=torch.bool)
                mass[candidate_ids], observed[candidate_ids] = mean, True
            else:
                mass, observed = mean, torch.ones(mean.numel(), dtype=torch.bool)
            update = self.state.update(mass, observed)
        candidate_modalities = (self.state.modality_ids[candidate_ids] if self.storage == "offload"
                                else self.state.modality_ids)
        candidate_counts = self._counts(candidate_modalities)
        peak_tokens = cache.get_seq_length()
        del output
        deleted_this_turn, compaction_peak_tokens = 0, peak_tokens
        if self.condition == "recurrent":
            keep, self.selection = self.state.select()
            selected_slots = keep.nonzero(as_tuple=True)[0]
            self.active_indices = (candidate_ids[selected_slots] if self.storage == "delete" else selected_slots)
        elif self.condition == "full":
            self.active_indices = torch.arange(self.total_seen)
        else:
            self.active_indices = old_indices
            keep = torch.zeros(self.state.n_tokens, dtype=torch.bool)
            keep[:old_active] = True
            selected_slots = torch.arange(old_active)
        if self.storage == "delete":
            if self.condition == "full":
                self.cache = cache
            else:
                # from_legacy_cache also cat-copies the gathered B columns:
                # old B+N, gathered B and final B coexist temporarily.
                compaction_peak_tokens += 2 * selected_slots.numel()
                self.cache = compact_cache(cache, selected_slots)
                deleted_this_turn = peak_tokens - selected_slots.numel()
                self.features.retain(selected_slots)
                self.state.retain(keep)
                _, self.selection = self.state.select()
        del cache
        selection_after = self._selection_snapshot()
        entered = int((~torch.isin(self.active_indices, old_indices)).sum())
        evicted = int((~torch.isin(old_indices, self.active_indices)).sum())
        _, state_info = self.state.select()
        state_bytes = state_info["state_bytes"]
        cold_bytes = self.cold.nbytes if self.cold is not None else 0
        resident_bytes = self.cache.get_seq_length() * self.token_bytes if self.cache is not None else 0
        retained_bytes = cold_bytes + resident_bytes
        metadata_bytes = (self.active_indices.numel() * self.active_indices.element_size()
                          + self.features.nbytes + self.adapter.metadata_bytes(self.template))
        deleted_by_modality = {
            name: count - selection_after["selected_tokens_by_modality"][name]
            if self.storage == "delete" else 0 for name, count in candidate_counts.items()}
        _sync(self.device)
        self.turn += 1
        return {
            "condition_id": self.condition, "step": self.turn,
            "adapter_id": self.adapter.adapter_id, "modality_names": dict(self.modality_names),
            "storage_mode": self.storage,
            "compression_applied": self.storage == "delete" and self.condition != "full",
            "prediction": self.adapter.decode(self.processor, generated),
            "generated_tokens": len(generated), "new_session_tokens": new_count,
            "predicted_stop_token_id": predicted_stop,
            "hit_generation_limit": predicted_stop is None,
            "termination_policy": "canonical_assistant_ending; generated_content_ids_preserved",
            "historical_tokens": old_length, "active_history_tokens": old_active,
            "next_active_history_tokens": self.active_indices.numel(),
            "active_history_kv_bytes": old_active * self.token_bytes,
            "peak_active_kv_tokens": peak_tokens, "peak_active_kv_bytes": peak_tokens * self.token_bytes,
            "compaction_peak_kv_bytes_upper_bound": compaction_peak_tokens * self.token_bytes,
            "cold_kv_bytes": cold_bytes, "resident_gpu_kv_bytes": resident_bytes,
            "retained_kv_tokens": self.cold.length if self.cold is not None else self.cache.get_seq_length(),
            "retained_kv_bytes": retained_bytes,
            "retained_kv_fraction_of_initial": retained_bytes / (self.initial_length * self.token_bytes),
            "logical_history_tokens_after": self.total_seen,
            "initial_deleted_tokens": self.initial_deleted_tokens,
            "initial_deleted_tokens_by_modality": self.initial_deleted_tokens_by_modality,
            "deleted_tokens_this_turn": deleted_this_turn,
            "deleted_tokens_by_modality": deleted_by_modality,
            "deleted_image_tokens_this_turn": deleted_by_modality.get("image", 0),
            "selector_state_bytes": state_bytes, "session_metadata_bytes": metadata_bytes,
            "token_feature_bytes": self.features.nbytes,
            "persistent_session_tensor_bytes": retained_bytes + state_bytes + metadata_bytes,
            "combined_kv_and_state_bytes": cold_bytes + compaction_peak_tokens * self.token_bytes + state_bytes,
            "h2d_kv_bytes": old_active * self.token_bytes if self.storage == "offload" else 0,
            "d2h_new_kv_bytes": new_count * self.token_bytes if self.storage == "offload" else 0,
            "initial_cache_setup_seconds": self.initial_cache_setup_seconds,
            "load_seconds": load_seconds, "ttft_seconds": ttft,
            "turn_seconds": time.perf_counter() - started,
            "active_indices": old_indices.tolist(), "next_active_indices": self.active_indices.tolist(),
            "selection_before": selection_before, "selection_after": selection_after,
            "entered_tokens": entered, "evicted_tokens": evicted, "state_update": update,
        }


class RecurrentSession(MultimodalSession):
    """Backwards-compatible image session; use MultimodalSession for adapters."""

    def __init__(self, model, processor, seed, device, budget, condition="recurrent",
                 image_floor=0.35, decay=0.9, n_sink=4, storage="delete", *, selector=None):
        if isinstance(seed, ImageSeed):
            seed = seed.as_session_seed()
        adapter = QwenImageAdapter()
        template = SessionTemplate(processor, model.config.image_token_id, seed.prefix_ids)
        super().__init__(
            model, processor, seed, device, budget, condition,
            prior_floor=image_floor, decay=decay, n_sink=n_sink, storage=storage,
            adapter=adapter, template=template, selector=selector)

    def _forward(self, prepared, cache):
        if isinstance(prepared, torch.Tensor):
            prepared = self.adapter.text_input(prepared)
        return super()._forward(prepared, cache)
