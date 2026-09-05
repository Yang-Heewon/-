"""Session-level recurrent compression under a GLOBAL physical K/V-pair budget."""
from __future__ import annotations

import time

import torch

from .pair_importance import PairImportance
from .ragged_kv import RaggedAttention, RaggedKVCache
from .session_types import SessionSeed
from .session_adapters import QwenPairAdapter
from .session_cache import _sync


class PairSession:
    """One global budget, independent survivors for each layer and KV head.

    This native backend currently supports Qwen single-image/text sessions.
    Per-pair importance is modality-labelled; other input backends are not
    silently routed through Qwen. Historical eviction is always irreversible.
    """

    def __init__(self, model, processor, seed: SessionSeed, device, budget_pairs,
                 condition="recurrent", prior_floor=.35, decay=.9, n_sink=4, *, adapter=None):
        self.adapter = QwenPairAdapter() if adapter is None else adapter
        if not isinstance(self.adapter, QwenPairAdapter) or seed.adapter_id != self.adapter.adapter_id:
            raise ValueError("native pair session requires a QwenPairAdapter-compatible seed")
        if condition not in {"full", "image_static", "recurrent"}:
            raise ValueError("unknown pair-session condition")
        if seed.pair_prior_scores is None:
            raise ValueError("pair session requires true layer/KV-head prefill scores, not a broadcast token score")
        if seed.token_features:
            raise ValueError("native pair session does not yet support extra per-token features")
        self.model, self.processor, self.device = model, processor, device
        self.condition = condition
        self.position, self.turn = seed.next_position, 0
        self.initial_length = seed.prefix_ids.shape[1]
        self.initial_pair_count = seed.pair_prior_scores.numel()
        self.budget_pairs = budget_pairs
        self.template = self.adapter.make_template(processor, seed)
        self.state = PairImportance(seed.pair_prior_scores, seed.modality_ids, budget_pairs,
                                    n_sink=n_sink, prior_floor=prior_floor, decay=decay,
                                    modality_names=seed.modality_names)
        keep, _ = self.state.select()
        ids = self.state.selected_ids(keep) if condition != "full" else self._resident_ids()
        _sync(device)
        started = time.perf_counter()
        self.cache = RaggedKVCache(seed.kv, ids, device=device)
        if condition != "full":
            self.state.retain(keep)
        _sync(device)
        self.initial_cache_setup_seconds = time.perf_counter() - started
        self.initial_deleted_pairs = self.initial_pair_count - self.cache.pair_count
        self._assert_alignment()

    def _assert_alignment(self):
        state_ids = self._resident_ids()
        if self.state.n_pairs != self.cache.pair_count or any(
                not torch.equal(ids, head.token_ids) for ids, head in zip(state_ids, self.cache.heads)):
            raise RuntimeError("pair importance slots and physical head caches disagree")

    def _resident_ids(self):
        groups, tokens = self.state.group_ids, self.state.token_ids
        return [tokens[groups == group] for group in range(self.state.groups)]

    @torch.no_grad()
    def answer(self, question, max_new_tokens=32):
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        suffix = self.adapter.prepare_turn(self.template, question, first=self.turn == 0)
        ending = self.adapter.text_input(self.template.ending_ids, kind="ending")
        old_total = self.cache.total_seen
        old_pairs = self.cache.pair_count
        before = self.state.snapshot()
        old_groups, old_tokens = self.state.group_ids, self.state.token_ids
        old_pair_keys = old_tokens * self.state.groups + old_groups
        generated, chunks = [], []
        predicted_stop = None
        eos_ids = self.adapter.stop_token_ids(self.model)
        _sync(self.device)
        started = time.perf_counter()

        def run(prepared):
            old = self.cache.total_seen
            output, self.position = self.adapter.forward(
                self.model, prepared, self.cache, self.position, self.device)
            if self.cache.total_seen - old != prepared.input_ids.shape[1]:
                raise RuntimeError("pair cache did not append exactly the new token count")
            chunks.append(prepared.modality_ids)
            return output

        with RaggedAttention(self.model, self.cache, collect=self.condition == "recurrent") as observer:
            output = run(suffix)
            del suffix
            _sync(self.device)
            ttft = time.perf_counter() - started
            for _ in range(max_new_tokens):
                token = int(output.logits[0, -1].argmax())
                if token in eos_ids:
                    predicted_stop = token
                    break
                generated.append(token)
                output = run(self.adapter.text_input(torch.tensor([[token]])))
            output = run(ending)
        del output
        new_tokens = self.cache.total_seen - old_total
        modalities = torch.cat(chunks)
        if modalities.numel() != new_tokens:
            raise RuntimeError("new token modalities and ragged cache growth disagree")
        self.state.append(modalities, old_total)
        update = self.state.observe(observer.means()) if self.condition == "recurrent" else {}
        candidate_pairs = self.cache.pair_count
        candidate_bytes = self.cache.nbytes
        self._assert_alignment()
        if self.condition != "full":
            if self.condition == "image_static":
                keep = self.state.token_ids < self.initial_length
            else:
                keep, _ = self.state.select()
            ids = self.state.selected_ids(keep)
            self.cache.retain(ids)
            self.state.retain(keep)
        self._assert_alignment()
        after = self.state.snapshot()
        _, score_info = self.state.select()
        new_pair_keys = self.state.token_ids * self.state.groups + self.state.group_ids
        entered = int((~torch.isin(new_pair_keys, old_pair_keys)).sum())
        evicted = int((~torch.isin(old_pair_keys, new_pair_keys)).sum())
        metadata_bytes = self.cache.metadata_bytes + self.adapter.metadata_bytes(self.template)
        persistent = self.cache.nbytes + self.state.nbytes + metadata_bytes
        _sync(self.device)
        self.turn += 1
        return {
            "condition_id": self.condition, "step": self.turn, "granularity": "kv_pair",
            "adapter_id": self.adapter.adapter_id, "storage_mode": "delete",
            "compression_applied": self.condition != "full",
            "prediction": self.adapter.decode(self.processor, generated),
            "generated_tokens": len(generated), "new_session_tokens": new_tokens,
            "predicted_stop_token_id": predicted_stop, "hit_generation_limit": predicted_stop is None,
            "termination_policy": "canonical_assistant_ending; generated_content_ids_preserved",
            "initial_prefix_tokens": self.initial_length, "initial_kv_pairs": self.initial_pair_count,
            "budget_pairs": self.budget_pairs, "active_history_pairs": old_pairs,
            "retained_kv_pairs": self.cache.pair_count, "retained_kv_bytes": self.cache.nbytes,
            "retained_kv_fraction_of_initial": self.cache.pair_count / self.initial_pair_count,
            "peak_active_kv_pairs": candidate_pairs, "peak_active_kv_bytes": candidate_bytes,
            # Conservative storage-only bound: old/new owned arrays and current
            # projected keys may coexist. Does not include attention/activations.
            "cache_storage_peak_bytes_upper_bound": 2 * candidate_bytes,
            "cold_kv_bytes": 0, "h2d_kv_bytes": 0, "d2h_new_kv_bytes": 0,
            "initial_deleted_pairs": self.initial_deleted_pairs,
            "deleted_pairs_this_turn": candidate_pairs - self.cache.pair_count,
            "entered_pairs": entered, "evicted_pairs": evicted,
            "logical_history_tokens_after": self.cache.total_seen,
            "selector_state_bytes": self.state.nbytes, "session_metadata_bytes": metadata_bytes,
            "persistent_session_tensor_bytes": persistent,
            "selection_before": before, "selection_after": after,
            "prior_weight": score_info["prior_weight"], "history_weight": score_info["history_weight"],
            "state_update": update, "initial_cache_setup_seconds": self.initial_cache_setup_seconds,
            "ttft_seconds": ttft, "turn_seconds": time.perf_counter() - started,
        }
