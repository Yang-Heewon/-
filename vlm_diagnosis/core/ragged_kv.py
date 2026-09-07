"""Physically ragged (layer, KV-head, token) cache and reference eager backend.

No dense/padded historical K/V is retained. Each head owns only its survivors.
This is a correctness-first, batch-one, inference-only backend for Qwen2.5/3
text decoders; Python head loops are not a fused-kernel speedup claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading

import torch
from transformers.cache_utils import Cache
import transformers.models.qwen2.modeling_qwen2 as _q2
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q25
import transformers.models.qwen3_vl.modeling_qwen3_vl as _q3


@dataclass
class HeadKV:
    key: torch.Tensor
    value: torch.Tensor
    token_ids: torch.Tensor


def _indices(ids):
    ids = torch.as_tensor(ids)
    if ids.ndim != 1 or ids.dtype != torch.long or (ids.numel() and (
            ids[0] < 0 or not bool((ids[1:] > ids[:-1]).all()))):
        raise ValueError("head token IDs must be sorted unique nonnegative int64")
    return ids.detach().cpu().clone()


class RaggedKVCache(Cache):
    """Only use with RaggedAttention; length is logical, not stored capacity."""

    def __init__(self, kv, keep_ids=None, device=None):
        super().__init__(layers=[])
        if not kv:
            raise ValueError("empty decoder cache")
        shape = kv[0][0].shape
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError("requires batch-one dense source K/V")
        if any(k.shape != shape or v.shape != shape or k.dtype != kv[0][0].dtype
               or v.dtype != k.dtype for k, v in kv):
            raise ValueError("global pair budgets require equal head dimensions/dtypes across layers")
        self.n_layers, self.n_heads = len(kv), shape[1]
        self.head_dim = shape[-1]
        self.total_seen = shape[-2]
        self.pair_bytes = 2 * self.head_dim * kv[0][0].element_size()
        groups = self.n_layers * self.n_heads
        if keep_ids is None:
            keep_ids = [torch.arange(self.total_seen) for _ in range(groups)]
        if len(keep_ids) != groups:
            raise ValueError("one keep-ID vector is required per layer/KV head")
        owned_ids = [_indices(ids) for ids in keep_ids]
        if any(ids.numel() and ids[-1] >= self.total_seen for ids in owned_ids):
            raise ValueError("initial head ID outside source prefix")
        self.heads = []
        for group, ids in enumerate(owned_ids):
            layer, head = divmod(group, self.n_heads)
            k, v = kv[layer]
            target = k.device if device is None else device
            self.heads.append(HeadKV(
                k[0, head].detach().index_select(0, ids.to(k.device)).to(target),
                v[0, head].detach().index_select(0, ids.to(v.device)).to(target), ids))
        self.backend_active = False
        self._next_layer = 0
        self.query_ids = None

    @property
    def counts(self):
        return [head.key.shape[0] for head in self.heads]

    @property
    def pair_count(self):
        return sum(self.counts)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for head in self.heads for t in (head.key, head.value))

    @property
    def metadata_bytes(self):
        return sum(head.token_ids.numel() * 8 for head in self.heads)

    def get_seq_length(self, layer_idx=0):
        return self.total_seen

    def get_mask_sizes(self, cache_position, layer_idx):
        raise RuntimeError("ragged attention owns its masks; pass a zero 4D (1,1,new_tokens,1) placeholder")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if not self.backend_active:
            raise RuntimeError("RaggedKVCache requires an active RaggedAttention context")
        if layer_idx != self._next_layer:
            raise RuntimeError("ragged cache decoder layers were not called exactly in order")
        if (key_states.ndim != 4 or key_states.shape != value_states.shape or key_states.shape[0] != 1
                or key_states.shape[1] != self.n_heads or key_states.shape[-1] != self.head_dim):
            raise ValueError("decoder K/V shape disagrees with ragged cache")
        n = key_states.shape[-2]
        if not n:
            raise ValueError("empty decoder update")
        if layer_idx == 0:
            self.query_ids = torch.arange(self.total_seen, self.total_seen + n)
        elif self.query_ids.numel() != n:
            raise RuntimeError("decoder layers appended different token counts")
        for h in range(self.n_heads):
            group = layer_idx * self.n_heads + h
            old = self.heads[group]
            self.heads[group] = HeadKV(
                torch.cat((old.key, key_states[0, h].detach())),
                torch.cat((old.value, value_states[0, h].detach())),
                torch.cat((old.token_ids, self.query_ids)))
        self._next_layer = (layer_idx + 1) % self.n_layers
        if self._next_layer == 0:
            self.total_seen += n
        # The patched attention consumes self.heads, not these projected
        # new-only tensors. Never synthesize a dense historical cache here.
        return key_states, value_states

    def retain(self, keep_ids):
        if self._next_layer:
            raise RuntimeError("cannot prune in the middle of a decoder forward")
        if len(keep_ids) != len(self.heads):
            raise ValueError("one keep-ID vector is required per head")
        selections = []
        for head, requested in zip(self.heads, keep_ids):
            ids = _indices(requested)
            positions = torch.searchsorted(head.token_ids, ids)
            if ids.numel() and (positions.max() >= head.token_ids.numel()
                               or not torch.equal(head.token_ids[positions], ids)):
                raise ValueError("cannot restore an evicted head/token pair")
            selections.append((ids, positions))
        compacted = []
        for head, (ids, slots) in zip(self.heads, selections):
            compacted.append(HeadKV(
                head.key.index_select(0, slots.to(head.key.device)),
                head.value.index_select(0, slots.to(head.value.device)), ids))
        self.heads = compacted


_PATCH_LOCK = threading.Lock()


class RaggedAttention:
    """Reference GQA attention over each head's physical survivors.

    Temporary module patches are scoped to the exact decoder attention objects
    and restored even on exceptions. Parallel/nested ragged forwards are rejected.
    """

    def __init__(self, model, cache, collect=True):
        from .session_cache import language_layers
        self.modules = [layer.self_attn for layer in language_layers(model)]
        if len(self.modules) != cache.n_layers:
            raise ValueError("model and ragged cache have different layer counts")
        for li, module in enumerate(self.modules):
            if not isinstance(module, (_q2.Qwen2Attention, _q25.Qwen2_5_VLAttention, _q3.Qwen3VLTextAttention)):
                raise TypeError("ragged backend only supports tested Qwen2 text and Qwen2.5/3 VL decoder attention")
            if module.layer_idx != li or module.config._attn_implementation != "eager" or module.training:
                raise ValueError("ragged backend requires ordered, eval-mode eager decoder layers")
            if getattr(module, "sliding_window", None):
                raise ValueError("sliding-window attention is not supported by this backend")
        self.lookup = {id(module): i for i, module in enumerate(self.modules)}
        self.cache, self.collect = cache, collect
        self.mass = [torch.zeros(n, dtype=torch.float32) for n in cache.counts]
        self.rows = [0] * len(cache.heads)
        self.calls = 0

    def _attention(self, module, query, key, value, mask, scaling, dropout=0., **kwargs):
        li = self.lookup[id(module)]
        if query.shape[0] != 1 or query.shape[1] % self.cache.n_heads:
            raise ValueError("ragged attention requires batch-one grouped query heads")
        if dropout:
            raise ValueError("dropout is unsupported during ragged inference")
        n, groups = query.shape[-2], query.shape[1] // self.cache.n_heads
        if (not isinstance(mask, torch.Tensor) or mask.shape != (1, 1, n, 1)
                or bool(mask.count_nonzero())):
            raise ValueError("ragged attention requires a zero (1,1,new_tokens,1) placeholder; external masks are unsupported")
        out = []
        for h in range(self.cache.n_heads):
            g = li * self.cache.n_heads + h
            head = self.cache.heads[g]
            q = query[0, h*groups:(h+1)*groups]
            # Prescale prevents the same fp16 QK overflow as the default loader.
            logits = (q * scaling) @ head.key.transpose(0, 1)
            blocked = head.token_ids[None, :] > self.cache.query_ids[:, None]
            logits.masked_fill_(blocked.to(logits.device)[None], float("-inf"))
            weights = logits.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
            out.append(weights @ head.value)
            if self.collect:
                received = weights.sum(dim=1, dtype=torch.float32).mean(0).detach().cpu()
                if not torch.isfinite(received).all():
                    raise RuntimeError("nonfinite ragged attention score")
                previous = self.mass[g]
                if received.numel() < previous.numel():
                    raise RuntimeError("cache shrank within an observation interval")
                self.mass[g] = torch.nn.functional.pad(previous, (0, received.numel()-previous.numel())) + received
                self.rows[g] += n
        self.calls += 1
        return torch.cat(out, dim=0).transpose(0, 1).unsqueeze(0).contiguous(), None

    def __enter__(self):
        if not _PATCH_LOCK.acquire(blocking=False):
            raise RuntimeError("concurrent/nested ragged attention is unsupported")
        if self.cache.backend_active:
            _PATCH_LOCK.release()
            raise RuntimeError("ragged cache is already active")
        self.originals = (_q2.eager_attention_forward, _q25.eager_attention_forward, _q3.eager_attention_forward)
        def wrap(original):
            def forward(module, query, key, value, attention_mask, scaling, dropout=0., **kw):
                if id(module) in self.lookup:
                    return self._attention(module, query, key, value, attention_mask, scaling, dropout, **kw)
                return original(module, query, key, value, attention_mask, scaling, dropout, **kw)
            return forward
        (_q2.eager_attention_forward, _q25.eager_attention_forward,
         _q3.eager_attention_forward) = map(wrap, self.originals)
        self.cache.backend_active = True
        return self

    def __exit__(self, *exc):
        (_q2.eager_attention_forward, _q25.eager_attention_forward,
         _q3.eager_attention_forward) = self.originals
        self.cache.backend_active = False
        self.cache.query_ids = None
        _PATCH_LOCK.release()
        if not any(exc) and (self.cache._next_layer or self.calls % self.cache.n_layers):
            raise RuntimeError("incomplete ragged decoder execution")
        return False

    def means(self):
        if not self.collect or not all(self.rows):
            raise RuntimeError("no complete per-head attention observations")
        return [mass / rows for mass, rows in zip(self.mass, self.rows)]


class HeadAttentionMass:
    """Initial dense prefill scores without averaging layers or KV heads."""

    def __init__(self, model, row_start, row_end):
        from .session_cache import language_layers
        self.modules = [layer.self_attn for layer in language_layers(model)]
        self.row_start, self.row_end = row_start, row_end
        self.scores = {}
        self.handles = []

    def __enter__(self):
        def hook(li, module, args, output):
            if li in self.scores:
                raise RuntimeError("expected one initial prefill per decoder layer")
            weights = output[1]
            if weights is None or weights.ndim != 4 or weights.shape[0] != 1:
                raise RuntimeError("per-head prefill requires eager attention")
            g = module.num_key_value_groups
            hq, _, nk = weights.shape[1:]
            reduced = weights[0, :, self.row_start:self.row_end].mean(dim=1, dtype=torch.float32)
            self.scores[li] = reduced.reshape(hq // g, g, nk).mean(1).detach().cpu()
        for li, module in enumerate(self.modules):
            self.handles.append(module.register_forward_hook(lambda m,a,o,li=li: hook(li,m,a,o)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def mean(self):
        if len(self.scores) != len(self.modules):
            raise RuntimeError("incomplete per-head prefill capture")
        scores = torch.stack([self.scores[i] for i in range(len(self.modules))])
        if not torch.isfinite(scores).all():
            raise RuntimeError("nonfinite per-head prefill prior")
        return scores
