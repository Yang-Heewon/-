"""Physical ragged per-(layer, KV-head, token) cache contracts."""
from __future__ import annotations

import gc
import unittest
import weakref

import torch
from transformers import DynamicCache
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q25
import transformers.models.qwen3_vl.modeling_qwen3_vl as _q3
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLTextModel,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from vlm_diagnosis.core.ragged_kv import HeadAttentionMass, RaggedAttention, RaggedKVCache


def _legacy_kv(layers=2, heads=3, tokens=5, dim=4):
    values = []
    for layer in range(layers):
        base = torch.arange(heads * tokens * dim, dtype=torch.float32)
        key = base.view(1, heads, tokens, dim) + 1000 * layer
        values.append((key, key + 0.25))
    return tuple(values)


def _tiny_model(config_type, model_type):
    config = config_type(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    return model_type(config).eval()


class _DenseHeadMask:
    """Layer-aware dense reference supporting cached multi-token suffixes."""

    def __init__(self, keep):
        self.keep = keep
        self.weights = {}

    def __enter__(self):
        self.originals = (_q25.eager_attention_forward, _q3.eager_attention_forward)
        owner = self

        def wrap(original):
            def forward(
                module, query, key, value, attention_mask, scaling,
                dropout=0.0, **kwargs,
            ):
                layer = getattr(module, "layer_idx", None)
                if layer is None:
                    return original(
                        module, query, key, value, attention_mask, scaling,
                        dropout, **kwargs,
                    )
                if attention_mask is None or attention_mask.ndim != 4:
                    raise RuntimeError("dense reference requires a prepared causal mask")
                groups = module.num_key_value_groups
                query_heads, query_length = query.shape[1], query.shape[-2]
                key_length = key.shape[-2]
                mask = attention_mask[..., :key_length].expand(
                    1, query_heads, query_length, key_length
                ).clone()
                evicted = (~owner.keep[layer]).repeat_interleave(groups, dim=0)
                blocked = torch.zeros(
                    query_heads, key_length, dtype=torch.bool, device=mask.device
                )
                blocked[:, :evicted.shape[-1]] = evicted.to(mask.device)
                mask.masked_fill_(blocked[None, :, None], torch.finfo(mask.dtype).min)
                result = original(
                    module, query, key, value, mask, scaling, dropout, **kwargs
                )
                if result[1] is None:
                    raise RuntimeError("dense eager reference did not expose attention")
                owner.weights[layer] = result[1].detach().cpu()
                return result
            return forward

        _q25.eager_attention_forward = wrap(self.originals[0])
        _q3.eager_attention_forward = wrap(self.originals[1])
        return self

    def __exit__(self, *exc):
        _q25.eager_attention_forward, _q3.eager_attention_forward = self.originals
        return False


class RaggedKVStorageTest(unittest.TestCase):
    def test_variable_head_counts_global_pair_budget_and_owning_storage(self):
        legacy = _legacy_kv()
        source_refs = [weakref.ref(tensor) for pair in legacy for tensor in pair]
        source_ptrs = {
            tensor.untyped_storage().data_ptr() for pair in legacy for tensor in pair
        }
        # Flattening order is layer-major then KV-head. The global physical
        # budget is the sum of these independently variable head counts.
        keep_ids = [
            torch.tensor([0]),
            torch.tensor([0, 2]),
            torch.tensor([1, 3, 4]),
            torch.tensor([4]),
            torch.tensor([1, 2]),
            torch.tensor([0, 3, 4]),
        ]
        cache = RaggedKVCache(legacy, keep_ids=keep_ids)
        del legacy
        gc.collect()

        self.assertEqual(cache.n_layers, 2)
        self.assertEqual(cache.n_heads, 3)
        self.assertEqual(cache.counts, [1, 2, 3, 1, 2, 3])
        self.assertEqual(cache.pair_count, 12)
        self.assertEqual(cache.get_seq_length(), 5)
        self.assertTrue(all(reference() is None for reference in source_refs))
        expected_bytes = cache.pair_count * 2 * 4 * torch.tensor(0.0).element_size()
        self.assertEqual(cache.nbytes, expected_bytes)

        for head, expected_ids in zip(cache.heads, keep_ids):
            self.assertEqual(head.token_ids.device.type, "cpu")
            self.assertEqual(head.token_ids.dtype, torch.long)
            self.assertTrue(torch.equal(head.token_ids, expected_ids))
            self.assertEqual(tuple(head.key.shape), (expected_ids.numel(), 4))
            self.assertEqual(head.key.shape, head.value.shape)
            for tensor in (head.key, head.value, head.token_ids):
                self.assertNotIn(tensor.untyped_storage().data_ptr(), source_ptrs)
                self.assertEqual(
                    tensor.untyped_storage().nbytes(),
                    tensor.numel() * tensor.element_size(),
                )

    def test_retain_is_atomic_irreversible_and_allows_empty_heads(self):
        cache = RaggedKVCache(_legacy_kv(layers=1, heads=3))
        keep = [
            torch.empty(0, dtype=torch.long),
            torch.tensor([1, 4]),
            torch.tensor([0, 2, 3]),
        ]
        cache.retain(keep)
        self.assertEqual(cache.counts, [0, 2, 3])
        self.assertEqual(cache.pair_count, 5)
        self.assertEqual(cache.get_seq_length(), 5)
        baseline = [
            (head.key.clone(), head.value.clone(), head.token_ids.clone())
            for head in cache.heads
        ]

        # Logical ID 2 was physically deleted from head 1. A later retain may
        # only choose current survivors, never resurrect it from hidden state.
        invalid = [keep[0], torch.tensor([1, 2]), keep[2]]
        with self.assertRaises(ValueError):
            cache.retain(invalid)
        for head, (key, value, token_ids) in zip(cache.heads, baseline):
            self.assertTrue(torch.equal(head.key, key))
            self.assertTrue(torch.equal(head.value, value))
            self.assertTrue(torch.equal(head.token_ids, token_ids))

        for invalid in (
            keep[:-1],
            [keep[0], torch.tensor([4, 1]), keep[2]],
            [keep[0], torch.tensor([1, 1]), keep[2]],
            [keep[0], torch.tensor([1], dtype=torch.int32), keep[2]],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                cache.retain(invalid)


class RaggedQwenParityTest(unittest.TestCase):
    def test_backend_restores_patches_on_error_and_rejects_unscoped_cache(self):
        model = _tiny_model(Qwen2_5_VLTextConfig, Qwen2_5_VLTextModel)
        cache = RaggedKVCache(_legacy_kv(layers=2, heads=2, dim=8))
        originals = (_q25.eager_attention_forward, _q3.eager_attention_forward)
        projected = torch.zeros(1, 2, 1, 8)
        with self.assertRaisesRegex(RuntimeError, "active RaggedAttention"):
            cache.update(projected, projected, 0)
        with self.assertRaisesRegex(ValueError, "external masks"):
            with RaggedAttention(model, cache) as backend:
                with self.assertRaisesRegex(RuntimeError, "concurrent/nested"):
                    with RaggedAttention(model, cache):
                        pass
                backend._attention(model.layers[0].self_attn, torch.zeros(1, 4, 1, 8),
                                   projected, projected, torch.ones(1, 1, 1, 1), .5)
        self.assertEqual(originals, (_q25.eager_attention_forward, _q3.eager_attention_forward))
        self.assertFalse(cache.backend_active)
        # An exception must also release the process-wide patch lock.
        with RaggedAttention(model, cache):
            pass

    @torch.no_grad()
    def test_tiny_real_qwen_matches_dense_per_head_mask_and_attention(self):
        families = (
            (Qwen2_5_VLTextConfig, Qwen2_5_VLTextModel),
            (Qwen3VLTextConfig, Qwen3VLTextModel),
        )
        for config_type, model_type in families:
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(19)
                model = _tiny_model(config_type, model_type)
                prefix_ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
                prefix_positions = torch.arange(6)[None, None].expand(3, 1, 6)
                with HeadAttentionMass(model, 1, 5) as initial_scores:
                    initial = model(
                        input_ids=prefix_ids,
                        position_ids=prefix_positions,
                        use_cache=True,
                        output_attentions=True,
                    )
                expected_prior = torch.stack([
                    weights[0, :, 1:5].mean(1).reshape(2, 2, 6).mean(1)
                    for weights in initial.attentions
                ])
                torch.testing.assert_close(initial_scores.mean(), expected_prior)
                self.assertEqual(tuple(initial_scores.mean().shape), (2, 2, 6))
                legacy = initial.past_key_values.to_legacy_cache()
                dense = DynamicCache.from_legacy_cache(tuple(
                    (key.clone(), value.clone()) for key, value in legacy
                ))
                # Every flattened (layer, KV-head) has a distinct survivor set
                # and a different count, including one initially empty head.
                keep_ids = [
                    torch.empty(0, dtype=torch.long),
                    torch.tensor([1]),
                    torch.tensor([0, 3]),
                    torch.tensor([0, 2, 5]),
                ]
                ragged = RaggedKVCache(legacy, keep_ids=keep_ids)
                keep = torch.zeros(2, 2, 6, dtype=torch.bool)
                for group, ids in enumerate(keep_ids):
                    keep[group // 2, group % 2, ids] = True

                # Two appended queries make the causal distinction observable:
                # the first may not attend the second, while both see only the
                # per-head survivors from the old prefix.
                ids = torch.tensor([[10, 11]])
                logical_position = 20
                positions = torch.arange(
                    logical_position, logical_position + 2
                )[None, None].expand(3, 1, 2)
                with _DenseHeadMask(keep) as dense_observation:
                    reference = model(
                        input_ids=ids,
                        position_ids=positions,
                        cache_position=torch.tensor([6, 7]),
                        attention_mask=torch.ones(1, 8, dtype=torch.long),
                        past_key_values=dense,
                        use_cache=True,
                        output_attentions=True,
                    )
                with RaggedAttention(model, ragged, collect=True) as observation:
                    actual = model(
                        input_ids=ids,
                        position_ids=positions,
                        cache_position=torch.tensor([6, 7]),
                        # Both supported model families accept a prepared 4D
                        # placeholder; the ragged backend owns the actual
                        # logical-ID causal mask independently for every head.
                        attention_mask=torch.zeros(
                            1, 1, 2, 1, dtype=next(model.parameters()).dtype
                        ),
                        past_key_values=ragged,
                        use_cache=True,
                        output_attentions=False,
                    )

                torch.testing.assert_close(
                    actual.last_hidden_state,
                    reference.last_hidden_state,
                    atol=1e-5,
                    rtol=1e-5,
                )
                self.assertIs(actual.past_key_values, ragged)
                self.assertEqual(ragged.get_seq_length(), 8)
                self.assertEqual(ragged.counts, [2, 3, 4, 5])
                self.assertEqual(ragged.pair_count, 14)
                for head, old_ids in zip(ragged.heads, keep_ids):
                    self.assertTrue(torch.equal(
                        head.token_ids,
                        torch.cat((old_ids, torch.tensor([6, 7]))),
                    ))

                means = observation.means()
                self.assertEqual(len(means), 4)
                groups = model.config.num_attention_heads // model.config.num_key_value_heads
                for group, observed in enumerate(means):
                    layer, kv_head = divmod(group, 2)
                    query_slice = slice(kv_head * groups, (kv_head + 1) * groups)
                    dense_mean = dense_observation.weights[layer][
                        0, query_slice
                    ].mean((0, 1))
                    logical_ids = torch.cat((keep_ids[group], torch.tensor([6, 7])))
                    torch.testing.assert_close(
                        observed,
                        dense_mean[logical_ids],
                        atol=1e-6,
                        rtol=1e-5,
                    )


if __name__ == "__main__":
    unittest.main()
