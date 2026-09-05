import gc
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import torch
from torch import nn
from transformers import DynamicCache

from vlm_diagnosis.core import session_cache as session_mod
from vlm_diagnosis.core import session_adapters as adapter_mod
from vlm_diagnosis.core.session_cache import (
    AttentionMass,
    ColdKV,
    ImageSeed,
    RecurrentSession,
    SessionTemplate,
    prefill_image,
)


def _legacy_kv(length, layers=2):
    """Make layer-distinguishable chronological K/V columns."""
    result = []
    for layer_idx in range(layers):
        base = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1)
        result.append((base + 100 * layer_idx, base + 1000 + 100 * layer_idx))
    return tuple(result)


class _FakeAttention(nn.Module):
    def forward(self, weights):
        output = torch.zeros(
            weights.shape[0], weights.shape[-2], weights.shape[1], dtype=weights.dtype
        )
        return output, weights


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeCore(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(layers)])


class _HookModel(nn.Module):
    """Small hook target that can deliberately violate decoder-layer ordering."""

    def __init__(self, layers=2):
        super().__init__()
        self.model = _FakeCore(layers)

    def forward(self, emissions):
        for layer_idx, weights in emissions:
            self.model.layers[layer_idx].self_attn(weights)
        return torch.tensor(0.0)


class _FakeCacheModel(nn.Module):
    """CPU autoregressive model whose KV records position and token identity."""

    def __init__(self, predictions=(), layers=2, eos=9):
        super().__init__()
        self.model = _FakeCore(layers)
        self.config = SimpleNamespace(image_token_id=99)
        self.generation_config = SimpleNamespace(eos_token_id=eos)
        self.predictions = list(predictions)
        self.records = []

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        if past_key_values is None:
            past_key_values = DynamicCache()
        old = past_key_values.get_seq_length()
        n = input_ids.shape[1]
        if cache_position is None:
            cache_position = torch.arange(old, old + n, device=input_ids.device)
        self.records.append(
            {
                "input_ids": input_ids.detach().cpu().clone(),
                "position_ids": position_ids.detach().cpu().clone(),
                "cache_position": cache_position.detach().cpu().clone(),
                "attention_mask": attention_mask.detach().cpu().clone(),
                "old": old,
                "pixel_values": kwargs.get("pixel_values"),
            }
        )

        logical = position_ids[0].reshape(1, 1, n, 1).to(torch.float32)
        token_values = input_ids.reshape(1, 1, n, 1).to(torch.float32)
        total = old + n
        for layer_idx, layer in enumerate(self.model.layers):
            past_key_values.update(
                logical + 100 * layer_idx,
                token_values + 1000 + 100 * layer_idx,
                layer_idx,
            )
            # Every row attends to its own just-appended slot. This is causal,
            # normalized, and gives every new turn token positive evidence.
            weights = torch.zeros(1, 2, n, total, dtype=torch.float32)
            for row in range(n):
                weights[:, :, row, old + row] = 1.0
            layer.self_attn(weights)

        if self.predictions:
            prediction = self.predictions.pop(0)
        else:
            eos = self.generation_config.eos_token_id
            prediction = eos if isinstance(eos, int) else eos[0]
        vocab = max(32, int(prediction) + 1)
        logits = torch.zeros(1, n, vocab, dtype=torch.float32)
        logits[:, -1, int(prediction)] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class _FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(token) for token in tokens)


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class _FakeSessionTemplate(SessionTemplate):
    def __init__(self, processor, image_token_id, prefix_ids):
        self.prefix_ids = prefix_ids.detach().cpu().clone()
        self.anchor_ids = torch.tensor([[40, 41]])
        self.ending_ids = torch.tensor([[9, 10]])
        self.calls = []

    def suffix(self, question, first):
        self.calls.append((question, first))
        return torch.tensor([[4, 5]]) if first else torch.tensor([[6, 7, 8]])


def _seed(length=5):
    return ImageSeed(
        kv=_legacy_kv(length),
        prefix_ids=torch.tensor([[90, 99, 99, 99, 91]])[:, :length],
        image_mask=torch.tensor([False, True, True, True, False])[:length],
        image_score=torch.tensor([0.0, 1.0, 0.8, 0.2, 0.0])[:length],
        next_position=11,
        prefill_seconds=0.0,
    )


def _make_session(
    condition="recurrent",
    predictions=(7, 9, 0),
    seed=None,
    budget=2,
    eos=9,
    storage="offload",
):
    model = _FakeCacheModel(predictions=predictions, eos=eos)
    processor = _FakeProcessor()
    with mock.patch.object(session_mod, "SessionTemplate", _FakeSessionTemplate):
        session = RecurrentSession(
            model,
            processor,
            seed or _seed(),
            "cpu",
            budget=budget,
            condition=condition,
            image_floor=0.0,
            decay=0.5,
            n_sink=1,
            storage=storage,
        )
    return model, session


class RecurrentSessionTest(unittest.TestCase):
    def test_cold_gather_append_is_token_common_and_chronological(self):
        cold = ColdKV(_legacy_kv(5))
        active = cold.gather(torch.tensor([0, 2, 4]), "cpu")
        self.assertEqual(active.get_seq_length(), 3)
        for layer_idx, (key, value) in enumerate(active.to_legacy_cache()):
            self.assertEqual(key.flatten().tolist(), [100 * layer_idx + x for x in (0, 2, 4)])
            self.assertEqual(
                value.flatten().tolist(), [1000 + 100 * layer_idx + x for x in (0, 2, 4)]
            )

        for layer_idx in range(2):
            new_key = torch.tensor([50.0, 51.0]).view(1, 1, 2, 1) + 100 * layer_idx
            new_value = new_key + 1000
            active.update(new_key, new_value, layer_idx)
        self.assertEqual(cold.append_from_active(active, old_active_length=3), 2)
        self.assertEqual(cold.length, 7)
        self.assertEqual(cold.token_bytes, 16)
        self.assertEqual(cold.nbytes, 112)
        for layer_idx, (key, value) in enumerate(cold.kv):
            self.assertEqual(key[..., -2:, :].flatten().tolist(), [50 + 100 * layer_idx, 51 + 100 * layer_idx])
            self.assertEqual(value[..., -2:, :].flatten().tolist(), [1050 + 100 * layer_idx, 1051 + 100 * layer_idx])

        for invalid in ([2, 2], [3, 1], [-1], [7], []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                cold.gather(invalid, "cpu")

    def test_attention_mass_reduces_layers_heads_rows_and_growing_keys(self):
        model = _HookModel()
        first = [
            torch.tensor([[[[0.8, 0.2, 0.0], [0.1, 0.3, 0.6]],
                           [[0.6, 0.4, 0.0], [0.2, 0.2, 0.6]]]]),
            torch.tensor([[[[0.5, 0.5, 0.0], [0.0, 0.4, 0.6]],
                           [[0.4, 0.6, 0.0], [0.1, 0.3, 0.6]]]]),
        ]
        second = [
            torch.tensor([[[[0.1, 0.2, 0.3, 0.4]], [[0.2, 0.2, 0.2, 0.4]]]]),
            torch.tensor([[[[0.2, 0.1, 0.3, 0.4]], [[0.1, 0.3, 0.2, 0.4]]]]),
        ]
        with AttentionMass(model) as capture:
            model([(0, first[0]), (1, first[1])])
            model([(0, second[0]), (1, second[1])])

        expected = torch.zeros(4)
        rows = 0
        for weights in first + second:
            reduced = weights[0].sum(dim=1).mean(dim=0)
            expected[: reduced.numel()] += reduced
            rows += weights.shape[-2]
        expected /= rows
        torch.testing.assert_close(capture.mean(), expected)
        self.assertEqual(capture.calls, 4)
        self.assertEqual(capture.row_count, 6)

    def test_attention_mass_rejects_per_forward_layer_or_shape_mismatch(self):
        weights3 = torch.ones(1, 1, 1, 3)
        weights4 = torch.ones(1, 1, 1, 4)

        # Two incomplete model forwards must not be accepted merely because
        # their aggregate hook count happens to equal the decoder layer count.
        model = _HookModel()
        with AttentionMass(model) as capture:
            model([(0, weights3)])
            model([(1, weights3)])
        with self.assertRaises(RuntimeError):
            capture.mean()

        # Layers within one decoder forward must describe the same Q/K shape.
        model = _HookModel()
        with AttentionMass(model) as capture:
            model([(0, weights3), (1, weights4)])
        with self.assertRaises(RuntimeError):
            capture.mean()

    def test_real_qwen_eager_attention_hooks_expose_weights(self):
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLTextConfig,
        )
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        configs_and_models = (
            (
                Qwen2_5_VLTextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
                    _attn_implementation="eager",
                ),
                Qwen2_5_VLTextModel,
            ),
            (
                Qwen3VLTextConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
                    _attn_implementation="eager",
                ),
                Qwen3VLTextModel,
            ),
        )
        ids = torch.tensor([[1, 2, 3]])
        positions = torch.arange(3).view(1, 1, 3).expand(3, 1, 3)
        for config, model_type in configs_and_models:
            with self.subTest(model=model_type.__name__):
                text_model = model_type(config).eval()
                wrapped = SimpleNamespace(model=text_model)
                with AttentionMass(wrapped) as capture:
                    output = text_model(
                        input_ids=ids,
                        position_ids=positions,
                        attention_mask=torch.ones_like(ids),
                        use_cache=True,
                    )
                self.assertEqual(output.past_key_values.get_seq_length(), 3)
                self.assertEqual(capture.calls, 2)
                self.assertEqual(capture.row_count, 6)
                self.assertEqual(capture.mean().shape, (3,))
                torch.testing.assert_close(capture.mean().sum(), torch.tensor(1.0))

    def test_forward_separates_logical_mrope_from_physical_cache_slots(self):
        model, session = _make_session()
        self.assertEqual(session.storage, "offload")
        self.assertIsNotNone(session.cold)
        self.assertIsNone(session.cache)
        cache = session.cold.gather(session.active_indices, "cpu")
        output = session._forward(torch.tensor([[20, 21]]), cache)
        first = model.records[-1]
        self.assertEqual(first["old"], 2)
        self.assertEqual(first["cache_position"].tolist(), [2, 3])
        self.assertEqual(first["attention_mask"].shape, (1, 4))
        self.assertTrue(bool(first["attention_mask"].all()))
        for axis in range(3):
            self.assertEqual(first["position_ids"][axis, 0].tolist(), [11, 12])
        self.assertEqual(output.past_key_values.get_seq_length(), 4)
        self.assertEqual(session.position, 13)

        session._forward(torch.tensor([[22]]), output.past_key_values)
        second = model.records[-1]
        self.assertEqual(second["cache_position"].tolist(), [4])
        self.assertEqual(second["position_ids"][:, 0, 0].tolist(), [13, 13, 13])

    def test_delete_is_default_and_initial_state_owns_only_selected_kv(self):
        seed = _seed()
        seed_tensors = [seed.prefix_ids, seed.image_mask, seed.image_score]
        seed_tensors.extend(tensor for pair in seed.kv for tensor in pair)
        seed_refs = [weakref.ref(tensor) for tensor in seed_tensors]
        seed_kv_pointers = {
            tensor.untyped_storage().data_ptr() for pair in seed.kv for tensor in pair
        }
        del seed_tensors

        model = _FakeCacheModel(predictions=(9, 0))
        with mock.patch.object(session_mod, "SessionTemplate", _FakeSessionTemplate):
            session = RecurrentSession(
                model,
                _FakeProcessor(),
                seed,
                "cpu",
                budget=2,
                condition="recurrent",
                image_floor=0.0,
                decay=0.5,
                n_sink=1,
            )

        self.assertEqual(session.storage, "delete")
        self.assertIsNone(session.cold)
        self.assertEqual(session.cache.get_seq_length(), 2)
        self.assertEqual(session.state.n_tokens, 2)
        self.assertEqual(session.active_indices.tolist(), [0, 1])
        self.assertEqual(session.initial_deleted_tokens, 3)
        for key, value in session.cache.to_legacy_cache():
            self.assertNotIn(key.untyped_storage().data_ptr(), seed_kv_pointers)
            self.assertNotIn(value.untyped_storage().data_ptr(), seed_kv_pointers)

        # Releasing the caller's seed must release every source tensor. The
        # compressed session owns copies rather than a hidden full-prefix alias.
        del seed
        gc.collect()
        self.assertTrue(all(reference() is None for reference in seed_refs))
        self.assertEqual(session.cache.to_legacy_cache()[0][0].flatten().tolist(), [0.0, 1.0])
        self.assertEqual(session.state.select()[1]["state_bytes"], 36)

        # FULL remains an explicitly uncompressed comparator, but it too owns
        # a resident cache rather than a second CPU ColdKV backup.
        full_model = _FakeCacheModel(predictions=(9, 0))
        with mock.patch.object(session_mod, "SessionTemplate", _FakeSessionTemplate):
            full = RecurrentSession(
                full_model,
                _FakeProcessor(),
                _seed(),
                "cpu",
                budget=2,
                condition="full",
                image_floor=0.0,
                decay=0.5,
                n_sink=1,
            )
        self.assertIsNone(full.cold)
        self.assertEqual(full.cache.get_seq_length(), 5)
        self.assertEqual(full.active_indices.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(full.initial_deleted_tokens, 0)

    def test_delete_turns_compact_cache_state_and_never_recall_evicted_ids(self):
        model, session = _make_session(
            predictions=(7, 9, 0, 8, 9, 0, 12, 9, 0),
            budget=3,
            storage="delete",
        )
        self.assertEqual(session.active_indices.tolist(), [0, 1, 2])
        self.assertEqual(session.cache.get_seq_length(), 3)
        self.assertEqual(session.state.n_tokens, 3)

        first = session.answer("q1", max_new_tokens=4)
        first_candidates = {0, 1, 2, *range(5, 10)}
        self.assertTrue(set(first["next_active_indices"]) <= first_candidates)
        self.assertEqual(first["next_active_indices"], [0, 1, 5])
        self.assertEqual(first["peak_active_kv_tokens"], 8)
        self.assertEqual(first["retained_kv_tokens"], 3)
        self.assertEqual(first["resident_gpu_kv_bytes"], 3 * session.token_bytes)
        self.assertEqual(first["cold_kv_bytes"], 0)
        self.assertEqual(first["deleted_tokens_this_turn"], 5)
        self.assertEqual(first["selector_state_bytes"], 54)
        self.assertEqual(session.cache.get_seq_length(), 3)
        self.assertEqual(session.state.n_tokens, 3)

        second = session.answer("q2", max_new_tokens=4)
        second_candidates = {*first["next_active_indices"], *range(10, 16)}
        self.assertTrue(set(second["next_active_indices"]) <= second_candidates)
        self.assertEqual(second["next_active_indices"], [0, 10, 11])
        self.assertNotIn(2, second["next_active_indices"])
        self.assertEqual(session.cache.get_seq_length(), 3)
        self.assertEqual(session.state.n_tokens, 3)

        third = session.answer("q3", max_new_tokens=4)
        third_candidates = {*second["next_active_indices"], *range(16, 22)}
        self.assertTrue(set(third["next_active_indices"]) <= third_candidates)
        self.assertEqual(third["next_active_indices"], [0, 16, 17])
        # IDs 1, 2, and 5 have all been physically deleted by this point; none
        # can return despite having held image/history score in an earlier turn.
        self.assertTrue({1, 2, 5}.isdisjoint(third["next_active_indices"]))
        self.assertEqual(session.cache.get_seq_length(), 3)
        self.assertEqual(session.state.n_tokens, 3)
        self.assertEqual(session.total_seen, 22)
        self.assertEqual(session.position, 28)

        # Physical slots restart from B after every compaction, while mRoPE
        # positions continue globally across all tokens that were processed.
        self.assertEqual(model.records[0]["cache_position"].tolist(), [3, 4])
        self.assertEqual(model.records[3]["cache_position"].tolist(), [3, 4, 5])
        self.assertEqual(model.records[6]["cache_position"].tolist(), [3, 4, 5])
        self.assertEqual(model.records[0]["position_ids"][0, 0].tolist(), [11, 12])
        self.assertEqual(model.records[3]["position_ids"][0, 0].tolist(), [16, 17, 18])
        self.assertEqual(model.records[6]["position_ids"][0, 0].tolist(), [22, 23, 24])
        self.assertTrue(all(record["pixel_values"] is None for record in model.records))

    def test_recurrent_turn_commits_own_tokens_then_reselects_exact_budget(self):
        # Three slots leave one lower-prior image position that first-turn
        # interaction evidence can displace; the strongest image position stays.
        model, session = _make_session(predictions=(7, 9, 0, 8, 9, 0), budget=3)
        initial_active = session.active_indices.tolist()
        first = session.answer("first unseen question", max_new_tokens=4)

        self.assertEqual(first["active_indices"], initial_active)
        self.assertEqual(first["prediction"], "7")
        self.assertEqual(first["generated_tokens"], 1)
        self.assertEqual(first["new_session_tokens"], 5)  # suffix + answer + closing delimiter
        self.assertEqual(session.cold.length, 10)
        self.assertEqual(first["next_active_history_tokens"], 3)
        self.assertEqual(first["entered_tokens"], 1)
        self.assertEqual(first["evicted_tokens"], 1)
        self.assertEqual(first["selection_after"]["selected_history_text_tokens"], 1)
        self.assertEqual(first["next_active_indices"], [0, 1, 5])
        # Values encode the model's actual input IDs, proving that the generated
        # answer and template delimiter, rather than a reference answer, entered cold KV.
        self.assertEqual(session.cold.kv[0][1][..., -5:, :].flatten().tolist(), [1004, 1005, 1007, 1009, 1010])

        second = session.answer("different unseen question", max_new_tokens=4)
        self.assertEqual(second["active_indices"], first["next_active_indices"])
        self.assertEqual(second["new_session_tokens"], 6)
        self.assertEqual(second["next_active_history_tokens"], 3)
        # Unobserved cold positions retain their prior interaction state, so
        # the next selection may promote them even though they were not hot in
        # this turn. They are still positions from this session's own first turn.
        self.assertEqual(second["next_active_indices"], [0, 6, 7])
        self.assertEqual(session.cold.length, 16)
        self.assertEqual(session.position, 22)
        self.assertEqual(session.template.calls, [("first unseen question", True), ("different unseen question", False)])
        self.assertEqual(len(model.records), 6)

    def test_conditions_have_independent_rollouts_and_expected_hot_sets(self):
        shared_seed = _seed()
        full_model, full = _make_session("full", predictions=(7, 9, 0), seed=shared_seed)
        static_model, static = _make_session("image_static", predictions=(6, 9, 0), seed=shared_seed)
        recurrent_model, recurrent = _make_session("recurrent", predictions=(8, 9, 0), seed=shared_seed)

        full_result = full.answer("same question")
        static_result = static.answer("same question")
        recurrent_result = recurrent.answer("same question")

        self.assertEqual(full_result["active_history_tokens"], 5)
        self.assertEqual(full_result["next_active_history_tokens"], 10)
        self.assertEqual(full_result["selection_after"]["selected_history_text_tokens"], 5)
        self.assertEqual(static_result["active_history_tokens"], 2)
        self.assertEqual(static_result["next_active_indices"], [0, 1])
        self.assertEqual(static_result["selection_after"]["selected_history_text_tokens"], 0)
        self.assertEqual(recurrent_result["active_history_tokens"], 2)
        self.assertEqual(recurrent_result["next_active_history_tokens"], 2)
        self.assertEqual(full_result["prediction"], "7")
        self.assertEqual(static_result["prediction"], "6")
        self.assertEqual(recurrent_result["prediction"], "8")
        self.assertEqual(shared_seed.kv[0][0].shape[-2], 5)
        self.assertEqual(full.cold.length, static.cold.length)
        self.assertEqual(static.cold.length, recurrent.cold.length)
        self.assertIsNot(full.cold.kv[0][0], static.cold.kv[0][0])
        self.assertEqual(len(full_model.records), 3)
        self.assertEqual(len(static_model.records), 3)
        self.assertEqual(len(recurrent_model.records), 3)

    def test_termination_policy_records_and_normalizes_alternate_eos(self):
        _, alternate = _make_session(
            "full", predictions=(11, 0), eos=[9, 11]
        )
        stopped = alternate.answer("alternate stop")
        self.assertEqual(stopped["prediction"], "")
        self.assertEqual(stopped["predicted_stop_token_id"], 11)
        self.assertFalse(stopped["hit_generation_limit"])
        self.assertEqual(
            stopped["termination_policy"],
            "canonical_assistant_ending; generated_content_ids_preserved",
        )
        # The alternate stop token is deliberately not cached. The canonical
        # template ending [9, 10] follows the exact generated content instead.
        self.assertEqual(
            alternate.cold.kv[0][1][..., -4:, :].flatten().tolist(),
            [1004, 1005, 1009, 1010],
        )

        _, limited = _make_session(
            "full", predictions=(7, 0), eos=[9, 11]
        )
        capped = limited.answer("generation cap", max_new_tokens=1)
        self.assertEqual(capped["prediction"], "7")
        self.assertIsNone(capped["predicted_stop_token_id"])
        self.assertTrue(capped["hit_generation_limit"])
        self.assertEqual(
            limited.cold.kv[0][1][..., -5:, :].flatten().tolist(),
            [1004, 1005, 1007, 1009, 1010],
        )

    def test_prefill_is_one_prefix_forward_and_masks_score_to_visual_tokens(self):
        model = _FakeCacheModel(predictions=(0,))
        inputs = {
            "input_ids": torch.tensor([[90, 99, 99, 91, 4, 5]]),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "pixel_values": torch.ones(4, 3),
        }
        positions = torch.tensor(
            [[[0, 1, 1, 2]], [[0, 0, 1, 2]], [[0, 0, 1, 2]]], dtype=torch.long
        )
        with (
            mock.patch("vlm_diagnosis.core.signals.vlm_inputs", return_value=inputs),
            mock.patch.object(
                adapter_mod,
                "token_spans",
                return_value={"visual": torch.tensor([1, 2]), "vis_end": 2, "L": 6},
            ),
            mock.patch.object(adapter_mod, "mrope_position_ids", return_value=positions),
        ):
            seed = prefill_image(model, _FakeProcessor(), object(), "cpu")

        self.assertEqual(len(model.records), 1)
        self.assertEqual(model.records[0]["input_ids"].tolist(), [[90, 99, 99, 91]])
        self.assertIs(model.records[0]["pixel_values"], inputs["pixel_values"])
        self.assertEqual(seed.prefix_ids.tolist(), [[90, 99, 99, 91]])
        self.assertEqual(seed.kv[0][0].shape[-2], 4)
        self.assertEqual(seed.next_position, 3)
        self.assertEqual(seed.image_mask.tolist(), [False, True, True, False])
        self.assertEqual(seed.image_score[[0, 3]].tolist(), [0.0, 0.0])
        self.assertTrue(bool((seed.image_score[1:3] > 0).all()))

    def test_session_template_slices_first_image_and_later_turn_suffixes(self):
        class Tokenizer:
            mapping = {
                "A session assistant anchor END": [1, 2, 3, 4],
                " END": [4],
                "FIRST": [10, 99, 11, 20, 21],
                "LATER": [1, 2, 3, 4, 30, 31],
            }

            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=torch.tensor([self.mapping[text]]))

        class Processor:
            tokenizer = Tokenizer()

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                if len(messages) == 2:
                    return "A session assistant anchor END"
                if isinstance(messages[0]["content"], list):
                    return "FIRST"
                return "LATER"

        template = SessionTemplate(
            Processor(), image_token_id=99, prefix_ids=torch.tensor([[10, 99, 99, 99, 11]])
        )
        self.assertEqual(template.ending_ids.tolist(), [[4]])
        self.assertEqual(template.suffix("q1", first=True).tolist(), [[20, 21]])
        self.assertEqual(template.suffix("q2", first=False).tolist(), [[30, 31]])


if __name__ == "__main__":
    unittest.main()
