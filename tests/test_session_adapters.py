from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
from transformers import DynamicCache

from vlm_diagnosis.core import session_adapters as adapters_mod
from vlm_diagnosis.core import signals as signals_mod
from vlm_diagnosis.core.session_adapters import (
    QwenImageAdapter,
    QwenImageTemplate,
    SessionAdapter,
)
from vlm_diagnosis.core.session_types import SessionInput, SessionSeed


def _legacy_kv(length, layers=2):
    result = []
    for layer_idx in range(layers):
        base = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1)
        result.append((base + 100 * layer_idx, base + 1000 + 100 * layer_idx))
    return tuple(result)


class _FakeAttention(nn.Module):
    def forward(self, weights):
        hidden = torch.zeros(
            weights.shape[0], weights.shape[-2], weights.shape[1],
            dtype=weights.dtype,
        )
        return hidden, weights


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeCore(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(layers)])


class _FakeModel(nn.Module):
    """Tiny decoder whose K records logical position and V records token ID."""

    def __init__(self, layers=2, eos=9):
        super().__init__()
        self.model = _FakeCore(layers)
        self.config = SimpleNamespace(image_token_id=99)
        self.generation_config = SimpleNamespace(eos_token_id=eos)
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
        old_length = past_key_values.get_seq_length()
        token_count = input_ids.shape[1]
        if cache_position is None:
            cache_position = torch.arange(
                old_length, old_length + token_count, device=input_ids.device
            )
        self.records.append(
            {
                "input_ids": input_ids.detach().cpu().clone(),
                "position_ids": position_ids.detach().cpu().clone(),
                "attention_mask": attention_mask.detach().cpu().clone(),
                "cache_position": cache_position.detach().cpu().clone(),
                "old_length": old_length,
                "pixel_values": kwargs.get("pixel_values"),
                "image_grid_thw": kwargs.get("image_grid_thw"),
                "kwargs": dict(kwargs),
            }
        )

        logical = position_ids[0].reshape(1, 1, token_count, 1).float()
        values = input_ids.reshape(1, 1, token_count, 1).float()
        total = old_length + token_count
        for layer_idx, layer in enumerate(self.model.layers):
            past_key_values.update(
                logical + 100 * layer_idx,
                values + 1000 + 100 * layer_idx,
                layer_idx,
            )
            weights = torch.zeros(1, 2, token_count, total)
            for row in range(token_count):
                weights[:, :, row, old_length + row] = 1.0
            layer.self_attn(weights)

        vocabulary = max(128, int(input_ids.max()) + 1)
        logits = torch.zeros(1, token_count, vocabulary)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class _FakeTokenizer:
    mapping = {
        "A session assistant anchor END": [1, 2, 3, 4],
        " END": [4],
        "FIRST": [10, 99, 11, 20, 21],
        "LATER": [1, 2, 3, 4, 30, 31],
    }

    def __init__(self):
        self.decode_calls = []

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([self.mapping[text]]))

    def decode(self, tokens, skip_special_tokens=True):
        self.decode_calls.append((list(tokens), skip_special_tokens))
        return "  " + " ".join(str(token) for token in tokens) + "  "


class _FakeProcessor:
    image_token_id = 99

    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if len(messages) == 2:
            return "A session assistant anchor END"
        if isinstance(messages[0]["content"], list):
            return "FIRST"
        return "LATER"


def _seed(adapter_id="qwen_image_v1"):
    return SessionSeed(
        kv=_legacy_kv(5),
        prefix_ids=torch.tensor([[10, 99, 99, 99, 11]]),
        prior_scores=torch.tensor([0.0, 0.9, 0.5, 0.1, 0.0]),
        modality_ids=torch.tensor([0, 2, 2, 2, 0]),
        next_position=11,
        prefill_seconds=0.0,
        modality_names={0: "control", 1: "text", 2: "image"},
        token_features={},
        adapter_id=adapter_id,
    )


class SessionAdapterContractTest(unittest.TestCase):
    def test_interface_is_abstract_and_qwen_adapter_is_stateless(self):
        with self.assertRaises(TypeError):
            SessionAdapter()
        adapter = QwenImageAdapter()
        self.assertEqual(adapter.adapter_id, "qwen_image_v1")
        self.assertEqual(adapter.supported_modalities, ("control", "text", "image"))
        self.assertEqual(vars(adapter), {})

    def test_template_and_inputs_preserve_legacy_chat_boundaries(self):
        adapter = QwenImageAdapter()
        processor = _FakeProcessor()
        seed = _seed()
        template = adapter.make_template(processor, seed)

        self.assertIsInstance(template, QwenImageTemplate)
        self.assertEqual(template.ending_ids.tolist(), [[4]])
        first = adapter.prepare_turn(template, "q1", first=True)
        later = adapter.prepare_turn(template, "q2", first=False)
        self.assertEqual(first.input_ids.tolist(), [[20, 21]])
        self.assertEqual(later.input_ids.tolist(), [[30, 31]])
        self.assertEqual(first.modality_ids.tolist(), [1, 1])
        self.assertEqual(later.modality_ids.tolist(), [1, 1])

        ending = adapter.text_input(template.ending_ids, kind="ending")
        control = adapter.text_input(torch.tensor([[7]]), kind="control")
        self.assertEqual(ending.modality_ids.tolist(), [1])
        self.assertEqual(control.modality_ids.tolist(), [0])

        # Both the template and SessionInput own their small metadata tensors.
        seed.prefix_ids.fill_(0)
        template.ending_ids.fill_(8)
        self.assertEqual(template.prefix_ids.tolist(), [[10, 99, 99, 99, 11]])
        self.assertEqual(ending.input_ids.tolist(), [[4]])
        self.assertEqual(vars(adapter), {})
        self.assertFalse(any(name in vars(template) for name in ("kv", "seed", "cache")))

    def test_template_rejects_incompatible_seed_processor_and_requests(self):
        adapter = QwenImageAdapter()
        processor = _FakeProcessor()
        with self.assertRaises(ValueError):
            adapter.make_template(processor, _seed(adapter_id="different_adapter"))

        mismatched = _FakeProcessor()
        mismatched.image_token_id = 98
        with self.assertRaises(ValueError):
            adapter.make_template(mismatched, _seed())

        mislabeled = _seed()
        mislabeled.prefix_ids[0, 0] = 99
        with self.assertRaises(ValueError):
            adapter.make_template(processor, mislabeled)

        context_sensitive = _FakeProcessor()
        context_sensitive.tokenizer.mapping = dict(_FakeTokenizer.mapping)
        context_sensitive.tokenizer.mapping[" END"] = [44]
        with self.assertRaises(ValueError):
            adapter.make_template(context_sensitive, _seed())

        template = adapter.make_template(processor, _seed())
        for request in ({"type": "image"}, ["q"], object()):
            with self.subTest(request=type(request).__name__), self.assertRaises(TypeError):
                adapter.prepare_turn(template, request, first=True)
        with self.assertRaises(TypeError):
            adapter.prepare_turn(template, "q", first=1)
        for kind in ("image", "audio", "video", "sensor"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                adapter.text_input(torch.tensor([[1]]), kind=kind)

    def test_prefill_encodes_only_image_prefix_and_masks_prior_to_image(self):
        adapter = QwenImageAdapter()
        processor = _FakeProcessor()
        model = _FakeModel()
        image = object()
        pixels = torch.tensor([[1.0, 2.0]])
        grid = torch.tensor([[1, 1, 2]])
        inputs = {
            "input_ids": torch.tensor([[90, 99, 99, 91, 70, 71]]),
            "pixel_values": pixels,
            "image_grid_thw": grid,
        }
        positions = torch.tensor(
            [[[0, 1, 2, 3]], [[0, 1, 2, 3]], [[0, 1, 4, 5]]],
            dtype=torch.long,
        )
        vlm = mock.Mock(return_value=inputs)
        with (
            mock.patch.object(signals_mod, "vlm_inputs", vlm),
            mock.patch.object(
                adapters_mod,
                "token_spans",
                return_value={
                    "visual": torch.tensor([1, 2]),
                    "vis_end": 2,
                    "sink": torch.arange(4),
                    "L": 6,
                },
            ),
            mock.patch.object(adapters_mod, "mrope_position_ids", return_value=positions),
        ):
            seed = adapter.prefill(model, processor, image, "cpu")

        vlm.assert_called_once_with(processor, image, "x", "cpu")
        self.assertEqual(len(model.records), 1)
        record = model.records[0]
        self.assertEqual(record["input_ids"].tolist(), [[90, 99, 99, 91]])
        torch.testing.assert_close(record["position_ids"], positions)
        self.assertEqual(record["attention_mask"].tolist(), [[1, 1, 1, 1]])
        self.assertIs(record["pixel_values"], pixels)
        self.assertIs(record["image_grid_thw"], grid)

        self.assertEqual(seed.prefix_ids.tolist(), [[90, 99, 99, 91]])
        self.assertEqual(seed.modality_ids.tolist(), [0, 2, 2, 0])
        torch.testing.assert_close(
            seed.prior_scores, torch.tensor([0.0, 0.5, 0.5, 0.0])
        )
        self.assertEqual(seed.modality_names, {0: "control", 1: "text", 2: "image"})
        self.assertEqual(seed.token_features, {})
        self.assertEqual(seed.adapter_id, adapter.adapter_id)
        self.assertEqual(seed.next_position, 6)
        self.assertGreaterEqual(seed.prefill_seconds, 0.0)
        self.assertEqual(len(seed.kv), 2)
        self.assertTrue(all(key.device.type == "cpu" for pair in seed.kv for key in pair))
        self.assertEqual(vars(adapter), {})

    def test_prefill_rejects_non_image_context_and_incomplete_processor_output(self):
        adapter = QwenImageAdapter()
        model = _FakeModel()
        processor = _FakeProcessor()
        for context in (None, "path.jpg", b"pixels", [], (), {}):
            with self.subTest(context=type(context).__name__), self.assertRaises(TypeError):
                adapter.prefill(model, processor, context, "cpu")

        incomplete = {
            "input_ids": torch.tensor([[90, 99, 91]]),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }
        with (
            mock.patch.object(signals_mod, "vlm_inputs", return_value=incomplete),
            self.assertRaises(ValueError),
        ):
            adapter.prefill(model, processor, object(), "cpu")

    def test_forward_separates_logical_mrope_from_physical_cache_positions(self):
        adapter = QwenImageAdapter()
        model = _FakeModel()
        cache = DynamicCache.from_legacy_cache(_legacy_kv(2))
        prepared = adapter.text_input(torch.tensor([[7, 8]]))

        with adapter.observe(model) as capture:
            output, next_position = adapter.forward(
                model, prepared, cache, position=11, device="cpu"
            )

        self.assertIs(output.past_key_values, cache)
        self.assertEqual(cache.get_seq_length(), 4)
        self.assertEqual(next_position, 13)
        record = model.records[-1]
        expected_logical = torch.tensor(
            [[[11, 12]], [[11, 12]], [[11, 12]]], dtype=torch.long
        )
        torch.testing.assert_close(record["position_ids"], expected_logical)
        self.assertEqual(record["cache_position"].tolist(), [2, 3])
        self.assertEqual(record["attention_mask"].tolist(), [[1, 1, 1, 1]])
        torch.testing.assert_close(
            capture.mean(), torch.tensor([0.0, 0.0, 0.5, 0.5])
        )

    def test_forward_validates_explicit_mrope_before_mutating_cache(self):
        adapter = QwenImageAdapter()
        model = _FakeModel()
        explicit = torch.tensor(
            [[[100, 101]], [[100, 101]], [[100, 101]]], dtype=torch.long
        )
        prepared = SessionInput(
            input_ids=torch.tensor([[7, 8]]),
            modality_ids=torch.tensor([1, 1]),
            position_ids=explicit,
            next_position=102,
            model_kwargs={"adapter_probe": torch.tensor(1)},
        )
        cache = DynamicCache.from_legacy_cache(_legacy_kv(1))
        _, next_position = adapter.forward(model, prepared, cache, 100, "cpu")
        self.assertEqual(next_position, 102)
        torch.testing.assert_close(model.records[-1]["position_ids"], explicit)
        self.assertEqual(model.records[-1]["cache_position"].tolist(), [1, 2])
        self.assertIn("adapter_probe", model.records[-1]["kwargs"])

        untouched = DynamicCache.from_legacy_cache(_legacy_kv(1))
        calls_before = len(model.records)
        with self.assertRaises(ValueError):
            adapter.forward(model, prepared, untouched, 200, "cpu")
        self.assertEqual(untouched.get_seq_length(), 1)
        self.assertEqual(len(model.records), calls_before)

        image_input = SessionInput(
            input_ids=torch.tensor([[99]]), modality_ids=torch.tensor([2])
        )
        with self.assertRaises(ValueError):
            adapter.forward(model, image_input, cache, 30, "cpu")

        wrong_shape = SessionInput(
            input_ids=torch.tensor([[1, 2]]),
            modality_ids=torch.tensor([1, 1]),
            position_ids=torch.tensor([[[1, 2]]]),
        )
        with self.assertRaises(ValueError):
            adapter.forward(
                model,
                wrong_shape,
                DynamicCache.from_legacy_cache(_legacy_kv(1)),
                30,
                "cpu",
            )

        backwards = SessionInput(
            input_ids=torch.tensor([[1, 2]]),
            modality_ids=torch.tensor([1, 1]),
            position_ids=explicit,
            next_position=25,
        )
        with self.assertRaises(ValueError):
            adapter.forward(
                model,
                backwards,
                DynamicCache.from_legacy_cache(_legacy_kv(1)),
                30,
                "cpu",
            )

    def test_decode_stop_ids_and_metadata_are_adapter_owned(self):
        adapter = QwenImageAdapter()
        processor = _FakeProcessor()
        template = adapter.make_template(processor, _seed())
        self.assertEqual(adapter.decode(processor, torch.tensor([7, 8])), "7 8")
        self.assertEqual(processor.tokenizer.decode_calls, [([7, 8], True)])
        with self.assertRaises(ValueError):
            adapter.decode(processor, torch.tensor([[7, 8]]))

        model = _FakeModel(eos=9)
        self.assertEqual(adapter.stop_token_ids(model), {9})
        model.generation_config.eos_token_id = [9, 10, 9]
        self.assertEqual(adapter.stop_token_ids(model), {9, 10})
        model.generation_config.eos_token_id = None
        self.assertEqual(adapter.stop_token_ids(model), set())
        model.generation_config.eos_token_id = [-1]
        with self.assertRaises(ValueError):
            adapter.stop_token_ids(model)
        for malformed in ([False], [1.5], ["3"], "12"):
            model.generation_config.eos_token_id = malformed
            with self.subTest(eos=malformed), self.assertRaises(ValueError):
                adapter.stop_token_ids(model)

        expected_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (template.prefix_ids, template.anchor_ids, template.ending_ids)
        )
        self.assertEqual(adapter.metadata_bytes(template), expected_bytes)
        with self.assertRaises(TypeError):
            adapter.metadata_bytes(object())


if __name__ == "__main__":
    unittest.main()
