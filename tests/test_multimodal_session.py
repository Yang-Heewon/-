"""Synthetic mixed-modality contract tests, not audio/video model validation."""
import gc
from types import SimpleNamespace
import unittest
import weakref

import torch

from vlm_diagnosis.core.session_adapters import SessionAdapter
from vlm_diagnosis.core.session_cache import AttentionMass, MultimodalSession
from vlm_diagnosis.core.session_types import SessionSeed, SessionInput, TokenFeatures
from test_recurrent_session import _FakeCacheModel, _FakeProcessor, _legacy_kv


class SyntheticAdapter(SessionAdapter):
    adapter_id = "synthetic_multimodal_test_v1"
    supported_modalities = ("control", "text", "image", "audio", "video", "sensor")

    def prefill(self, model, processor, context, device):
        raise NotImplementedError("synthetic test seed only; no native media encoder")

    def make_template(self, processor, seed):
        return SimpleNamespace(prefix_ids=seed.prefix_ids.clone(),
                               anchor_ids=torch.empty(1, 0, dtype=torch.long),
                               ending_ids=torch.tensor([[9, 10]]))

    def prepare_turn(self, template, request, first):
        if not isinstance(request, SessionInput):
            raise TypeError("synthetic requests must already be prepared")
        return request

    def text_input(self, ids, kind="text"):
        return SessionInput(ids, torch.full((ids.shape[1],), 0 if kind == "ending" else 1))

    def forward(self, model, prepared, cache, position, device):
        ids = prepared.input_ids.to(device)
        n, old = ids.shape[1], cache.get_seq_length()
        # Deliberately not Qwen's (3, 1, n) mRoPE shape. Positions are an
        # adapter concern, and may advance differently from physical slots.
        positions = torch.arange(position, position + n)[None]
        out = model(input_ids=ids, position_ids=positions,
                    cache_position=torch.arange(old, old+n),
                    attention_mask=torch.ones(1, old+n), past_key_values=cache)
        return out, position + n + 1

    def observe(self, model):
        return AttentionMass(model)

    def decode(self, processor, tokens):
        return processor.tokenizer.decode(tokens)

    def stop_token_ids(self, model):
        return {9}

    def metadata_bytes(self, template):
        return sum(t.numel() * t.element_size() for t in (
            template.prefix_ids, template.anchor_ids, template.ending_ids))


def make_seed():
    return SessionSeed(
        kv=_legacy_kv(5), prefix_ids=torch.tensor([[90, 99, 80, 81, 20]]),
        prior_scores=torch.tensor([0., 1., .8, .6, 0.]),
        modality_ids=torch.tensor([0, 2, 3, 3, 1]),
        next_position=11, prefill_seconds=0.,
        token_features={"coordinate": torch.arange(10).view(5, 2),
                        "time": torch.arange(5, dtype=torch.float32)},
        adapter_id=SyntheticAdapter.adapter_id)


def make_session(seed=None, storage="delete", condition="recurrent"):
    model = _FakeCacheModel(predictions=(7, 9, 0, 8, 9, 0, 12, 9, 0))
    session = MultimodalSession(
        model, _FakeProcessor(), seed or make_seed(), "cpu", budget=3,
        adapter=SyntheticAdapter(), n_sink=1, prior_floor=0., decay=.5,
        storage=storage, condition=condition)
    return model, session


class MultimodalSessionTest(unittest.TestCase):
    def test_delete_mixed_cache_and_metadata_without_seed_alias(self):
        seed = make_seed()
        pointers = {t.untyped_storage().data_ptr() for pair in seed.kv for t in pair}
        refs = [weakref.ref(t) for pair in seed.kv for t in pair]
        feature_refs = [weakref.ref(t) for t in seed.token_features.values()]
        model, session = make_session(seed)
        del seed
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs + feature_refs))
        self.assertEqual(session.cache.get_seq_length(), 3)
        self.assertEqual(session.state.n_tokens, 3)
        self.assertEqual(session.features.length, 3)
        self.assertEqual(session.features.nbytes, 3 * (2 * 8 + 4))
        self.assertEqual(session._selection_snapshot()["selected_tokens_by_modality"]["audio"], 1)
        for pair in session.cache.to_legacy_cache():
            for tensor in pair:
                self.assertNotIn(tensor.untyped_storage().data_ptr(), pointers)
                self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())

        previous = set(session.active_indices.tolist())
        for modality in (3, 4, 5):
            old_total = session.total_seen
            result = session.answer(SessionInput(torch.tensor([[5, 6]]),
                                                  torch.tensor([modality, 1])))
            allowed = previous | set(range(old_total, result["logical_history_tokens_after"]))
            next_ids = set(result["next_active_indices"])
            self.assertTrue(next_ids <= allowed)
            self.assertEqual(result["active_indices"], sorted(previous))
            self.assertEqual(result["retained_kv_tokens"], 3)
            self.assertEqual(result["selector_state_bytes"], 3 * 18)
            self.assertEqual(result["token_feature_bytes"], 3 * 20)
            for field in ("cold_kv_bytes", "h2d_kv_bytes", "d2h_new_kv_bytes"):
                self.assertEqual(result[field], 0)
            self.assertEqual(session.state.n_tokens, session.features.length)
            self.assertEqual(session.features.length, session.cache.get_seq_length())
            actual = {name: int((session.state.modality_ids == code).sum())
                      for code, name in session.modality_names.items()}
            self.assertEqual(result["selection_after"]["selected_tokens_by_modality"], actual)
            self.assertEqual(sum(result["deleted_tokens_by_modality"].values()),
                             result["deleted_tokens_this_turn"])
            self.assertEqual(result["persistent_session_tensor_bytes"],
                             result["retained_kv_bytes"] + result["selector_state_bytes"]
                             + result["session_metadata_bytes"])
            for t in session.features.tensors.values():
                self.assertEqual(t.untyped_storage().nbytes(), t.numel() * t.element_size())
            previous = next_ids
        self.assertTrue(all(r["position_ids"].ndim == 2 for r in model.records))

    def test_rejected_input_leaves_live_cache_untouched(self):
        model, session = make_session()
        cache, position = session.cache, session.position
        invalid = [SessionInput(torch.tensor([[5]]), torch.tensor([99])),
                   SessionInput(torch.tensor([[5]]), torch.tensor([3]),
                                token_features={"undeclared": torch.ones(1)}),
                   SessionInput(torch.tensor([[5]]), torch.tensor([3]),
                                token_features={"coordinate": torch.ones(1, 3)})]
        for request in invalid:
            with self.assertRaises(ValueError):
                session.answer(request)
            self.assertIs(session.cache, cache)
            self.assertEqual(session.position, position)
            self.assertEqual(model.records, [])
        bad_seed = make_seed()
        bad_seed.adapter_id = "some_other_model"
        with self.assertRaises(ValueError):
            make_session(bad_seed)

    def test_offload_and_full_preserve_typed_history(self):
        for storage, condition in (("offload", "recurrent"), ("delete", "full")):
            with self.subTest(storage=storage, condition=condition):
                _, session = make_session(storage=storage, condition=condition)
                result = session.answer(SessionInput(torch.tensor([[5, 6]]), torch.tensor([3, 4])))
                self.assertEqual(session.features.length, session.total_seen)
                self.assertEqual(session.state.n_tokens, session.total_seen)
                self.assertEqual(result["deleted_tokens_this_turn"], 0)
                self.assertEqual(sum(result["deleted_tokens_by_modality"].values()), 0)
                if storage == "offload":
                    self.assertEqual(session.cold.length, session.total_seen)
                    self.assertIsNone(session.cache)
                else:
                    counts = result["selection_after"]["selected_tokens_by_modality"]
                    self.assertEqual(counts["audio"], 3)
                    self.assertEqual(counts["video"], 1)
                    self.assertEqual(counts["text"], 2)  # one seed + generated answer

    def test_token_features_append_and_retain_are_aligned_and_atomic(self):
        features = TokenFeatures({"xy": torch.arange(6).view(3, 2)}, 3)
        source = features.tensors["xy"]
        features.retain(torch.tensor([0, 2]))
        self.assertNotEqual(source.data_ptr(), features.tensors["xy"].data_ptr())
        features.append({}, 1)
        self.assertEqual(features.tensors["xy"].tolist(), [[0, 1], [4, 5], [-1, -1]])
        with self.assertRaises(ValueError):
            features.append({"xy": torch.zeros(1, 3)}, 1)
        self.assertEqual(features.length, 3)

    def test_input_contract_rejects_overflow_positions_and_cache_override(self):
        for kwargs in (
            {"prior_scores": torch.tensor([1e100], dtype=torch.float64)},
            {"position_ids": torch.tensor(1)},
            {"next_position": True},
            {"model_kwargs": {"past_key_values": object()}},
        ):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                SessionInput(torch.tensor([[5]]), torch.tensor([3]), **kwargs)


if __name__ == "__main__":
    unittest.main()
