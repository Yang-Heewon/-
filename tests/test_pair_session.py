"""End-to-end physical pair-session tests on a real tiny Qwen decoder."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLTextModel,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from vlm_diagnosis.core.pair_session import PairSession
from vlm_diagnosis.core.session_adapters import (
    QwenImageTemplate,
    QwenPairAdapter,
)
from vlm_diagnosis.core.session_types import SessionSeed


class _TinyCausalLM(nn.Module):
    """A real Qwen text stack plus a deterministic zeroed LM logits head."""

    def __init__(self, config_type, model_type):
        super().__init__()
        config = config_type(
            vocab_size=128,
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
        config.image_token_id = 99
        self.model = model_type(config).eval()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        self.config = config
        # All logits tie at zero, so greedy argmax immediately emits EOS 0.
        self.generation_config = SimpleNamespace(eos_token_id=0)
        self.eval()

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        return SimpleNamespace(
            logits=self.lm_head(output.last_hidden_state),
            past_key_values=output.past_key_values,
        )


class _FakeQwenTemplate(QwenImageTemplate):
    """Qwen protocol-shaped template without a real processor dependency."""

    def __init__(self, prefix_ids):
        # Deliberately bypass QwenImageTemplate's tokenizer construction.
        self.prefix_ids = prefix_ids.detach().cpu().clone()
        self.anchor_ids = torch.tensor([[70]], dtype=torch.long)
        self.ending_ids = torch.tensor([[71]], dtype=torch.long)
        self.calls = []

    def suffix(self, question, first):
        if not isinstance(question, str) or not isinstance(first, bool):
            raise TypeError("fake template expects a text question and bool first flag")
        self.calls.append((question, first))
        base = 72 if first else 74
        return torch.tensor([[base, base + 1]], dtype=torch.long)


class _Tokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(map(str, tokens))


class _Processor:
    tokenizer = _Tokenizer()


def _make_model(family="qwen25"):
    if family == "qwen25":
        return _TinyCausalLM(Qwen2_5_VLTextConfig, Qwen2_5_VLTextModel)
    return _TinyCausalLM(Qwen3VLTextConfig, Qwen3VLTextModel)


@torch.no_grad()
def _make_seed(model):
    prefix_ids = torch.tensor([[10, 99, 99, 99, 11, 12]])
    positions = torch.arange(prefix_ids.shape[1])[None, None].expand(3, 1, -1)
    dense = model.model(
        input_ids=prefix_ids,
        position_ids=positions,
        use_cache=True,
    )
    kv = tuple(
        (key.detach().cpu(), value.detach().cpu())
        for key, value in dense.past_key_values.to_legacy_cache()
    )

    # A tiny random perturbation makes these real-valued pair priors, while
    # separated bands force an auditable non-token-common global selection:
    # flattened group counts are exactly [3, 2, 1, 0] at budget six.
    generator = torch.Generator().manual_seed(33)
    pair_prior = torch.rand(2, 2, 6, generator=generator) * 1e-3
    pair_prior[:, :, [0, 4, 5]] = 0.0
    pair_prior[0, 0, [1, 2, 3]] += torch.tensor([9.0, 8.0, 7.0])
    pair_prior[0, 1, [1, 2]] += torch.tensor([6.0, 5.0])
    pair_prior[1, 0, 3] += 4.0
    pair_prior[1, 1] = 0.0
    modalities = torch.tensor([0, 2, 2, 2, 0, 0], dtype=torch.long)
    return SessionSeed(
        kv=kv,
        prefix_ids=prefix_ids,
        prior_scores=pair_prior.mean((0, 1)),
        pair_prior_scores=pair_prior,
        modality_ids=modalities,
        next_position=10,
        prefill_seconds=0.0,
        modality_names={0: "control", 1: "text", 2: "image"},
        adapter_id=QwenPairAdapter.adapter_id,
    )


def _make_session(model, condition):
    seed = _make_seed(model)
    adapter = QwenPairAdapter()
    template = _FakeQwenTemplate(seed.prefix_ids)
    with mock.patch.object(adapter, "make_template", return_value=template) as make:
        session = PairSession(
            model,
            _Processor(),
            seed,
            "cpu",
            budget_pairs=6,
            condition=condition,
            prior_floor=0.0,
            decay=0.5,
            n_sink=0,
            adapter=adapter,
        )
    make.assert_called_once()
    return session, template


class PairSessionTest(unittest.TestCase):
    def test_initial_selection_is_global_and_deliberately_not_token_common(self):
        torch.manual_seed(7)
        model = _make_model()
        session, _ = _make_session(model, "recurrent")
        self.assertEqual(session.cache.counts, [3, 2, 1, 0])
        self.assertEqual(session.cache.pair_count, session.budget_pairs)
        self.assertEqual(
            [head.token_ids.tolist() for head in session.cache.heads],
            [[1, 2, 3], [1, 2], [3], []],
        )
        # A token-common selection would give every group the same ID set and
        # equal count; this physical cache does neither.
        self.assertGreater(len({tuple(h.token_ids.tolist()) for h in session.cache.heads}), 1)
        self.assertGreater(len(set(session.cache.counts)), 1)
        session._assert_alignment()

    def test_two_turn_conditions_keep_exact_storage_and_never_resurrect(self):
        torch.manual_seed(11)
        model = _make_model()
        for condition in ("full", "image_static", "recurrent"):
            with self.subTest(condition=condition):
                session, template = _make_session(model, condition)
                groups = session.state.groups
                deleted = [set() for _ in range(groups)]
                initial_ids = [set(head.token_ids.tolist()) for head in session.cache.heads]
                initial_pairs = session.cache.pair_count

                for turn in range(2):
                    old_total = session.cache.total_seen
                    old_ids = [set(head.token_ids.tolist()) for head in session.cache.heads]
                    result = session.answer(f"question-{turn}", max_new_tokens=2)
                    new_total = session.cache.total_seen
                    next_ids = [set(head.token_ids.tolist()) for head in session.cache.heads]
                    fresh = set(range(old_total, new_total))

                    for group in range(groups):
                        self.assertTrue(next_ids[group] <= old_ids[group] | fresh)
                        self.assertTrue(next_ids[group].isdisjoint(deleted[group]))
                        deleted[group].update(old_ids[group] - next_ids[group])
                    session._assert_alignment()
                    self.assertEqual(session.state.n_pairs, session.cache.pair_count)
                    self.assertEqual(
                        result["selection_after"]["pairs_by_group"],
                        session.cache.counts,
                    )
                    self.assertEqual(result["retained_kv_pairs"], session.cache.pair_count)
                    self.assertEqual(result["retained_kv_bytes"], session.cache.nbytes)
                    self.assertEqual(result["selector_state_bytes"], session.state.nbytes)
                    self.assertEqual(
                        result["persistent_session_tensor_bytes"],
                        result["retained_kv_bytes"]
                        + result["selector_state_bytes"]
                        + result["session_metadata_bytes"],
                    )
                    for field in ("cold_kv_bytes", "h2d_kv_bytes", "d2h_new_kv_bytes"):
                        self.assertEqual(result[field], 0)

                    if condition == "full":
                        expected = initial_pairs + groups * (new_total - 6)
                        self.assertEqual(session.cache.pair_count, expected)
                    else:
                        self.assertEqual(session.cache.pair_count, session.budget_pairs)
                    if condition == "image_static":
                        self.assertEqual(next_ids, initial_ids)

                self.assertEqual(
                    template.calls,
                    [("question-0", True), ("question-1", False)],
                )

    def test_qwen3_pair_adapter_runs_the_same_native_ragged_path(self):
        torch.manual_seed(13)
        model = _make_model("qwen3")
        session, _ = _make_session(model, "recurrent")
        before = [set(head.token_ids.tolist()) for head in session.cache.heads]
        old_total = session.cache.total_seen
        result = session.answer("qwen3-question", max_new_tokens=1)
        fresh = set(range(old_total, session.cache.total_seen))
        for old, head in zip(before, session.cache.heads):
            self.assertTrue(set(head.token_ids.tolist()) <= old | fresh)
        self.assertEqual(result["granularity"], "kv_pair")
        self.assertEqual(result["retained_kv_pairs"], 6)
        self.assertEqual(result["selection_after"]["resident_pairs"], 6)
        session._assert_alignment()


if __name__ == "__main__":
    unittest.main()
