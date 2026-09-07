"""Offline contracts for question-blind text prefixes and real Qwen2 attention."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM
import transformers.models.qwen2.modeling_qwen2 as qwen2

from vlm_diagnosis.core.ragged_kv import RaggedAttention, RaggedKVCache
from vlm_diagnosis.core.text_context_adapter import (
    TextContextTemplate, TextPairAdapter, dense_text_inputs,
)


class _Tokenizer:
    """Tiny reversible, offline tokenizer with Qwen-style chat boundaries."""

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        assert not add_special_tokens and return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) + 4 for c in text]], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 4) for i in ids if i >= 4)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": "You are helpful."}, *messages]
        rendered = "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" for m in messages)
        return rendered + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def _model():
    torch.manual_seed(13)
    cfg = Qwen2Config(vocab_size=260, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=1024, eos_token_id=2, pad_token_id=0)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


@contextmanager
def _dense_head_mask(keep):
    original = qwen2.eager_attention_forward

    def forward(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        mask = attention_mask[..., :key.shape[-2]].expand(1, query.shape[1], query.shape[-2], key.shape[-2]).clone()
        blocked_prefix = (~keep[module.layer_idx]).repeat_interleave(module.num_key_value_groups, dim=0)
        mask[..., :blocked_prefix.shape[-1]].masked_fill_(blocked_prefix[None, :, None], float("-inf"))
        return original(module, query, key, value, mask, scaling, dropout, **kwargs)

    qwen2.eager_attention_forward = forward
    try:
        yield
    finally:
        qwen2.eager_attention_forward = original


def test_text_template_never_stores_raw_context_and_matches_whole_encoding():
    tokenizer = _Tokenizer()
    context = "Device lumen has code 7319."
    template = TextContextTemplate(tokenizer, context)
    initial = template.prefix_ids.clone()
    for question in ("What is the code?", "Which device?", "Name the device.\nBe concise."):
        whole = dense_text_inputs(tokenizer, context, question, template=template)
        assert torch.equal(whole["input_ids"], template.full_ids(question))
        assert torch.equal(whole["position_ids"], torch.arange(whole["input_ids"].shape[1])[None])
        assert torch.equal(template.prefix_ids, initial)
        assert question not in tokenizer.decode(template.prefix_ids[0].tolist())
    assert not any(isinstance(v, str) and context in v for v in vars(template).values())
    assert set(vars(template)) == {"tokenizer", "processor", "prefix_ids", "ending_ids"}
    with pytest.raises(ValueError, match="independent question"):
        template.suffix("Which device?", first=False)
    with pytest.raises(ValueError, match="whole text-chat encoding"):
        dense_text_inputs(tokenizer, "Different context.", "Which device?", template=template)
    with pytest.raises(ValueError, match="nonempty"):
        template.suffix("")
    with pytest.raises(ValueError, match="nonempty"):
        TextContextTemplate(tokenizer, " ")


def test_template_rejects_non_append_compatible_chat_template():
    class BadTemplate(_Tokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            rendered = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            if messages[0]["role"] == "system" and len(messages) > 1:
                rendered = "changed-prefix" + rendered
            return rendered

    with pytest.raises(ValueError, match="preceding system message"):
        TextContextTemplate(BadTemplate(), "A context.")


@torch.no_grad()
def test_real_qwen2_full_ragged_matches_genuine_whole_prompt_for_independent_questions():
    tokenizer, model, adapter = _Tokenizer(), _model(), TextPairAdapter()
    context = "Device lumen has code 7319."
    template = TextContextTemplate(tokenizer, context)
    initial = model(input_ids=template.prefix_ids, use_cache=True)
    legacy = initial.past_key_values.to_legacy_cache()
    saved = [(k.clone(), v.clone()) for k, v in legacy]
    for question in ("Code?", "Device?"):
        reference = model(**dense_text_inputs(tokenizer, context, question, template=template), use_cache=False)
        branch = RaggedKVCache(legacy)
        prepared = adapter.prepare_turn(template, question)
        with RaggedAttention(model, branch, collect=False):
            actual, next_position = adapter.forward(model, prepared, branch, template.prefix_ids.shape[1], "cpu")
        torch.testing.assert_close(actual.logits, reference.logits[:, template.prefix_ids.shape[1]:], atol=2e-6, rtol=2e-5)
        assert next_position == branch.get_seq_length() == template.full_ids(question).shape[1]
        assert actual.past_key_values is branch
        for (k, v), (sk, sv) in zip(legacy, saved):
            assert torch.equal(k, sk) and torch.equal(v, sv)
    assert adapter.stop_token_ids(model) == {2}
    assert adapter.decode(template.processor, [ord(c) + 4 for c in " code "]) == "code"
    assert adapter.metadata_bytes(template) == (template.prefix_ids.numel() + template.ending_ids.numel()) * 8


@torch.no_grad()
def test_real_qwen2_partial_ragged_matches_dense_head_mask_and_keeps_logical_ids():
    model, adapter = _model(), TextPairAdapter()
    initial = model(input_ids=torch.tensor([[4, 5, 6, 7, 8, 9]]), use_cache=True)
    legacy = initial.past_key_values.to_legacy_cache()
    keep_ids = [torch.tensor([], dtype=torch.long), torch.tensor([1]), torch.tensor([0, 3]), torch.tensor([0, 2, 5])]
    ragged = RaggedKVCache(legacy, keep_ids)
    dense = DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in legacy))
    keep = torch.zeros(2, 2, 6, dtype=torch.bool)
    for g, ids in enumerate(keep_ids):
        keep[g // 2, g % 2, ids] = True
    original = qwen2.eager_attention_forward
    logical = 6
    for suffix in (torch.tensor([[10, 11]]), torch.tensor([[12]])):
        n = suffix.shape[1]
        with _dense_head_mask(keep):
            expected = model(input_ids=suffix, position_ids=torch.arange(logical, logical + n)[None],
                             cache_position=torch.arange(logical, logical + n),
                             attention_mask=torch.ones(1, logical + n), past_key_values=dense, use_cache=True)
        with RaggedAttention(model, ragged, collect=True) as observation:
            actual, logical = adapter.forward(model, adapter.text_input(suffix), ragged, logical, "cpu")
        assert qwen2.eager_attention_forward is original
        torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
        assert len(observation.means()) == 4
    assert ragged.counts == [3, 4, 5, 6]
    assert ragged.get_seq_length() == 9
    for head, ids in zip(ragged.heads, keep_ids):
        assert torch.equal(head.token_ids, torch.cat((ids, torch.tensor([6, 7, 8]))))
    with pytest.raises(ValueError, match="logical cache clock"):
        with RaggedAttention(model, ragged):
            adapter.forward(model, adapter.text_input(torch.tensor([[13]])), ragged, 6, "cpu")
    assert qwen2.eager_attention_forward is original
    assert not ragged.backend_active


def test_text_adapter_rejects_unsafe_forward_arguments():
    model, adapter = _model(), TextPairAdapter()
    cache = RaggedKVCache(tuple((torch.zeros(1, 2, 2, 8), torch.zeros(1, 2, 2, 8)) for _ in range(2)))
    prepared = adapter.text_input(torch.tensor([[3]]))
    with pytest.raises(TypeError, match="active ragged"):
        adapter.forward(model, prepared, cache, 2, "cpu")
    with RaggedAttention(model, cache):
        with pytest.raises(ValueError, match="position"):
            adapter.forward(model, prepared, cache, True, "cpu")
        prepared.position_ids = torch.tensor([[2]])
        with pytest.raises(ValueError, match="owns position"):
            adapter.forward(model, prepared, cache, 2, "cpu")
        prepared.position_ids = None
        prepared.modality_ids[:] = 2
        with pytest.raises(ValueError, match="multimodal"):
            adapter.forward(model, prepared, cache, 2, "cpu")
    with pytest.raises(ValueError, match="text/control/ending"):
        adapter.text_input(torch.tensor([[3]]), "image")
    model.generation_config.eos_token_id = True
    with pytest.raises(ValueError, match="EOS"):
        adapter.stop_token_ids(model)
