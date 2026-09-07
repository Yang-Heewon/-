"""Question-blind text context boundaries and a Qwen2 ragged-cache adapter.

The canonical prefix is one *completed system message* containing the context.
An independently encoded user question and assistant opening follow only during
evaluation. No actual question is accepted by ``TextContextTemplate.__init__``.
Templates retain prefix token IDs, not a second copy of the raw context or KV.
``dense_text_inputs`` is the evaluation-only genuine whole-chat reference.
"""
from __future__ import annotations

from numbers import Integral
from types import SimpleNamespace

import torch

from .loader import assert_finite_logits
from .session_types import SessionInput


_CONTEXT_INSTRUCTION = "Answer the question using the following context.\n\nContext:\n"
_BOUNDARY_ANCHOR = "text context boundary anchor"


def _encode(tokenizer, text):
    if not isinstance(text, str):
        raise TypeError("chat template must render text")
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or ids.dtype != torch.long:
        raise ValueError("text encoding must be a nonempty batch-one int64 tensor")
    return ids.detach().cpu().clone()


def _render(tokenizer, messages, generation=False):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)


def _context_message(context):
    if not isinstance(context, str) or not context.strip():
        raise ValueError("text context must be a nonempty string")
    return {"role": "system", "content": _CONTEXT_INSTRUCTION + context}


def _question_message(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    return {"role": "user", "content": question}


class TextContextTemplate:
    """Immutable-by-contract tensor metadata for independent text questions."""

    adapter_id = "qwen_text_context_v1"

    def __init__(self, tokenizer, context: str):
        self.tokenizer = tokenizer
        # Match the existing memory decoding interface without retaining a model,
        # image, raw context, or prefill output.
        self.processor = SimpleNamespace(tokenizer=tokenizer)
        message = _context_message(context)
        rendered = _render(tokenizer, [message])
        self.prefix_ids = _encode(tokenizer, rendered)
        anchor_messages = [{"role": "user", "content": _BOUNDARY_ANCHOR},
                           {"role": "assistant", "content": _BOUNDARY_ANCHOR}]
        completed = _render(tokenizer, anchor_messages)
        if not isinstance(completed, str) or completed.count(_BOUNDARY_ANCHOR) != 2:
            raise ValueError("cannot identify text-chat assistant closing delimiter")
        self.ending_ids = _encode(tokenizer, completed.rsplit(_BOUNDARY_ANCHOR, 1)[1])
        if not torch.equal(_encode(tokenizer, completed)[:, -self.ending_ids.shape[1]:], self.ending_ids):
            raise ValueError("assistant ending is not independently append-compatible")
        # An artificial boundary probe is not an evaluation question. This is
        # only tokenizer work, never an additional decoder/scoring forward.
        whole = _encode(tokenizer, _render(tokenizer, [message, _question_message(_BOUNDARY_ANCHOR)], True))
        if not torch.equal(whole, self.full_ids(_BOUNDARY_ANCHOR)):
            raise ValueError("text chat template is not append-compatible at the context boundary")

    def encode(self, text):
        return _encode(self.tokenizer, text)

    def suffix(self, question: str, first: bool = True):
        if first is not True:
            raise ValueError("text context evaluation requires a fresh independent question")
        # Using an explicit dummy system message suppresses the tokenizer's
        # implicit default-system insertion for a user-only chat. The dummy
        # message is stripped before encoding and never reaches the model.
        anchor = {"role": "system", "content": _BOUNDARY_ANCHOR}
        prefix = _render(self.tokenizer, [anchor])
        whole = _render(self.tokenizer, [anchor, _question_message(question)], True)
        if not isinstance(prefix, str) or not isinstance(whole, str) or not whole.startswith(prefix):
            raise ValueError("text-chat user suffix changes the preceding system message")
        suffix = _encode(self.tokenizer, whole[len(prefix):])
        if not torch.equal(_encode(self.tokenizer, whole), torch.cat((_encode(self.tokenizer, prefix), suffix), dim=1)):
            raise ValueError("text-chat suffix is not independently append-compatible")
        return suffix

    def full_ids(self, question: str):
        """Concatenated cache protocol IDs; use dense_text_inputs for reference."""
        return torch.cat((self.prefix_ids, self.suffix(question)), dim=1)


def dense_text_inputs(tokenizer, context: str, question: str, device="cpu", template=None):
    """Evaluation-only genuine whole encoding, with strict prefix/suffix parity.

    This function intentionally requires raw context again. The compressor and
    its retained template cannot replay that context during question answering.
    """
    template = TextContextTemplate(tokenizer, context) if template is None else template
    ids = _encode(tokenizer, _render(tokenizer, [_context_message(context), _question_message(question)], True))
    expected = template.full_ids(question)
    if not torch.equal(ids, expected):
        raise ValueError("whole text-chat encoding differs from cached prefix plus question suffix")
    ids = ids.to(device)
    return {"input_ids": ids, "position_ids": torch.arange(ids.shape[1], device=device)[None],
            "attention_mask": torch.ones_like(ids)}


class TextPairAdapter:
    """Stateless, append-only Qwen2 text forward over physical head survivors."""

    adapter_id = TextContextTemplate.adapter_id
    supported_modalities = ("control", "text")

    def prepare_turn(self, template, request, first=True):
        return self.text_input(template.suffix(request, first))

    def text_input(self, ids, kind="text"):
        if kind not in {"text", "control", "ending"}:
            raise ValueError("text adapter only accepts text/control/ending")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
            raise ValueError("text input IDs must have batch and token axes")
        return SessionInput(input_ids=ids, modality_ids=torch.full((ids.shape[1],), 0 if kind == "control" else 1, dtype=torch.long))

    def forward(self, model, prepared, cache, position, device):
        from .ragged_kv import RaggedKVCache
        if not isinstance(prepared, SessionInput):
            raise TypeError("prepared must be a SessionInput")
        if not isinstance(cache, RaggedKVCache) or not cache.backend_active:
            raise TypeError("TextPairAdapter requires an active ragged attention backend")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise ValueError("text position must be a nonnegative integer")
        if position != cache.get_seq_length():
            raise ValueError("text positions must match the uncompressed logical cache clock")
        if set(prepared.modality_ids.tolist()) - {0, 1}:
            raise ValueError("text adapter does not accept multimodal inputs")
        if prepared.model_kwargs or prepared.position_ids is not None or prepared.next_position is not None:
            raise ValueError("text adapter owns position and forward arguments")
        ids = prepared.input_ids.to(device)
        n = ids.shape[1]
        logical = torch.arange(position, position + n, device=device)
        out = model(input_ids=ids, position_ids=logical[None], cache_position=logical,
                    attention_mask=torch.zeros(1, 1, n, 1, device=device), past_key_values=cache,
                    use_cache=True, output_attentions=False)
        assert_finite_logits(out.logits, "text_context_adapter_turn")
        return out, position + n

    def decode(self, processor, tokens):
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim != 1:
                raise ValueError("generated tokens must be one-dimensional")
            tokens = tokens.detach().cpu().tolist()
        tokenizer = getattr(processor, "tokenizer", processor)
        text = tokenizer.decode(list(tokens), skip_special_tokens=True)
        if not isinstance(text, str):
            raise TypeError("tokenizer.decode must return text")
        return text.strip()

    def stop_token_ids(self, model):
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            return set()
        values = [eos] if isinstance(eos, Integral) and not isinstance(eos, bool) else eos
        if not isinstance(values, (list, tuple, set)) or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in values):
            raise ValueError("invalid generation EOS token IDs")
        return {int(v) for v in values}

    def metadata_bytes(self, template):
        if not isinstance(template, TextContextTemplate):
            raise TypeError("TextPairAdapter requires a TextContextTemplate")
        return sum(t.numel() * t.element_size() for t in (template.prefix_ids, template.ending_ids))
