"""Modality adapters for the recurrent physical-KV session engine.

The cache engine owns selection, retention, and storage policy.  An adapter
owns only model-family details: one-time context prefill, chat suffixes,
modality labels, logical positions, and decoding.  Adapter instances are
stateless and never retain a :class:`SessionSeed` or any K/V tensors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from numbers import Integral
import time
from typing import Any, ContextManager

import torch

from .loader import assert_finite_logits
from .masked_eval import mrope_position_ids
from .session_types import DEFAULT_MODALITY_NAMES, SessionInput, SessionSeed
from .spans import token_spans


QWEN_IMAGE_MODALITY_NAMES = {
    modality_id: name
    for modality_id, name in DEFAULT_MODALITY_NAMES.items()
    if name in {"control", "text", "image"}
}
_QWEN_MODALITY_IDS = {name: modality_id for modality_id, name in QWEN_IMAGE_MODALITY_NAMES.items()}


class SessionAdapter(ABC):
    """Abstract boundary between a modality/model family and cache policy."""

    adapter_id: str
    supported_modalities: tuple[str, ...]

    @abstractmethod
    def prefill(self, model, processor, context, device) -> SessionSeed:
        """Encode context once and return owned cache plus token metadata."""

    @abstractmethod
    def make_template(self, processor, seed: SessionSeed):
        """Return metadata with batch-one int64 ``ending_ids`` for turn closure.

        The template must not retain the seed, raw media, or source K/V.
        """

    @abstractmethod
    def prepare_turn(self, template, request, first: bool) -> SessionInput:
        """Render one new request without replaying previous answers."""

    @abstractmethod
    def text_input(self, ids: torch.Tensor, kind: str = "text") -> SessionInput:
        """Wrap generated (``text``) or closing (``ending``) token IDs.

        Both kinds are required; the adapter chooses their modality labels.
        """

    @abstractmethod
    def forward(self, model, prepared: SessionInput, cache, position: int, device):
        """Run one append-only decoder step and return output, next position."""

    @abstractmethod
    def observe(self, model) -> ContextManager:
        """Return a context manager reducing actual decoder attention.

        Its context value must expose ``mean() -> CPU Tensor[physical_tokens]``
        after exit, aligned to the growing cache's physical slots.
        """

    @abstractmethod
    def decode(self, processor, tokens) -> str:
        """Decode generated content tokens for evaluation."""

    @abstractmethod
    def stop_token_ids(self, model) -> set[int]:
        """Return model-family stop token IDs."""

    @abstractmethod
    def metadata_bytes(self, template) -> int:
        """Count persistent tensor metadata owned by a template."""


def _batch_one_long(ids: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(ids, torch.Tensor)
        or ids.ndim != 2
        or ids.shape[0] != 1
        or ids.shape[1] < 1
        or ids.dtype != torch.long
    ):
        raise ValueError(f"{name} must be a nonempty batch-one int64 tensor")
    return ids.detach().cpu().clone()


class QwenImageTemplate:
    """Qwen image-chat boundaries with append-compatible fresh suffixes."""

    def __init__(self, processor, image_token_id: int, prefix_ids: torch.Tensor):
        if isinstance(image_token_id, bool) or not isinstance(image_token_id, int):
            raise ValueError("image_token_id must be an integer")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_token_id = image_token_id
        self.prefix_ids = _batch_one_long(prefix_ids, "prefix_ids")
        self.anchor = [
            {"role": "user", "content": "session user anchor"},
            {"role": "assistant", "content": "session assistant anchor"},
        ]
        completed = processor.apply_chat_template(
            self.anchor, tokenize=False, add_generation_prompt=False
        )
        if not isinstance(completed, str) or completed.count("session assistant anchor") != 1:
            raise ValueError("cannot identify assistant closing delimiter")
        self.anchor_ids = self.encode(completed)
        ending = completed.split("session assistant anchor", 1)[1]
        self.ending_ids = self.encode(ending)
        if not self.ending_ids.numel():
            raise ValueError("chat template lacks an assistant closing delimiter")
        ending_length = self.ending_ids.shape[1]
        if (
            ending_length > self.anchor_ids.shape[1]
            or not torch.equal(
                self.anchor_ids[:, -ending_length:], self.ending_ids
            )
        ):
            raise ValueError(
                "assistant closing delimiter is not independently append-compatible"
            )

    def encode(self, text: str) -> torch.Tensor:
        if not isinstance(text, str):
            raise TypeError("chat-template output must be text")
        encoded = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        return _batch_one_long(encoded, "encoded template IDs")

    def suffix(self, question: str, first: bool) -> torch.Tensor:
        if not isinstance(question, str):
            raise TypeError("Qwen image-session requests must be text strings")
        if not isinstance(first, bool):
            raise TypeError("first must be bool")
        if first:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            rendered = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            raw = self.encode(rendered)
            visual = (raw[0] == self.image_token_id).nonzero(as_tuple=True)[0]
            if visual.numel() != 1:
                raise ValueError("first turn must contain exactly one image placeholder")
            placeholder = int(visual[0])
            n_visual = int((self.prefix_ids == self.image_token_id).sum())
            if n_visual < 1:
                raise ValueError("cached prefix has no image tokens")
            expanded_prefix = torch.cat(
                (
                    raw[:, :placeholder],
                    raw[:, placeholder : placeholder + 1].repeat(1, n_visual),
                    raw[:, placeholder + 1 : placeholder + 2],
                ),
                dim=1,
            )
            if not torch.equal(expanded_prefix, self.prefix_ids):
                raise ValueError("question template does not match cached image prefix")
            # Skip the single placeholder and its closing vision boundary. The
            # expanded image prefix has already been encoded exactly once.
            return raw[:, placeholder + 2 :].clone()

        rendered = self.processor.apply_chat_template(
            self.anchor + [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        raw = self.encode(rendered)
        prefix_length = self.anchor_ids.shape[1]
        if not torch.equal(raw[:, :prefix_length], self.anchor_ids):
            raise ValueError("chat template is not append-compatible")
        return raw[:, prefix_length:].clone()


class QwenImageAdapter(SessionAdapter):
    """Single-image adapter for Qwen2.5-VL and Qwen3-VL decoder caches."""

    adapter_id = "qwen_image_v1"
    supported_modalities = ("control", "text", "image")

    @torch.no_grad()
    def prefill(self, model, processor, context, device) -> SessionSeed:
        if context is None or isinstance(context, (str, bytes, list, tuple, dict)):
            raise TypeError("QwenImageAdapter expects one image context object")

        # Lazy runtime import avoids a cycle when session_cache exposes the
        # backwards-compatible adapter wrappers.
        from .session_cache import AttentionMass, _sync
        from .signals import vlm_inputs

        inputs = vlm_inputs(processor, context, "x", device)
        required = {"input_ids", "image_grid_thw", "pixel_values"}
        if not required <= set(inputs):
            raise ValueError("Qwen image processor output is missing required tensors")
        spans = token_spans(inputs["input_ids"], model.config)
        visual = torch.as_tensor(spans["visual"], dtype=torch.long)
        if visual.ndim != 1 or not visual.numel():
            raise ValueError("Qwen image prefill requires visual tokens")
        prefix_length = int(spans["vis_end"]) + 2
        ids = inputs["input_ids"][:, :prefix_length]
        positions = mrope_position_ids(
            model,
            ids,
            inputs["image_grid_thw"],
            torch.ones_like(ids),
        )
        if tuple(positions.shape) != (3, 1, prefix_length):
            raise ValueError("Qwen mRoPE positions must have shape (3, 1, tokens)")

        _sync(device)
        started = time.perf_counter()
        if getattr(self, "capture_pair_scores", False):
            from .ragged_kv import HeadAttentionMass
            pair_observer = HeadAttentionMass(model, int(visual.min()), int(spans["vis_end"]) + 1)
        else:
            pair_observer = nullcontext()
        with AttentionMass(model, int(visual.min()), int(spans["vis_end"]) + 1) as capture, pair_observer as pairs:
            output = model(
                input_ids=ids,
                position_ids=positions,
                attention_mask=torch.ones_like(ids),
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                use_cache=True,
                output_attentions=False,
            )
        assert_finite_logits(output.logits, "qwen_image_adapter_prefill")
        kv = tuple(
            (key.detach().cpu(), value.detach().cpu())
            for key, value in output.past_key_values.to_legacy_cache()
        )
        _sync(device)

        modality_ids = torch.full(
            (prefix_length,), _QWEN_MODALITY_IDS["control"], dtype=torch.long
        )
        modality_ids[visual.cpu()] = _QWEN_MODALITY_IDS["image"]
        prior_scores = capture.mean()
        if prior_scores.shape != (prefix_length,):
            raise RuntimeError("prefill attention and cached prefix lengths disagree")
        prior_scores[modality_ids != _QWEN_MODALITY_IDS["image"]] = 0
        pair_prior = pairs.mean() if pairs is not None else None
        if pair_prior is not None:
            pair_prior[..., modality_ids != _QWEN_MODALITY_IDS["image"]] = 0
        return SessionSeed(
            kv=kv,
            prefix_ids=ids.detach().cpu(),
            prior_scores=prior_scores,
            modality_ids=modality_ids,
            next_position=int(positions.max()) + 1,
            prefill_seconds=time.perf_counter() - started,
            modality_names=dict(QWEN_IMAGE_MODALITY_NAMES),
            token_features={},
            adapter_id=self.adapter_id,
            pair_prior_scores=pair_prior,
        )

    def make_template(self, processor, seed: SessionSeed) -> QwenImageTemplate:
        if not isinstance(seed, SessionSeed):
            raise TypeError("seed must be a SessionSeed")
        if seed.adapter_id != self.adapter_id:
            raise ValueError("seed was produced by a different adapter")
        image_modality_ids = [
            modality_id
            for modality_id, name in seed.modality_names.items()
            if name == "image"
        ]
        if len(image_modality_ids) != 1:
            raise ValueError("Qwen image seed requires exactly one image modality ID")
        image_positions = seed.modality_ids == image_modality_ids[0]
        image_tokens = seed.prefix_ids[0, image_positions].unique()
        if image_tokens.numel() != 1:
            raise ValueError("Qwen image positions must share one placeholder token ID")
        image_token_id = int(image_tokens[0])
        if not torch.equal(seed.prefix_ids[0] == image_token_id, image_positions):
            raise ValueError("Qwen image placeholder and modality positions disagree")
        processor_token_id = getattr(processor, "image_token_id", None)
        if processor_token_id is not None and int(processor_token_id) != image_token_id:
            raise ValueError("processor and cached image token IDs disagree")
        return QwenImageTemplate(processor, image_token_id, seed.prefix_ids)

    def prepare_turn(self, template, request, first: bool) -> SessionInput:
        if not isinstance(template, QwenImageTemplate):
            raise TypeError("QwenImageAdapter requires a QwenImageTemplate")
        if not isinstance(request, str):
            raise TypeError("Qwen image-session requests must be text strings")
        ids = template.suffix(request, first)
        return SessionInput(
            input_ids=ids,
            modality_ids=torch.full(
                (ids.shape[1],), _QWEN_MODALITY_IDS["text"], dtype=torch.long
            ),
        )

    def text_input(self, ids: torch.Tensor, kind: str = "text") -> SessionInput:
        if kind not in {"text", "control", "ending"}:
            raise ValueError(
                "QwenImageAdapter text_input supports only text/control/ending; "
                "images require the one-time prefill and audio/video are unsupported"
            )
        owned_ids = _batch_one_long(ids, "input IDs")
        # The assistant closing delimiter is generated history, just like the
        # answer body. ``ending`` names its role in the turn protocol without
        # inventing a fourth Qwen modality.
        modality = "text" if kind == "ending" else kind
        return SessionInput(
            input_ids=owned_ids,
            modality_ids=torch.full(
                (owned_ids.shape[1],), _QWEN_MODALITY_IDS[modality], dtype=torch.long
            ),
        )

    def forward(
        self,
        model,
        prepared: SessionInput,
        cache,
        position: int,
        device,
    ):
        if not isinstance(prepared, SessionInput):
            raise TypeError("prepared must be a SessionInput")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise ValueError("position must be a nonnegative integer")
        if cache is None or not hasattr(cache, "get_seq_length"):
            raise TypeError("Qwen decoder forward requires a cache object")
        allowed_turn_ids = {
            _QWEN_MODALITY_IDS["control"],
            _QWEN_MODALITY_IDS["text"],
        }
        if not set(prepared.modality_ids.tolist()) <= allowed_turn_ids:
            raise ValueError("QwenImageAdapter does not accept new multimodal turn inputs")

        ids = prepared.input_ids.to(device)
        token_count = ids.shape[1]
        old_length = cache.get_seq_length()
        logical = torch.arange(position, position + token_count, device=device)
        position_ids = logical[None, None].expand(3, 1, token_count)
        # This adapter only appends text/control after a single image seed;
        # arbitrary multimodal positions require a different adapter.
        if prepared.position_ids is not None and not torch.equal(
                prepared.position_ids.to(device), position_ids):
            raise ValueError("Qwen text positions must continue the current logical cursor")
        next_position = position + token_count
        if prepared.next_position is not None and prepared.next_position != next_position:
            raise ValueError("Qwen text next_position must follow the appended token count")

        output = model(
            input_ids=ids,
            position_ids=position_ids,
            cache_position=torch.arange(
                old_length, old_length + token_count, device=device
            ),
            attention_mask=torch.ones(
                1, old_length + token_count, dtype=torch.long, device=device
            ),
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
            **dict(prepared.model_kwargs),
        )
        assert_finite_logits(output.logits, "qwen_image_adapter_turn")
        return output, next_position

    def observe(self, model) -> ContextManager:
        from .session_cache import AttentionMass

        return AttentionMass(model)

    def decode(self, processor, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim != 1:
                raise ValueError("generated tokens must be one-dimensional")
            tokens = tokens.detach().cpu().tolist()
        result = processor.tokenizer.decode(
            list(tokens), skip_special_tokens=True
        )
        if not isinstance(result, str):
            raise TypeError("tokenizer.decode must return text")
        return result.strip()

    def stop_token_ids(self, model) -> set[int]:
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            return set()
        if isinstance(eos, Integral) and not isinstance(eos, bool):
            values = (eos,)
        elif isinstance(eos, (list, tuple, set)):
            values = eos
        else:
            raise ValueError("invalid generation EOS token IDs")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
            for value in values
        ):
            raise ValueError("invalid generation EOS token IDs")
        return {int(value) for value in values}

    def metadata_bytes(self, template) -> int:
        if not isinstance(template, QwenImageTemplate):
            raise TypeError("QwenImageAdapter requires a QwenImageTemplate")
        tensors = (template.prefix_ids, template.anchor_ids, template.ending_ids)
        return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))


class QwenPairAdapter(QwenImageAdapter):
    """Same image protocol, with true layer/KV-head priors and ragged decode."""

    adapter_id = "qwen_image_pairs_v1"
    capture_pair_scores = True

    def forward(self, model, prepared, cache, position, device):
        from .ragged_kv import RaggedKVCache
        if not isinstance(cache, RaggedKVCache) or not cache.backend_active:
            raise TypeError("QwenPairAdapter requires an active ragged attention backend")
        if set(prepared.modality_ids.tolist()) - {0, 1}:
            raise ValueError("pair image adapter supports only new text/control inputs")
        if prepared.model_kwargs or prepared.position_ids is not None or prepared.next_position is not None:
            raise ValueError("pair image adapter owns text-only position and forward arguments")
        ids = prepared.input_ids.to(device)
        n, old = ids.shape[1], cache.get_seq_length()
        logical = torch.arange(position, position+n, device=device)
        output = model(
            input_ids=ids, position_ids=logical[None, None].expand(3, 1, n),
            cache_position=torch.arange(old, old+n, device=device),
            # Both Qwen families bypass dense mask construction for 4D masks.
            # Actual ragged causal visibility is applied per head by the backend.
            attention_mask=torch.zeros(1, 1, n, 1, device=device), past_key_values=cache,
            use_cache=True, output_attentions=False)
        assert_finite_logits(output.logits, "qwen_pair_adapter_turn")
        return output, position+n
