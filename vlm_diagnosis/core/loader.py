"""모델 로더 — V100 제약: fp16 강제(bf16은 에뮬레이션뿐), FA2 불가 → eager.

fp16 NaN 원인 확정 (2026-08-13, M0-04 진단; results/smoke/nan_diagnosis/):
HF eager attention은 `matmul(Q, K^T) * scaling` 순서라 스케일 전 QK^T가 fp16 matmul
출력에서 65504를 넘으면 inf → softmax NaN이 된다. 실측: doc4733 q1의 layer 0에서
스케일 후 max|logit| 6145 (스케일 전 ≈ 69.5k > 65504), layer 27은 상시 ~10.5k
(스케일 전 ≈ 118k) — 과거 layer-27 fp32 패치가 우연히 맞았던 이유도 같은 메커니즘.

근본 처방: `(Q*scaling) @ K^T`로 순서를 바꾸는 patch_stable_qk_scale() (수학적 동일,
스케일 후 최대 ~11k는 fp16 범위 내). 기본 적용된다. `fp32_layers`는 진단·비교용으로
남겨두며 prescale 패치가 검증된 뒤에는 기본 비활성이다.
"""
import torch
from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                          Qwen3VLForConditionalGeneration)
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen_mod
import transformers.models.qwen3_vl.modeling_qwen3_vl as _qwen3_mod

# 모델별 실측 오버플로 층 — prescale 패치 이전의 1차 완화 기록 (진단 비교용)
LEGACY_FP32_LAYERS = {"Qwen/Qwen2.5-VL-7B-Instruct": (27,)}
DEFAULT_FP32_LAYERS = {}

_ORIG_EAGER_ATTENTION = _qwen_mod.eager_attention_forward
_ORIG_EAGER_ATTENTION_Q3 = _qwen3_mod.eager_attention_forward


def _stable_eager_attention(module, query, key, value, attention_mask,
                            scaling, dropout=0.0, **kwargs):
    """(Q*scaling)@K^T — matmul 출력의 fp16 overflow(스케일 전 |QK^T|>65504) 차단."""
    return _ORIG_EAGER_ATTENTION(module, query * scaling, key, value,
                                 attention_mask, 1.0, dropout, **kwargs)


def _stable_eager_attention_q3(module, query, key, value, attention_mask,
                               scaling, dropout=0.0, **kwargs):
    return _ORIG_EAGER_ATTENTION_Q3(module, query * scaling, key, value,
                                    attention_mask, 1.0, dropout, **kwargs)


def patch_stable_qk_scale(enable=True):
    _qwen_mod.eager_attention_forward = (
        _stable_eager_attention if enable else _ORIG_EAGER_ATTENTION)
    _qwen3_mod.eager_attention_forward = (
        _stable_eager_attention_q3 if enable else _ORIG_EAGER_ATTENTION_Q3)


def _run_layer_in_fp32(layer):
    """디코더 층 하나를 fp32로 실행: 입력 업캐스트 → fp32 연산 → fp16 반환."""
    layer.float()
    orig = layer.forward

    def fwd(hidden_states, **kw):
        cast = {}
        for k, v in kw.items():
            if torch.is_tensor(v) and v.is_floating_point():
                cast[k] = v.float()
            elif k == "position_embeddings" and v is not None:
                cast[k] = tuple(t.float() for t in v)
            else:
                cast[k] = v
        out = orig(hidden_states.float(), **cast)
        if isinstance(out, tuple):
            return (out[0].half(),) + out[1:]
        return out.half()

    layer.forward = fwd


def load_qwen25vl(model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0",
                  min_pixels=None, max_pixels=None, fp32_layers="auto",
                  stable_qk_scale=True):
    patch_stable_qk_scale(stable_qk_scale)
    pkw = {}
    if min_pixels is not None:
        pkw["min_pixels"] = min_pixels
    if max_pixels is not None:
        pkw["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_id, **pkw)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to(device).eval()

    if fp32_layers == "auto":
        fp32_layers = DEFAULT_FP32_LAYERS.get(model_id, ())
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for i in fp32_layers or ():
        _run_layer_in_fp32(lm.layers[i])
    return model, processor


def assert_finite_logits(logits, context=""):
    """모든 실험 공통 가드: NaN 재발 시 즉시 중단 (fp32_layers 추가 필요 신호)."""
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"logits에 NaN/Inf — fp32_layers 확장 필요 ({context})")


def _patch_fast_patch_embed(visual):
    """V100 함정: fp16 Conv3d(kernel=stride)가 cuDNN 병리 커널로 ~85s.
    kernel==stride라 이 conv는 flatten 후 linear와 수학적으로 동일 — matmul로 교체
    (SQA 1화면 기준 84.5s → <0.1s, 값 동일성은 loader 테스트에서 확인)."""
    pe = visual.patch_embed
    W = pe.proj.weight.view(pe.embed_dim, -1)
    b = pe.proj.bias

    def fwd(hidden_states):
        return torch.addmm(b, hidden_states.to(W.dtype), W.t())

    pe.forward = fwd


def load_qwen3vl(model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda:0",
                 min_pixels=None, max_pixels=None, stable_qk_scale=True):
    """Qwen3-VL 로더 — 같은 V100 제약(fp16, eager)과 NaN prescale 처방 적용.
    구조 차이: 36층/kv_head 8, DeepStack 시각 주입(초기 층 hidden state에 합산 —
    시각 KV column eviction 의미는 동일: 해당 위치의 KV를 읽지 못하게 됨)."""
    patch_stable_qk_scale(stable_qk_scale)
    pkw = {}
    if min_pixels is not None:
        pkw["min_pixels"] = min_pixels
    if max_pixels is not None:
        pkw["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_id, **pkw)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16,
        attn_implementation={"text_config": "eager", "vision_config": "sdpa"},
    ).to(device).eval()
    _patch_fast_patch_embed(model.model.visual)
    return model, processor


def load_vlm(family, **kw):
    """러너용 통합 로더. family: qwen25vl | qwen3vl"""
    return {"qwen25vl": load_qwen25vl, "qwen3vl": load_qwen3vl}[family](**kw)


def kv_dims(model):
    """KV 회계용 (layers, kv_heads, head_dim) — config에서 유도 (하드코딩 금지)."""
    c = model.config
    tc = getattr(c, "text_config", c)
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    return tc.num_hidden_layers, tc.num_key_value_heads, hd
