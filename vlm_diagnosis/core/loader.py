"""모델 로더 — V100 제약: fp16 강제(bf16은 에뮬레이션뿐), FA2 불가 → eager.

fp16 NaN 대응 (2026-08-13 실측): Qwen2.5-VL-7B는 fp16에서 LLM 마지막 층(27) 내부
연산이 오버플로해 logits가 NaN이 된다 (잔차 스트림 absmax는 ~6.7k로 정상 범위,
ViT도 정상 — 층 내부 중간값이 문제). 해당 층만 fp32로 실행하면 해결되며 비용은 ~1GB.
`fp32_layers`로 제어. 새 입력 분포에서 NaN이 재발하면 층을 추가할 것.
"""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 모델별 실측 오버플로 층
DEFAULT_FP32_LAYERS = {"Qwen/Qwen2.5-VL-7B-Instruct": (27,)}


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
                  min_pixels=None, max_pixels=None, fp32_layers="auto"):
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
