"""모델 로더 — V100 제약: fp16 강제(bf16은 에뮬레이션뿐), FA2 불가 → eager.

fp16 NaN 1차 대응 (2026-08-13 실측): Qwen2.5-VL-7B의 일부 입력은 LLM 마지막 층(27)을
fp32로 실행하면 finite가 된다. 그러나 legacy D4의 4D-mask S0 경로에서는 이 패치 뒤에도
NaN이 재현됐다. 따라서 이 설정은 완전한 해결책이 아니며 M0의 mask/position/layer finite
진단을 통과하기 전에는 본실험에 사용하지 않는다. 비용은 약 1GB이고 `fp32_layers`로 제어한다.
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
