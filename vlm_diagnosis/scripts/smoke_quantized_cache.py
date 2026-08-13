"""Physical Transformers QuantizedCache smoke test for Qwen2.5-VL.

HQQ works on the local V100.  Quanto currently tries to compile sm_80+
instructions and is therefore unsupported on sm_70.  This smoke only validates
API/runtime compatibility; it is not an accuracy or speed benchmark.
"""

import argparse

import torch
from PIL import Image, ImageDraw

from vlm_diagnosis.core.loader import assert_finite_logits, load_qwen25vl
from vlm_diagnosis.core.signals import vlm_inputs


def main(args):
    model, processor = load_qwen25vl(device=args.device, max_pixels=224 * 224)
    image = Image.new("RGB", (224, 224), "white")
    ImageDraw.Draw(image).text((20, 90), "CODE 42", fill="black")
    inputs = vlm_inputs(processor, image, "Read the code. Answer briefly.", args.device)
    common = dict(
        max_new_tokens=2,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    with torch.no_grad():
        dense = model.generate(**inputs, cache_implementation="dynamic", **common)
        quant = model.generate(
            **inputs,
            cache_implementation="quantized",
            cache_config={
                "backend": args.backend,
                "nbits": args.nbits,
                "axis_key": 1 if args.backend == "hqq" else 0,
                "axis_value": 1 if args.backend == "hqq" else 0,
                "q_group_size": 64,
                "residual_length": 0,
            },
            **common,
        )

    for name, result in (("dense", dense), ("quant", quant)):
        scores = torch.stack(result.scores)
        assert_finite_logits(scores, name)
        new_tokens = result.sequences[0, inputs["input_ids"].shape[1] :]
        print(name, new_tokens.tolist(), repr(processor.decode(new_tokens)))
    for step in range(len(dense.scores)):
        delta = (dense.scores[step].float() - quant.scores[step].float()).abs()
        print(f"step={step} max_abs={delta.max().item():.6f} mean_abs={delta.mean().item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("hqq", "quanto"), default="hqq")
    parser.add_argument("--nbits", type=int, choices=(2, 4), default=4)
    main(parser.parse_args())
