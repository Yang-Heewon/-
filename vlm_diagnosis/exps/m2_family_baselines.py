"""M2 quality diagnostic: byte-matched SPARSE / QUANT / TRANSFORMED / HYBRID KV.

This runner intentionally uses the existing 4D-mask, teacher-forced path.  It
answers the quality question at an equal *serialized visual-KV byte budget*;
it does not claim physical GPU memory or latency savings.  M6 must use a real
cache backend after its physical-cache smoke test passes.

The target byte budget is the storage occupied by all visual tokens under a
KIVI-style ``--reference-bits`` quantizer.  At that same budget:

* SPARSE keeps the largest number of fp16 visual tokens that fit.
* QUANT keeps every visual token at ``reference_bits``.
* TRANSFORMED uses merge-on-evict with the same fp16 keep count as SPARSE.
* HYBRID keeps as many visual tokens as fit at ``hybrid_bits`` and quantizes
  those tokens; by default 8-bit HYBRID is compared with 4-bit QUANT.

Run a one-document smoke test first:

    python -m vlm_diagnosis.exps.m2_family_baselines --limit 1 --device cuda:0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
import zlib

import torch
from PIL import Image

from vlm_diagnosis.core import signals as S
from vlm_diagnosis.core.kv_baselines import (
    VisualKVTransform,
    dense_storage,
    hybrid_storage,
    max_keep_for_budget,
    quantized_storage,
    shape_from_config,
    sparse_storage,
)
from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.masked_eval import (
    answer_logp,
    causal_mask_4d,
    evict_columns,
    mrope_position_ids,
)
from vlm_diagnosis.core.spans import token_spans


ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
META = os.path.join(ROOT, "data", "d4_mini", "meta.jsonl")
DEFAULT_OUTPUT = os.path.join(ROOT, "results", "smoke", "m2_family_baselines.jsonl")
MAX_PIXELS = 1280 * 28 * 28


def _mask_for_keep(base_mask, visual_positions, keep_ordinals, row_start):
    keep = set(int(i) for i in keep_ordinals)
    evict = torch.tensor(
        [int(position) for ordinal, position in enumerate(visual_positions) if ordinal not in keep],
        device=base_mask.device,
        dtype=torch.long,
    )
    return evict_columns(base_mask, evict, row_start)


@torch.no_grad()
def evaluate_one(model, processor, image, question, answer, device, reference_bits, hybrid_bits, seed):
    ins = S.vlm_inputs(processor, image, question, device)
    answer_ids = processor.tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([ins["input_ids"], answer_ids], dim=1)
    spans = token_spans(full_ids, model.config)
    visual_positions = spans["visual"]
    n_visual = int(visual_positions.numel())
    answer_start = int(ins["input_ids"].shape[1])
    seq_len = int(full_ids.shape[1])

    shape = shape_from_config(model.config, n_visual)
    full_bytes = dense_storage(shape).total_bytes
    target_bytes = quantized_storage(shape, reference_bits).total_bytes
    sparse_keep_n = max_keep_for_budget(shape, target_bytes, "sparse")
    hybrid_keep_n = max_keep_for_budget(
        shape, target_bytes, "hybrid", nbits=hybrid_bits
    )

    # S1 is a VLM adaptation of a query-attention/SnapKV-style selector.  S0 is
    # included so a sophisticated selector is never compared only to itself.
    s1 = S.score_s1(model, processor, image, question, device)
    s0 = S.score_s0(n_visual, seed=seed)
    sparse_s1 = sorted(S.topk_keep(s1, sparse_keep_n / n_visual))
    sparse_s0 = sorted(S.topk_keep(s0, sparse_keep_n / n_visual))
    hybrid_s1 = sorted(S.topk_keep(s1, hybrid_keep_n / n_visual))

    attn2d = torch.ones(1, seq_len, dtype=torch.long, device=device)
    position_ids = mrope_position_ids(model, full_ids, ins["image_grid_thw"], attn2d)
    base_mask = causal_mask_4d(seq_len, device)
    common = dict(
        input_ids=full_ids,
        pixel_values=ins["pixel_values"],
        image_grid_thw=ins["image_grid_thw"],
        answer_start=answer_start,
        position_ids=position_ids,
    )

    def run(mask, transform=None):
        cm = transform if transform is not None else contextlib.nullcontext()
        with cm:
            total, token_logp = answer_logp(model, attention_mask=mask, **common)
        if not math.isfinite(total) or not torch.isfinite(token_logp).all():
            raise RuntimeError("non-finite baseline result; M0 finite gate failed")
        return {
            "logp": total,
            "logp_per_answer_token": total / max(1, int(answer_ids.shape[1])),
        }

    mask_sparse_s1 = _mask_for_keep(base_mask, visual_positions.tolist(), sparse_s1, spans["vis_end"] + 1)
    mask_sparse_s0 = _mask_for_keep(base_mask, visual_positions.tolist(), sparse_s0, spans["vis_end"] + 1)
    mask_hybrid = _mask_for_keep(base_mask, visual_positions.tolist(), hybrid_s1, spans["vis_end"] + 1)

    results = {
        "FULL_FP16": run(base_mask),
        "SPARSE_S1_FP16": run(mask_sparse_s1),
        "SPARSE_RANDOM_FP16": run(mask_sparse_s0),
        f"QUANT_KIVI_{reference_bits}BIT": run(
            base_mask,
            VisualKVTransform(model, visual_positions, nbits=reference_bits),
        ),
        "TRANSFORM_MERGE_S1_FP16": run(
            mask_sparse_s1,
            VisualKVTransform(model, visual_positions, keep_indices=sparse_s1, merge=True),
        ),
        f"HYBRID_S1_{hybrid_bits}BIT": run(
            mask_hybrid,
            VisualKVTransform(model, visual_positions, nbits=hybrid_bits),
        ),
    }

    storage = {
        "FULL_FP16": dense_storage(shape).total_bytes,
        "SPARSE_S1_FP16": sparse_storage(shape, sparse_keep_n).total_bytes,
        "SPARSE_RANDOM_FP16": sparse_storage(shape, sparse_keep_n).total_bytes,
        f"QUANT_KIVI_{reference_bits}BIT": quantized_storage(shape, reference_bits).total_bytes,
        "TRANSFORM_MERGE_S1_FP16": sparse_storage(shape, sparse_keep_n).total_bytes,
        f"HYBRID_S1_{hybrid_bits}BIT": hybrid_storage(shape, hybrid_keep_n, hybrid_bits).total_bytes,
    }
    for name, result in results.items():
        result["estimated_visual_kv_bytes"] = storage[name]
        result["estimated_ratio_to_full"] = storage[name] / full_bytes
        result["delta_logp_per_token"] = (
            result["logp_per_answer_token"] - results["FULL_FP16"]["logp_per_answer_token"]
        )

    return {
        "n_visual": n_visual,
        "answer_tokens": int(answer_ids.shape[1]),
        "target_bytes": target_bytes,
        "target_ratio_to_full": target_bytes / full_bytes,
        "sparse_keep": sparse_keep_n,
        "hybrid_keep": hybrid_keep_n,
        "reference_bits": reference_bits,
        "hybrid_bits": hybrid_bits,
        "sample_seed": seed,
        "metric_scope": "teacher_forced_quality_simulation_not_physical_cache",
        "results": results,
    }


def run(args):
    if args.reference_bits not in (2, 4):
        raise ValueError("reference_bits must be 2 or 4")
    if args.hybrid_bits <= args.reference_bits:
        raise ValueError("hybrid_bits must exceed reference_bits so HYBRID actually sparsifies")
    model, processor = load_qwen25vl(device=args.device, max_pixels=MAX_PIXELS)
    with open(args.meta) as handle:
        docs = [json.loads(line) for line in handle]
    if args.limit:
        docs = docs[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as output:
        for index, doc in enumerate(docs):
            started = time.time()
            image = Image.open(doc["image"]).convert("RGB")
            question = doc["questions"][0]
            sample_seed = zlib.crc32(
                f"{args.seed}:M2B:{doc['docId']}:{question['qid']}".encode()
            ) & 0x7FFFFFFF
            record = evaluate_one(
                model,
                processor,
                image,
                question["q"],
                question["answers"][0],
                args.device,
                args.reference_bits,
                args.hybrid_bits,
                sample_seed,
            )
            record.update(
                {
                    "docId": doc["docId"],
                    "qid": question["qid"],
                    "question": question["q"],
                    "answer": question["answers"][0],
                    "schema_version": "legacy-m2-family-smoke-v1",
                    "stage": "M2B",
                    "run_kind": "smoke",
                    "base_seed": args.seed,
                }
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"{index + 1}/{len(docs)} doc={doc['docId']} visual={record['n_visual']} "
                f"target={record['target_ratio_to_full']:.3f} time={time.time() - started:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=META)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--reference-bits", type=int, default=4)
    parser.add_argument("--hybrid-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
