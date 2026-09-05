"""Same-image multi-turn chat with image-initialized recurrent KV retention.

python -m vlm_diagnosis.exps.recurrent_session --device cuda:1 --limit 1 --steps 3

This runner is separate from the legacy independent-question/full-trace sweep.
It performs one image prefill and irreversibly deletes unselected KV by default.
Use --storage offload explicitly for the older uncompressed CPU reservoir.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image
import torch

from vlm_diagnosis.core.loader import load_vlm
from vlm_diagnosis.core.metrics import anls, exact_match, normalize_text
from vlm_diagnosis.core.session_cache import MultimodalSession
from vlm_diagnosis.core.session_adapters import QwenImageAdapter, QwenPairAdapter
from vlm_diagnosis.core.pair_session import PairSession

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    parser.add_argument("--model", choices=["qwen25vl", "qwen3vl"], default="qwen25vl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--budget", type=float, default=0.2,
                        help="fixed retained fraction of initial KV; global pairs or common tokens")
    parser.add_argument("--granularity", choices=["kv_pair", "token"], default="kv_pair",
                        help="global layer/KV-head/token pair selection (default), or legacy common-token selection")
    parser.add_argument("--prior-floor", "--image-floor", dest="prior_floor", type=float, default=0.35,
                        help="initial modality-prior weight floor (--image-floor is a legacy alias)")
    parser.add_argument("--adapter", choices=["qwen_image"], default="qwen_image",
                        help="only the image adapter has native model integration currently")
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--conditions", default="full,image_static,recurrent")
    parser.add_argument("--storage", choices=["delete", "offload"], default="delete",
                        help="delete evicted KV permanently (default), or retain all KV on CPU")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=1280*28*28)
    parser.add_argument("--out", default="results/smoke/recurrent_session_pairs.jsonl")
    args = parser.parse_args()
    if not 0 < args.budget <= 1 or args.steps < 1 or args.max_new_tokens < 1:
        parser.error("budget must be in (0,1]; steps and max-new-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.granularity == "kv_pair" and args.storage != "delete":
        parser.error("kv_pair mode physically deletes KV; offload is only supported with --granularity token")
    if not 0 <= args.prior_floor <= 1 or not 0 <= args.decay <= 1 or args.max_pixels < 1:
        parser.error("prior-floor and decay must be in [0,1]; max-pixels must be positive")
    conditions = args.conditions.split(",")
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= {"full", "image_static", "recurrent"}:
        parser.error("conditions must be unique members of full,image_static,recurrent")
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Fail rather than silently overwriting another experiment.
    if out_path.exists():
        parser.error(f"output already exists: {out_path}; choose a new --out")
    with (ROOT / args.manifest).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    model, processor = load_vlm(args.model, device=args.device, max_pixels=args.max_pixels)
    is_pair = args.granularity == "kv_pair"
    adapter = QwenPairAdapter() if is_pair else QwenImageAdapter()
    run_id = "session-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with out_path.open("x", encoding="utf-8") as handle:
        metadata = {
            "record_type": "run_metadata", "schema_version": "2.0" if is_pair else "1.1", "run_id": run_id,
            "stage": "RECURRENT_PAIRS" if is_pair else "RECURRENT_SESSION", **vars(args),
            "storage_mode": args.storage,
            "implementation_version": "ragged_pairs_v1" if is_pair else "modality_adapter_v1",
            "adapter_id": adapter.adapter_id,
            "supported_modalities": list(adapter.supported_modalities),
            "initial_signal": "actual visual-row decoder attention from one image-only prefill",
            "granularity": "kv_pair" if is_pair else "common token positions across all layers and KV heads",
            "selection_timing": "committed before the next question; updated after own completed answer",
            "history": "each condition has its own generated-answer KV trajectory",
            "termination": "generated content IDs preserved; EOS alternatives normalized to canonical assistant ending",
            "baseline_semantics": "FULL is an independent full-cache trajectory, not matched-history intervention",
            "budget_semantics": "one GLOBAL (layer,KV-head,token) pair cap; sinks included; no equal head quotas" if is_pair else (
                "fixed initial-prefix token cap for all persistent compressed KV; current turn and compaction copies extra"
                if args.storage == "delete" else
                "fixed initial-prefix token cap for historical GPU KV; current turn tokens extra"),
            "cold_storage": ("none; evicted KV and per-token state permanently deleted"
                             if args.storage == "delete" else
                             "all original image and own historical text KV retained uncompressed on CPU"),
            "seed_lifetime": "full CPU prefill seed released after independent condition caches are constructed",
            "gpu_peak_scope": "absolute process allocation including shared model and other condition caches; not per-method memory",
            "scoring": "actual selected-cache eager attention; no full-cache teacher trace or scoring forward",
            "gate": "training-free data-dependent recurrence; not a trained LSTM",
            "attention_backend": "ragged per-KV-head eager reference; not fused or throughput optimized" if is_pair else "dense eager",
        }
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for row in rows:
            with Image.open(ROOT / row["image"]) as raw:
                seed = adapter.prefill(model, processor, raw.convert("RGB"), args.device)
            initial_prefix_tokens = seed.prefix_ids.shape[1]
            image_prefill_seconds = seed.prefill_seconds
            budget = round((seed.pair_prior_scores.numel() if is_pair else initial_prefix_tokens) * args.budget)
            if is_pair:
                sessions = {condition: PairSession(
                    model, processor, seed, args.device, budget, condition,
                    prior_floor=args.prior_floor, decay=args.decay, adapter=adapter)
                    for condition in conditions}
            else:
                sessions = {condition: MultimodalSession(
                    model, processor, seed, args.device, budget, condition,
                    prior_floor=args.prior_floor, decay=args.decay, storage=args.storage, adapter=adapter)
                    for condition in conditions}
            # A compressed session must not keep the temporary full CPU seed
            # alive via this runner. FULL owns only its independent baseline.
            del seed
            for step, question in enumerate(row["questions"][:args.steps], start=1):
                turn_records = []
                for condition, session in sessions.items():
                    if torch.device(args.device).type == "cuda":
                        torch.cuda.reset_peak_memory_stats(args.device)
                    result = session.answer(question["question"] + " Answer with a single word or phrase.",
                                            max_new_tokens=args.max_new_tokens)
                    result.update({
                        "run_id": run_id, "sample_id": str(row["sample_id"]),
                        "dataset": row["dataset"], "question_id": question["question_id"],
                        "model": args.model, "gold": question["answers"],
                        ("budget_pairs" if is_pair else "budget_tokens"): budget,
                        "initial_prefix_tokens": initial_prefix_tokens,
                        "image_prefill_seconds": image_prefill_seconds,
                        "em": exact_match(result["prediction"], question["answers"]),
                        "anls": anls(result["prediction"], question["answers"]),
                    })
                    if torch.device(args.device).type == "cuda":
                        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
                    turn_records.append(result)
                full = next((r for r in turn_records if r["condition_id"] == "full"), None)
                for result in turn_records:
                    if full is not None:
                        result["full_em"] = full["em"]
                        result["full_anls"] = full["anls"]
                        result["full_correct_retained"] = result["em"] if full["em"] == 1 else None
                        result["loyalty"] = float(normalize_text(result["prediction"]) == normalize_text(full["prediction"]))
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if is_pair:
                        counts = result["selection_after"]["pairs_by_group"]
                        print(f"{row['sample_id']} t={step} {result['condition_id']} "
                              f"EM={result['em']:.3f} pairs={result['retained_kv_pairs']} "
                              f"head_range={min(counts)}..{max(counts)} "
                              f"deleted={result['deleted_pairs_this_turn']} {result['prediction']!r}", flush=True)
                        continue
                    print(f"{row['sample_id']} t={step} {result['condition_id']} "
                          f"EM={result['em']:.3f} hot={result['active_history_tokens']} "
                          f"next={result['next_active_history_tokens']} "
                          f"history_text={result['selection_after']['selected_history_text_tokens']} "
                          f"deleted={result['deleted_tokens_this_turn']} "
                          f"entered={result['entered_tokens']} {result['prediction']!r}", flush=True)
                handle.flush()
            # Do not overlap a previous image's resident conditions with the
            # next image prefill. The inner loop variable also owns a session.
            sessions.clear()
            session = None
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
