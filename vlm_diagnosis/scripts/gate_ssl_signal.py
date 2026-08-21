"""Self-supervised input-gate signal via the VLM's own vision encoder.

Idea: the supervised "needed budget" label asks whether the ANSWER survives
compression; a label-free proxy asks whether the pretrained vision encoder's
REPRESSENTATION survives it. For each image and byte budget we AVIF-encode to
the cap, run the frozen encoder on original vs compressed, and measure token
embedding distortion. No questions, no generation, no training.

Evaluated with the exact same harness as the pixel-signal study
(gate_input_natural / gate_input_analysis): Spearman vs measured needed
budget, then 5-fold cross-validated bin-curve allocation at equal total
bytes. Also measures the serving cost of the gate itself (write-path
latency, throughput, VRAM) against the supervised alternative (a full VLM
QA call, timed from the D1 records).

Evidence grade: discovery (exploratory signal study, not preregistered).

Usage:
  python -m vlm_diagnosis.scripts.gate_ssl_signal --device cuda:0 \
      --out results/discovery/gates/ssl_signal_gqa.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from io import BytesIO

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vlm_diagnosis" / "scripts"))

from gate_input_analysis import (  # noqa: E402
    load_codec, load_text_bytes, needed_budget, spearman,
    signal_curves, greedy_allocate, envelope_points, realized_value,
)
from vlm_diagnosis.core.byte_codecs import encode_image_to_budget  # noqa: E402
from vlm_diagnosis.core.loader import load_vlm  # noqa: E402

BUDGETS = (2048, 8192, 32768)
SENTINEL = 1e9  # distortion rank for images infeasible at the probe budget


def image_paths(ds: str) -> dict[str, Path]:
    paths = {}
    for line in open(REPO / f"experiments/manifests/{'gqa' if ds == 'gqa' else 'screenqa'}_discovery.jsonl"):
        r = json.loads(line)
        paths[r["sample_id"]] = REPO / r["image"]
    return paths


@torch.inference_mode()
def embed(model, processor, img: Image.Image, device: str) -> torch.Tensor:
    """Frozen vision-tower token embeddings for one image."""
    feats = processor.image_processor(images=[img], return_tensors="pt")
    pv = feats["pixel_values"].to(device, torch.float16)
    grid = feats["image_grid_thw"].to(device)
    return model.visual(pv, grid_thw=grid).float()


def distortion(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    d = 1.0 - cos
    return {
        "mean": d.mean().item(),
        "p90": d.quantile(0.9).item(),
        "frac_hi": (d > 0.1).float().mean().item(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="gqa")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = args.dataset
    imgs = load_codec(ds)
    tb = load_text_bytes(ds)
    paths = image_paths(ds)
    sids = [s for s in imgs if s in tb and s in paths and paths[s].exists()]
    imgs = {s: imgs[s] for s in sids}
    print(f"{ds}: {len(sids)} labeled images")

    model, processor = load_vlm("qwen25vl", device=args.device)

    # ---- pass 1: distortion curves + timing ----
    per_image: dict[str, dict] = {}
    t_compress_all, t_embed_all = [], []
    torch.cuda.reset_peak_memory_stats()
    for n, sid in enumerate(sids, 1):
        src = Image.open(paths[sid]).convert("RGB")
        t0 = time.perf_counter()
        e_src = embed(model, processor, src, args.device)
        t_embed_all.append(time.perf_counter() - t0)
        row = {}
        for b in BUDGETS:
            t0 = time.perf_counter()
            res = encode_image_to_budget(src, "AVIF", b)
            t_compress_all.append(time.perf_counter() - t0)
            if not res.feasible:
                row[b] = None
                continue
            comp = Image.open(BytesIO(res.payload)).convert("RGB")
            t0 = time.perf_counter()
            e_cmp = embed(model, processor, comp, args.device)
            t_embed_all.append(time.perf_counter() - t0)
            row[b] = distortion(e_src, e_cmp)
        per_image[sid] = row
        if n % 40 == 0:
            print(f"  [{n}/{len(sids)}] embedded")
    vram_gb = torch.cuda.max_memory_allocated() / 2**30

    # ---- pass 2: signal evaluation (same harness as the pixel-signal study) ----
    need = {s: needed_budget(imgs[s]) for s in sids}
    defined = [s for s in sids if need[s]]
    signals = {}
    for b in (2048, 8192):
        for metric in ("mean", "p90", "frac_hi"):
            name = f"encdist_{metric}@{b}"
            signals[name] = {
                s: (per_image[s][b][metric] if per_image[s][b] else SENTINEL)
                for s in sids
            }
    corr = {
        name: round(spearman([sig[s] for s in defined], [need[s] for s in defined]), 3)
        for name, sig in signals.items()
    }
    print("Spearman vs needed_budget:", corr)

    alloc = {}
    oracle_curves = {s: envelope_points(imgs[s]) for s in sids}
    for B in (8192, 32768):
        total = B * len(sids)
        row = {"uniform": round(realized_value(imgs, {s: B for s in sids}), 4)}
        for name, sig in signals.items():
            a, _ = greedy_allocate(imgs, signal_curves(imgs, sig), total)
            row[name] = round(realized_value(imgs, a), 4)
        a, _ = greedy_allocate(imgs, oracle_curves, total)
        row["oracle"] = round(realized_value(imgs, a), 4)
        alloc[B] = row
        print(f"alloc @{B}:", row)

    # ---- serving reference: measured VLM QA latency from D1 records ----
    qa_secs = []
    for line in open(REPO / f"results/discovery/d1_codec_{ds}.jsonl"):
        r = json.loads(line)
        if r.get("record_type") == "question_result" and r.get("answer_seconds"):
            qa_secs.append(r["answer_seconds"])

    serving = {
        "write_path_per_image_seconds": {
            "compress_avif_median": round(statistics.median(t_compress_all), 3),
            "encoder_forward_median": round(statistics.median(t_embed_all), 3),
            "gate_total_median": round(
                statistics.median(t_compress_all) + 2 * statistics.median(t_embed_all), 3),
        },
        "throughput_images_per_second_one_gpu": round(
            1.0 / (statistics.median(t_compress_all) + 2 * statistics.median(t_embed_all)), 2),
        "vram_peak_gb": round(vram_gb, 1),
        "supervised_alternative_vlm_qa_median_seconds": round(statistics.median(qa_secs), 2),
        "read_path_overhead": "none (gate runs at write time only)",
    }
    print("serving:", json.dumps(serving, indent=2))

    out = {
        "evidence_grade": "discovery (not preregistered)",
        "dataset": ds, "n_images": len(sids),
        "spearman": corr,
        "pixel_signal_reference": {"best_prior_spearman": 0.491, "prior_alloc_gain_pp": 0.0},
        "allocation": alloc,
        "serving": serving,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
