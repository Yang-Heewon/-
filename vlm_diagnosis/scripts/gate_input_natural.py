"""Input-gate signals for NATURAL images (GQA): cheap write-time visual
signals computed with PIL only, tested as predictors of needed_budget and
as drivers of adaptive byte allocation.

Reuses load_codec / needed_budget / spearman / signal_curves /
greedy_allocate / envelope_points / realized_value from
gate_input_analysis.py. CPU-only, no models.

Output: results/discovery/gates/input_gate_natural.json
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageFilter

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from gate_input_analysis import (  # noqa: E402
    ROOT,
    OUT_DIR,
    load_codec,
    needed_budget,
    spearman,
    signal_curves,
    greedy_allocate,
    envelope_points,
    realized_value,
)

MANIFEST = ROOT / "experiments/manifests/gqa_discovery.jsonl"


def image_paths():
    paths = {}
    for line in open(MANIFEST):
        r = json.loads(line)
        paths[r["sample_id"]] = ROOT / r["image"]
    return paths


def hist_entropy(hist):
    total = sum(hist)
    if total == 0:
        return 0.0
    h = 0.0
    for c in hist:
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def jpeg_bytes(im, quality):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.tell()


def tile_std_mean(gray, tile=16):
    w, h = gray.size
    px = gray.load()
    stds = []
    for ty in range(0, h - tile + 1, tile):
        for tx in range(0, w - tile + 1, tile):
            s = 0.0
            s2 = 0.0
            n = tile * tile
            for y in range(ty, ty + tile):
                for x in range(tx, tx + tile):
                    v = px[x, y]
                    s += v
                    s2 += v * v
            m = s / n
            var = max(s2 / n - m * m, 0.0)
            stds.append(math.sqrt(var))
    return sum(stds) / len(stds) if stds else 0.0


def compute_signals(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    pixels = w * h
    gray = im.convert("L")

    sig = {}
    # recompression sizes (in-memory)
    q50 = jpeg_bytes(im, 50)
    q30 = jpeg_bytes(im, 30)
    sig["jpeg_q50_bytes"] = q50
    sig["jpeg_q30_bytes"] = q30
    sig["jpeg_q50_bpp"] = q50 / pixels  # bits-ish per pixel (bytes/pixel)

    # thumbnail compressibility: max dim 64
    th = im.copy()
    th.thumbnail((64, 64))
    sig["thumb64_jpeg_bytes"] = jpeg_bytes(th, 50)

    # edge density: mean of FIND_EDGES response on grayscale
    edges = gray.filter(ImageFilter.FIND_EDGES)
    ehist = edges.histogram()
    n = sum(ehist)
    sig["edge_density"] = sum(i * c for i, c in enumerate(ehist)) / n

    # entropies
    sig["gray_entropy"] = hist_entropy(gray.histogram())
    rhist = im.histogram()  # 768 bins: R,G,B
    sig["rgb_entropy"] = (
        hist_entropy(rhist[0:256])
        + hist_entropy(rhist[256:512])
        + hist_entropy(rhist[512:768])
    ) / 3.0

    # local variance: mean per-16x16-tile grayscale std (on a bounded-size copy
    # for speed; structure statistic, resolution-normalized)
    gsmall = gray.copy()
    gsmall.thumbnail((256, 256))
    sig["tile16_std"] = tile_std_mean(gsmall, 16)

    # extra ideas -----------------------------------------------------
    # high-frequency energy: mean |gray - blurred(gray)|
    from PIL import ImageChops

    blur = gsmall.filter(ImageFilter.GaussianBlur(2))
    diff = ImageChops.difference(gsmall, blur)
    dh = diff.histogram()
    sig["hf_energy"] = sum(i * c for i, c in enumerate(dh)) / sum(dh)

    # color diversity: entropy of 4x4x4 quantized RGB cube (on small copy)
    small = im.copy()
    small.thumbnail((128, 128))
    counts = {}
    for r, g, b in small.getdata():
        key = (r >> 6, g >> 6, b >> 6)
        counts[key] = counts.get(key, 0) + 1
    sig["color_cube_entropy"] = hist_entropy(list(counts.values()))

    # quality-drop ratio: how much q30 saves over q50 (texture-vs-flat cue)
    sig["q30_over_q50"] = q30 / q50 if q50 else 0.0

    return sig


def rank_of(values):
    """values: {sid: v} -> {sid: average rank}"""
    sids = sorted(values, key=lambda s: values[s])
    ranks = {}
    i = 0
    while i < len(sids):
        j = i
        while j + 1 < len(sids) and values[sids[j + 1]] == values[sids[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sids[k]] = avg
        i = j + 1
    return ranks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imgs = load_codec("gqa")
    paths = image_paths()
    missing = [s for s in imgs if s not in paths or not paths[s].exists()]
    sids = [s for s in imgs if s in paths and paths[s].exists()]
    imgs = {s: imgs[s] for s in sids}
    print(f"GQA images with full-correct questions: {len(imgs)} (missing files: {len(missing)})")

    # ---- compute signals ----
    signals = {}  # name -> {sid: value}
    for s in sids:
        for name, v in compute_signals(paths[s]).items():
            signals.setdefault(name, {})[s] = v
    # baselines from records
    signals["source_bytes"] = {s: imgs[s]["src_bytes"] for s in sids}
    signals["pixels"] = {s: imgs[s]["pixels"] for s in sids}

    # ---- Spearman vs needed budget ----
    need = {s: needed_budget(imgs[s]) for s in sids}
    defined = [s for s in sids if need[s]]
    print(f"images with defined needed_budget: {len(defined)}")
    corr = {}
    for name, vals in signals.items():
        corr[name] = round(
            spearman([vals[s] for s in defined], [need[s] for s in defined]), 3
        )
    for name, r in sorted(corr.items(), key=lambda kv: -abs(kv[1])):
        print(f"  spearman {name:22s} {r:+.3f}")

    # ---- 2-signal rank-sum combinations over the top single signals ----
    top = [n for n, _ in sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:5]]
    combos = {}
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            ra, rb = rank_of(signals[a]), rank_of(signals[b])
            # orient each signal so higher rank = more bytes needed
            sa = 1.0 if corr[a] >= 0 else -1.0
            sb = 1.0 if corr[b] >= 0 else -1.0
            comb = {s: sa * ra[s] + sb * rb[s] for s in sids}
            name = f"ranksum({a},{b})"
            combos[name] = comb
            corr[name] = round(
                spearman([comb[s] for s in defined], [need[s] for s in defined]), 3
            )
            print(f"  spearman {name:48s} {corr[name]:+.3f}")
    signals.update(combos)

    # ---- allocation simulation ----
    oracle_curves = {s: envelope_points(imgs[s]) for s in sids}
    alloc_rows = []
    # policies: uniform, oracle, each signal (top singles + combos), 3 and 4 bins
    policy_signals = {n: signals[n] for n in top}
    policy_signals.update(combos)
    for B in (8192, 32768):
        total = B * len(sids)
        row = {
            "budget": B,
            "policy": "uniform",
            "value": round(realized_value(imgs, {s: B for s in sids}), 4),
        }
        alloc_rows.append(row)
        print(f"\nB={B}  uniform={row['value']}")
        for name, vals in policy_signals.items():
            for nb in (3, 4):
                curves = signal_curves(imgs, vals, n_bins=nb)
                a, spent = greedy_allocate(imgs, curves, total)
                v = round(realized_value(imgs, a), 4)
                alloc_rows.append(
                    {
                        "budget": B,
                        "policy": f"{name}[{nb}bins]",
                        "value": v,
                        "spent_frac": round(spent / total, 3),
                    }
                )
                print(f"B={B}  {name}[{nb}bins] = {v} (spent {spent/total:.3f})")
        a, spent = greedy_allocate(imgs, oracle_curves, total)
        v = round(realized_value(imgs, a), 4)
        alloc_rows.append(
            {"budget": B, "policy": "oracle", "value": v, "spent_frac": round(spent / total, 3)}
        )
        print(f"B={B}  oracle={v}")

    out = {
        "n_images": len(imgs),
        "n_defined_needed_budget": len(defined),
        "signals": corr,
        "allocation": alloc_rows,
        "notes": (
            "Natural-image (GQA) cheap write-time signals vs needed_budget "
            "(smallest grid budget with retention>=1.0). Signals computed with "
            "PIL only from data/gqa_pilot sources. Allocation uses the same "
            "5-fold cross-validated bin-curve policy as gate_input_analysis "
            "signal_curves(), greedy marginal-gain allocation, realized on "
            "measured retention points."
        ),
    }
    with open(OUT_DIR / "input_gate_natural.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote", OUT_DIR / "input_gate_natural.json")


if __name__ == "__main__":
    main()
