"""Input-gate verification for the gated visual memory design.

Step 1 — signal validation: do cheap, write-time signals (OCR text bytes,
resolution, source container bytes) predict how many bytes an image needs
so that answers that were correct at full quality stay correct?

Step 2 — budget economy: at an equal TOTAL byte budget, does adaptive
per-image allocation (driven only by the cheap signal) beat uniform
allocation, and how close does it get to the measured oracle?

Inputs (all pre-existing measurements; no GPU, no model calls):
  results/discovery/d1_codec_{sqa,gqa}.jsonl   per-image per-budget scores
  results/discovery/ocr_pkg_{sqa,gqa}_plain.jsonl  per-image OCR text bytes

Output: results/discovery/gates/input_gate.json (+ printed summary)
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "discovery" / "gates"

BUDGET_GRID = [2048, 8192, 32768, 131072]


def load_codec(ds: str):
    """Return per-image data: full-correct question ids, retention at each
    feasible budget (best codec), source bytes, resolution."""
    per_q = defaultdict(dict)  # (sid, qid) -> {"full": s, (codec,b): s}
    src = {}
    for line in open(ROOT / f"results/discovery/d1_codec_{ds}.jsonl"):
        r = json.loads(line)
        if r["record_type"] == "representation_feasibility":
            if r["condition_id"] == "SOURCE_IMAGE":
                src[r["sample_id"]] = {
                    "bytes": r["package_bytes"],
                    "pixels": r["source_width"] * r["source_height"],
                }
        elif r["record_type"] == "question_result":
            key = (r["sample_id"], r["question_id"])
            if r["condition_id"] == "SOURCE_IMAGE":
                per_q[key]["full"] = r["task_score"]
            else:
                per_q[key][(r["codec"], r["target_bytes"])] = r["task_score"]

    images = {}
    for (sid, qid), scores in per_q.items():
        if scores.get("full", 0.0) < 1.0:
            continue  # retention is conditional on full-quality correctness
        img = images.setdefault(sid, {"budgets": defaultdict(list), "n_q": 0})
        img["n_q"] += 1
        for b in BUDGET_GRID:
            cand = [scores[(c, b)] for c in ("AVIF", "JPEG") if (c, b) in scores]
            if cand:
                img["budgets"][b].append(max(cand))
    out = {}
    for sid, img in images.items():
        if sid not in src or img["n_q"] == 0:
            continue
        ret = {b: sum(v) / len(v) for b, v in img["budgets"].items()}
        out[sid] = {
            "retention": ret,          # budget -> mean retention (best codec)
            "n_q": img["n_q"],
            "src_bytes": src[sid]["bytes"],
            "pixels": src[sid]["pixels"],
        }
    return out


def load_text_bytes(ds: str):
    tb = {}
    for line in open(ROOT / f"results/discovery/ocr_pkg_{ds}_plain.jsonl"):
        r = json.loads(line)
        if r.get("record_type") == "materialized_text_memory_package":
            rep = r["representations"]["plain"]
            tb[r["sample_id"]] = rep["original_utf8_bytes"]
    return tb


def needed_budget(img, thresh=1.0):
    """Smallest grid budget whose retention >= thresh; None if no budget
    (including infeasible ones) reaches it below the source size."""
    for b in BUDGET_GRID:
        if b in img["retention"] and img["retention"][b] >= thresh:
            return b
    return None  # needs more than 128KiB (or full source)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def auroc(pos, neg):
    """P(signal_pos > signal_neg) with tie=0.5."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


# ---------------- Step 2: allocation policies ----------------

def concave_hull(pts):
    """Upper concave envelope of monotone (bytes, value) points so that the
    greedy allocator sees strictly decreasing marginal gain per byte and can
    jump over zero-gain intermediate steps."""
    hull = []
    for p in sorted(pts):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # drop middle point if it lies below the chord to p
            if (y2 - y1) * (p[0] - x1) <= (p[1] - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(p)
    # enforce monotone value (drop points that add bytes without gain)
    out = [hull[0]]
    for p in hull[1:]:
        if p[1] > out[-1][1]:
            out.append(p)
    return out


def envelope_points(img):
    """(bytes, retention) points available to the allocator, made concave
    and monotone. Always includes (0, 0.0) meaning 'store nothing'."""
    pts = [(0, 0.0)]
    best = 0.0
    for b in BUDGET_GRID:
        if b in img["retention"]:
            best = max(best, img["retention"][b])
            pts.append((b, best))
    return concave_hull(pts)


def greedy_allocate(imgs, curves, total_bytes):
    """Greedy marginal-gain-per-byte allocation over per-image step curves.
    curves[sid] = list of (bytes, value) monotone points. Returns
    {sid: chosen_bytes} with sum(chosen) <= total_bytes."""
    state = {sid: 0 for sid in imgs}   # index into curve (0 = (0,0))
    spent = 0
    heap = []
    import heapq
    for sid in imgs:
        c = curves[sid]
        if len(c) > 1:
            db = c[1][0] - c[0][0]
            dv = c[1][1] - c[0][1]
            heapq.heappush(heap, (-dv / db if db else 0.0, sid, 1))
    while heap:
        negrate, sid, nxt = heapq.heappop(heap)
        c = curves[sid]
        cur = state[sid]
        if nxt != cur + 1:
            continue  # stale entry
        cost = c[nxt][0] - c[cur][0]
        if spent + cost > total_bytes:
            continue
        state[sid] = nxt
        spent += cost
        if nxt + 1 < len(c):
            db = c[nxt + 1][0] - c[nxt][0]
            dv = c[nxt + 1][1] - c[nxt][1]
            heapq.heappush(heap, (-dv / db if db else 0.0, sid, nxt + 1))
    return {sid: curves[sid][state[sid]][0] for sid in imgs}, spent


def realized_value(imgs, alloc):
    """Mean measured retention when each image is stored at alloc[sid] bytes
    (largest measured point <= alloc; 0 if nothing fits)."""
    vals = []
    for sid, img in imgs.items():
        best = 0.0
        for b, r in img["retention"].items():
            if b <= alloc[sid]:
                best = max(best, r)
        vals.append(best)
    return sum(vals) / len(vals)


def signal_curves(imgs, text_bytes, n_bins=3, n_folds=5, seed=13):
    """Cross-validated predicted curves: bin images by OCR text bytes,
    predicted retention(budget) for an image = training-fold mean retention
    of its bin at that budget, conditioned on feasibility. The candidate
    byte points for an image are its own FEASIBLE budgets (write-time
    observable: the writer sees whether the compressor fits the cap).
    The image's own retention scores are never used."""
    sids = sorted(imgs)
    rng = random.Random(seed)
    rng.shuffle(sids)
    folds = [sids[i::n_folds] for i in range(n_folds)]
    # global bin edges from signal quantiles
    sig = sorted(text_bytes.get(s, 0) for s in sids)
    edges = [sig[len(sig) * (i + 1) // n_bins - 1] for i in range(n_bins - 1)]

    def bin_of(s):
        t = text_bytes.get(s, 0)
        for i, e in enumerate(edges):
            if t <= e:
                return i
        return n_bins - 1

    curves = {}
    for fi, fold in enumerate(folds):
        train = [s for s in sids if s not in set(fold)]
        # train-mean retention per bin per budget, conditioned on the
        # package actually fitting the cap (feasible images only)
        agg = {b: defaultdict(list) for b in BUDGET_GRID}
        for s in train:
            bn = bin_of(s)
            for b in BUDGET_GRID:
                if b in imgs[s]["retention"]:
                    agg[b][bn].append(imgs[s]["retention"][b])
        for s in fold:
            bn = bin_of(s)
            pts = [(0, 0.0)]
            best = 0.0
            for b in BUDGET_GRID:
                if b not in imgs[s]["retention"]:
                    continue  # infeasible for this image: writer observes this
                if agg[b][bn]:
                    m = sum(agg[b][bn]) / len(agg[b][bn])
                    best = max(best, m)
                    pts.append((b, best))
            curves[s] = concave_hull(pts)
    return curves


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for ds in ("sqa", "gqa"):
        imgs = load_codec(ds)
        tb = load_text_bytes(ds)
        sids = [s for s in imgs if s in tb]
        imgs = {s: imgs[s] for s in sids}
        print(f"\n===== {ds.upper()} — {len(imgs)} images with full-correct questions =====")

        # ---- Step 1: needed budget & signals ----
        need = {s: needed_budget(imgs[s]) for s in sids}
        dist = defaultdict(int)
        for s in sids:
            dist[need[s] if need[s] else ">131072"] += 1
        print("needed-budget distribution:", dict(sorted(dist.items(), key=lambda kv: str(kv[0]))))

        # correlation on images with a defined needed budget
        defined = [s for s in sids if need[s]]
        signals = {
            "ocr_text_bytes": lambda s: tb[s],
            "source_bytes": lambda s: imgs[s]["src_bytes"],
            "pixels": lambda s: imgs[s]["pixels"],
            "bytes_per_pixel": lambda s: imgs[s]["src_bytes"] / imgs[s]["pixels"],
        }
        corr = {}
        for name, fn in signals.items():
            corr[name] = round(spearman([fn(s) for s in defined], [need[s] for s in defined]), 3)
        print("Spearman(signal, needed_budget):", corr)

        # AUROC: does the signal separate "32KiB is enough" vs "needs more"?
        small_ok = [s for s in sids if need[s] is not None and need[s] <= 32768]
        needs_big = [s for s in sids if need[s] is None or need[s] > 32768]
        aur = {}
        for name, fn in signals.items():
            aur[name] = auroc([fn(s) for s in needs_big], [fn(s) for s in small_ok])
            aur[name] = round(aur[name], 3) if aur[name] is not None else None
        print(f"AUROC(needs >32KiB | signal)  n_small_ok={len(small_ok)} n_needs_big={len(needs_big)}:", aur)

        # ---- Step 2: allocation at equal total bytes ----
        oracle_curves = {s: envelope_points(imgs[s]) for s in sids}
        sig_curves = signal_curves(imgs, tb)
        alloc_rows = []
        for B in (8192, 32768):
            total = B * len(sids)
            row = {"per_image_budget": B}
            # uniform
            row["uniform"] = round(realized_value(imgs, {s: B for s in sids}), 4)
            # signal-adaptive (cross-validated bin curves)
            a, spent = greedy_allocate(imgs, sig_curves, total)
            row["signal_adaptive"] = round(realized_value(imgs, a), 4)
            row["signal_spent_frac"] = round(spent / total, 3)
            # oracle (measured curves)
            a, spent = greedy_allocate(imgs, oracle_curves, total)
            row["oracle"] = round(realized_value(imgs, a), 4)
            row["oracle_spent_frac"] = round(spent / total, 3)
            alloc_rows.append(row)
            print(f"total={B}B/image  uniform={row['uniform']}  "
                  f"signal={row['signal_adaptive']}  oracle={row['oracle']}")

        report[ds] = {
            "n_images": len(imgs),
            "needed_budget_distribution": {str(k): v for k, v in dist.items()},
            "spearman_needed_budget": corr,
            "auroc_needs_more_than_32k": aur,
            "n_small_ok": len(small_ok),
            "n_needs_big": len(needs_big),
            "allocation": alloc_rows,
        }

    with open(OUT_DIR / "input_gate.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nwrote", OUT_DIR / "input_gate.json")


if __name__ == "__main__":
    main()
