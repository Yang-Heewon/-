"""Forget-gate verification: can a cheap pixel-level detector decide, at
write time, whether a new observation of the same screen changed the stored
fact — so the gate can invalidate the stale memory?

Fixtures: memory_dynamics controlled (32 ep, seed 42) + scaled (64 ep,
seed 777). Each episode has target_old.png / target_current.png plus ground
truth: factorial.state_changed and the queried field's evidence_bbox.

Screens deliberately render a snapshot id + capture timestamp that differ in
EVERY pair, so "any pixel differs" must fail on unchanged episodes; the
detector has to separate content-region change from metadata change.

Detectors (both tile-grid mean-abs-diff, 16px tiles, grayscale):
  naive        — change score = max tile diff anywhere
  layout-aware — change score = max tile diff inside the content area
                 (the four field-value rows, y in [244, 776))

Outputs episode-level ROC/accuracy, localization hit rate, and per-pair
wall time. CPU only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "discovery" / "gates" / "forget_gate.json"

TILE = 16
CONTENT_Y = (244, 776)  # field rows region per generator layout
CONTENT_X = (72, 696)


def tile_diff(a: Image.Image, b: Image.Image):
    ga, gb = a.convert("L"), b.convert("L")
    pa, pb = ga.load(), gb.load()
    w, h = ga.size
    tiles = {}
    for ty in range(0, h - TILE + 1, TILE):
        for tx in range(0, w - TILE + 1, TILE):
            s = 0
            for y in range(ty, ty + TILE, 4):       # 4px subsampling
                for x in range(tx, tx + TILE, 4):
                    s += abs(pa[x, y] - pb[x, y])
            tiles[(tx, ty)] = s / ((TILE // 4) ** 2)
    return tiles


def in_box(tx, ty, box):
    x1, y1, x2, y2 = box
    return tx + TILE > x1 and tx < x2 and ty + TILE > y1 and ty < y2


def auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def load_episodes():
    eps = []
    # NB: both fixtures reuse the same episode id prefix, so the image
    # directory must be pinned per manifest rather than searched.
    for manifest, data_dir in (
        ("experiments/manifests/memory_dynamics_controlled.jsonl",
         "data/memory_dynamics_controlled"),
        ("experiments/manifests/md_scaled_read.jsonl", "data/md_scaled"),
    ):
        for line in open(ROOT / manifest):
            r = json.loads(line)
            ep = r["episode_id"]
            changed = bool(r["factorial"]["state_changed"])
            bbox = r["question"]["current_evidence"]["evidence_bbox"]
            old = ROOT / data_dir / f"{ep}_target_old.png"
            cur = ROOT / data_dir / f"{ep}_target_current.png"
            if old.exists() and cur.exists():
                eps.append({
                    "episode": ep, "fixture": data_dir.split("/")[-1],
                    "changed": changed, "bbox": bbox,
                    "old": old, "cur": cur,
                })
    # manifests may repeat episode ids across fixtures; key on (fixture, ep)
    seen = set()
    uniq = []
    for e in eps:
        k = (e["fixture"], e["episode"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq


def main():
    eps = load_episodes()
    print(f"{len(eps)} old/current pairs "
          f"({sum(e['changed'] for e in eps)} changed, "
          f"{sum(not e['changed'] for e in eps)} unchanged)")

    scores = {"naive": {"pos": [], "neg": []},
              "layout": {"pos": [], "neg": []}}
    loc_hits = 0
    loc_total = 0
    t0 = time.perf_counter()
    for e in eps:
        tiles = tile_diff(Image.open(e["old"]), Image.open(e["cur"]))
        naive = max(tiles.values())
        content = {k: v for k, v in tiles.items()
                   if in_box(k[0], k[1], (*CONTENT_X[:1], CONTENT_Y[0],
                                          CONTENT_X[1], CONTENT_Y[1]))}
        layout = max(content.values()) if content else 0.0
        bucket = "pos" if e["changed"] else "neg"
        scores["naive"][bucket].append(naive)
        scores["layout"][bucket].append(layout)
        if e["changed"]:
            loc_total += 1
            (tx, ty) = max(content, key=content.get)
            if in_box(tx, ty, e["bbox"]):
                loc_hits += 1
    elapsed = time.perf_counter() - t0

    report = {"n_pairs": len(eps),
              "ms_per_pair": round(1000 * elapsed / len(eps), 2)}
    for name in ("naive", "layout"):
        pos, neg = scores[name]["pos"], scores[name]["neg"]
        a = auroc(pos, neg)
        # accuracy at the best single threshold (synthetic separability check)
        cands = sorted(set(pos + neg))
        best = 0.0
        for t in cands:
            acc = (sum(p > t for p in pos) + sum(n <= t for n in neg)) / (len(pos) + len(neg))
            best = max(best, acc)
        report[name] = {
            "auroc": round(a, 4) if a is not None else None,
            "best_threshold_accuracy": round(best, 4),
            "changed_score_min": round(min(pos), 2) if pos else None,
            "unchanged_score_max": round(max(neg), 2) if neg else None,
        }
    report["localization_hit_rate"] = round(loc_hits / loc_total, 4) if loc_total else None
    report["localization_hits"] = f"{loc_hits}/{loc_total}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
