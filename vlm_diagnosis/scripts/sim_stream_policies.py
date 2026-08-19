#!/usr/bin/env python3
"""Memory-policy simulator for the preregistered E2E stream benchmark.

Replays every episode timeline of results/discovery/gates/stream/manifest.jsonl
(produced by gen_stream_bench.py) through 5 storage policies at 2 per-
observation byte budgets B in {8192, 24576} and emits, per (episode, question,
policy, budget), the ordered memory items a GPU orchestrator must inject for
the VLM answering step.  No VLM here; CPU only.

Policies (docs/E2E-STREAM-PREREG.md section 3):
  no_memory        stores nothing.
  keep_everything  every observation as full AVIF within 24576 bytes (or best
                   quality that fits 65536 when 24576 is infeasible); no cap.
  uniform          every observation as full AVIF within B (store-failure if
                   infeasible at min quality); running cap = B x observations
                   so far, FIFO eviction.
  GVM              gated visual memory. New page -> full AVIF, feasibility
                   input gate (8192 cap first, else 24576, else best<=65536).
                   Revisit -> mask(64px header) -> row-profile-NCC alignment
                   -> tile-diff detector against the latest stored state's
                   rendered PNG; unchanged -> store nothing (dedup); changed
                   -> JPEG q85 crop of the changed field row only (true bbox
                   +-36px vertically, x = content row 100..1180), linked to
                   the base snapshot.  Same running cap as uniform; when over
                   cap: drop superseded (non-latest) patches oldest-first,
                   then demote old full snapshots to 25%-area JPEG thumbnails;
                   never delete a page's only item, never touch items written
                   at the current step; a remaining overage is logged as an
                   over_cap_residual event.
  text_only        per observation a perfect-OCR UTF-8 dump ("label: value"
                   for every fully visible field + page title); no cap.

Outputs under results/discovery/gates/stream/:
  read_manifest.jsonl  one row per (episode, question, policy, budget)
  ledger.json          per policy x budget byte/event totals + GVM detector
                       confusion vs ground truth
  events.jsonl         every store/dedup/demote/delete/failure event
  store/               encoded memory payloads (AVIF fulls, JPEG patches and
                       thumbnails)
  verify/              3 sampled changed-revisit patch crops + the aligned
                       old-value crop from the detector reference

Implementation notes (explicit deviations are listed in the ledger):
  * AVIF quality search uses bisection (feasibility probe at q=1, then binary
    search for the highest fitting quality) instead of the exhaustive
    descending scan in byte_codecs.encode_image_to_budget, for speed; the
    returned payload is always measured and within budget.
  * Thumbnail JPEG quality is chosen by byte target (half the item's current
    bytes, quality bisected in [10, 85]) because the prereg fixes only the
    25% area, not a quality.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
STREAM = ROOT / "results" / "discovery" / "gates" / "stream"
MANIFEST = STREAM / "manifest.jsonl"
STORE = STREAM / "store"
VERIFY = STREAM / "verify"
READ_MANIFEST = STREAM / "read_manifest.jsonl"
LEDGER = STREAM / "ledger.json"
EVENTS = STREAM / "events.jsonl"

import sys

sys.path.insert(0, str(ROOT))
import vlm_diagnosis.core.byte_codecs  # noqa: F401,E402  (registers AVIF)

_spec = importlib.util.spec_from_file_location(
    "gate_realgui_probe",
    ROOT / "vlm_diagnosis" / "scripts" / "gate_realgui_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

W, H, HEADER_H = probe.W, probe.H, probe.HEADER_H
TEMPLATES = probe.TEMPLATES
LABELS = {t: {fid: lbl for fid, lbl, _ in TEMPLATES[t][0]} for t in TEMPLATES}
TITLES = {t: TEMPLATES[t][2] for t in TEMPLATES}

SEED = 20260820
BUDGETS = (8192, 24576)
POLICIES = ("no_memory", "keep_everything", "uniform", "GVM", "text_only")
DETECT_THRESHOLD = 3.0   # realgui_forget_gate.json aligned_masked:
                         # changed min 7.38 vs unchanged max 0.06
ROW_X = (100, 1180)      # content row extent (1080px wrap centered in 1280)
PATCH_MARGIN = 36
KEEP_CAP = 24576
KEEP_FALLBACK_CAP = 65536
GVM_TIGHT_CAP = 8192
GVM_FALLBACK_CAP = 24576
PATCH_JPEG_Q = 85


# ---------------------------------------------------------------------------
# byte-bounded encoding (bisection; see module docstring for the deviation)
# ---------------------------------------------------------------------------

def _avif_bytes(im: Image.Image, q: int) -> bytes:
    buf = BytesIO()
    im.save(buf, format="AVIF", quality=q)
    return buf.getvalue()


def budget_avif(im: Image.Image, target: int) -> dict:
    """Highest-quality AVIF payload <= target bytes, by quality bisection."""
    im = im.convert("RGB")
    p_lo = _avif_bytes(im, 1)
    if len(p_lo) > target:
        return {"feasible": False, "smallest_tested_bytes": len(p_lo)}
    p_hi = _avif_bytes(im, 100)
    if len(p_hi) <= target:
        return {"feasible": True, "quality": 100, "payload": p_hi,
                "bytes": len(p_hi)}
    lo, hi, best = 1, 100, (1, p_lo)      # invariant: lo fits, hi does not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        p = _avif_bytes(im, mid)
        if len(p) <= target:
            lo, best = mid, (mid, p)
        else:
            hi = mid
    return {"feasible": True, "quality": best[0], "payload": best[1],
            "bytes": len(best[1])}


def _encode_task(args):
    png, target = args
    res = budget_avif(Image.open(png), target)
    return png, target, res


def budget_jpeg(im: Image.Image, target: int, q_min=10, q_max=85) -> tuple:
    """(quality, payload) for the highest JPEG quality <= target bytes;
    falls back to q_min's payload when even that misses the target."""
    def enc(q):
        buf = BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
        return buf.getvalue()
    p_lo = enc(q_min)
    if len(p_lo) > target:
        return q_min, p_lo
    p_hi = enc(q_max)
    if len(p_hi) <= target:
        return q_max, p_hi
    lo, hi, best = q_min, q_max, (q_min, p_lo)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        p = enc(mid)
        if len(p) <= target:
            lo, best = mid, (mid, p)
        else:
            hi = mid
    return best


# ---------------------------------------------------------------------------
# change detector (mask -> align -> tile diff), reusing probe internals
# ---------------------------------------------------------------------------

def detect_change(ref_png: str, cur_png: str) -> dict:
    old, new = Image.open(ref_png), Image.open(cur_png)
    go = np.asarray(old.convert("L"), dtype=np.float64)[HEADER_H:]
    gn = np.asarray(new.convert("L"), dtype=np.float64)[HEADER_H:]
    d, ncc = probe.estimate_dy(go, gn)
    hh = H - HEADER_H
    ys, ye = max(0, -d), hh - max(0, d)
    new_c = new.crop((0, HEADER_H + ys, W, HEADER_H + ye))
    old_c = old.crop((0, HEADER_H + ys + d, W, HEADER_H + ye + d))
    tiles = probe.tile_diff(old_c, new_c)
    tx, ty = max(tiles, key=tiles.get)
    return {"score": max(tiles.values()), "dy": d, "ncc": round(ncc, 4),
            "top_tile": (tx, ty + HEADER_H + ys)}


# ---------------------------------------------------------------------------
# replay machinery
# ---------------------------------------------------------------------------

class Recorder:
    def __init__(self):
        self.events = []
        self.deleted_keys = set()
        self.thrash = 0

    def ev(self, policy, budget, ep, step, etype, **kw):
        self.events.append({"policy": policy, "budget": budget,
                            "episode": ep, "step": step, "type": etype, **kw})

    def store(self, policy, budget, ep, step, item):
        key = (policy, budget, ep, item["page"], item["kind"],
               item["step"], item["revision_id"])
        if key in self.deleted_keys:
            self.thrash += 1
        self.ev(policy, budget, ep, step, "store", item_id=item["item_id"],
                bytes=item["bytes"], kind=item["kind"], page=item["page"])

    def delete(self, policy, budget, ep, step, item):
        key = (policy, budget, ep, item["page"], item["kind"],
               item["step"], item["revision_id"])
        self.deleted_keys.add(key)
        self.ev(policy, budget, ep, step, "delete", item_id=item["item_id"],
                bytes=item["bytes"], kind=item["kind"], page=item["page"])


def _item(policy, budget, ep, obs, kind, path, nbytes, meta, **extra):
    return {"item_id": f"{policy}|{budget}|{ep}|s{obs['step']:02d}|{kind}",
            "kind": kind, "page": obs["page"], "step": obs["step"],
            "revision_id": obs["revision_id"], "scroll": obs["scroll"],
            "clock": obs["clock"], "path": path, "bytes": nbytes,
            "live": True, "thumb": False, "meta": meta, **extra}


def full_meta(obs, quality, nbytes, cap):
    return (f"captured at step {obs['step'] + 1} of 12, page {obs['page']} "
            f"('{TITLES[obs['page']]}'), revision {obs['revision_id']}, "
            f"full screenshot (AVIF q{quality}, {nbytes} bytes, cap {cap}, "
            f"scroll {obs['scroll']}px, clock {obs['clock']})")


def replay_uniform(rec_ep, budget, cache, recorder, outdir):
    ep = rec_ep["episode_id"]
    items, total, failures = [], 0, 0
    for obs in rec_ep["observations"]:
        step = obs["step"]
        enc = cache[(obs["png"], budget)]
        if not enc["feasible"]:
            failures += 1
            recorder.ev("uniform", budget, ep, step, "store_failure",
                        page=obs["page"],
                        smallest_tested_bytes=enc["smallest_tested_bytes"])
        else:
            path = outdir / ep / f"s{step:02d}_{obs['page']}.avif"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(enc["payload"])
            it = _item("uniform", budget, ep, obs, "full", str(path),
                       enc["bytes"],
                       full_meta(obs, enc["quality"], enc["bytes"], budget))
            items.append(it)
            total += it["bytes"]
            recorder.store("uniform", budget, ep, step, it)
        cap = budget * (step + 1)
        live = [it for it in items if it["live"]]
        while total > cap and live:
            victim = live.pop(0)                      # FIFO
            victim["live"] = False
            total -= victim["bytes"]
            recorder.delete("uniform", budget, ep, step, victim)
    return items, total, {"store_failures": failures}


def replay_keep(rec_ep, cache, recorder, outdir):
    ep = rec_ep["episode_id"]
    items, total, fb = [], 0, 0
    for obs in rec_ep["observations"]:
        step = obs["step"]
        enc = cache[(obs["png"], KEEP_CAP)]
        cap = KEEP_CAP
        if not enc["feasible"]:
            enc = cache[(obs["png"], KEEP_FALLBACK_CAP)]
            cap = KEEP_FALLBACK_CAP
            fb += 1
            assert enc["feasible"], f"{obs['png']} infeasible even at 65536"
        path = outdir / ep / f"s{step:02d}_{obs['page']}.avif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(enc["payload"])
        it = _item("keep_everything", "-", ep, obs, "full", str(path),
                   enc["bytes"],
                   full_meta(obs, enc["quality"], enc["bytes"], cap))
        items.append(it)
        total += it["bytes"]
        recorder.store("keep_everything", "-", ep, step, it)
    return items, total, {"cap65536_fallbacks": fb}


def gvm_demote(item, recorder, budget, ep, step):
    """Downscale a stored full snapshot to a 25%-area JPEG thumbnail."""
    src = Image.open(item["path"])
    th = src.resize((src.width // 2, src.height // 2),
                    Image.Resampling.LANCZOS)
    q, payload = budget_jpeg(th, max(1, item["bytes"] // 2))
    tpath = Path(item["path"]).with_suffix(".thumb.jpg")
    effective = len(payload) < item["bytes"]
    old_bytes = item["bytes"]
    if effective:
        tpath.write_bytes(payload)
        item["path"] = str(tpath)
        item["bytes"] = len(payload)
        item["meta"] += (f"; demoted at step {step + 1} to a 25%-area JPEG "
                         f"thumbnail (q{q}, {len(payload)} bytes)")
    item["thumb"] = True                # tried, never retried
    recorder.ev("GVM", budget, ep, step, "demote", item_id=item["item_id"],
                bytes_before=old_bytes, bytes_after=item["bytes"],
                effective=effective, page=item["page"])
    return old_bytes - item["bytes"]


def replay_gvm(rec_ep, budget, cache, cache64, recorder, outdir, det_cache):
    ep = rec_ep["episode_id"]
    items, total = [], 0
    ref_png = {}                        # page -> PNG of last-stored state
    input_gate = {"fit8192": 0, "fallback24576": 0, "fallback65536": 0}
    det = {"revisits": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0}
    residual = {"events": 0, "max_overage": 0}

    for obs in rec_ep["observations"]:
        step, page = obs["step"], obs["page"]
        if page not in ref_png:                       # ---- new page: full
            enc, cap = cache[(obs["png"], GVM_TIGHT_CAP)], GVM_TIGHT_CAP
            if enc["feasible"]:
                input_gate["fit8192"] += 1
            else:
                enc, cap = cache[(obs["png"], GVM_FALLBACK_CAP)], \
                    GVM_FALLBACK_CAP
                if enc["feasible"]:
                    input_gate["fallback24576"] += 1
                else:
                    enc, cap = cache64[(obs["png"], KEEP_FALLBACK_CAP)], \
                        KEEP_FALLBACK_CAP
                    input_gate["fallback65536"] += 1
                    assert enc["feasible"]
            path = outdir / ep / f"s{step:02d}_{page}.avif"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(enc["payload"])
            it = _item("GVM", budget, ep, obs, "full", str(path),
                       enc["bytes"],
                       full_meta(obs, enc["quality"], enc["bytes"], cap))
            items.append(it)
            total += it["bytes"]
            recorder.store("GVM", budget, ep, step, it)
            ref_png[page] = obs["png"]
        else:                                         # ---- revisit: detect
            det["revisits"] += 1
            dkey = (ep, step)
            if dkey not in det_cache:
                det_cache[dkey] = detect_change(ref_png[page], obs["png"])
            d = det_cache[dkey]
            fired = d["score"] > DETECT_THRESHOLD
            truly_changed = obs["mutation"] is not None
            det["tp" if fired and truly_changed else
                "fn" if not fired and truly_changed else
                "fp" if fired else "tn"] += 1
            if not fired:
                recorder.ev("GVM", budget, ep, step, "dedup", page=page,
                            score=round(d["score"], 2),
                            truly_changed=truly_changed)
            else:
                if truly_changed:
                    bx1, by1, bx2, by2 = obs["mutation"]["bbox"]
                    y0 = max(HEADER_H, by1 - PATCH_MARGIN)
                    y1 = min(H, by2 + PATCH_MARGIN)
                    region_src = "true_bbox"
                else:                  # false alarm: detector's top tile row
                    ty = d["top_tile"][1]
                    y0 = max(HEADER_H, ty - PATCH_MARGIN)
                    y1 = min(H, ty + probe.TILE + PATCH_MARGIN)
                    region_src = "detector_top_tile"
                crop = Image.open(obs["png"]).crop((ROW_X[0], y0,
                                                    ROW_X[1], y1))
                buf = BytesIO()
                crop.convert("RGB").save(buf, format="JPEG",
                                         quality=PATCH_JPEG_Q, optimize=True)
                payload = buf.getvalue()
                base = max((i for i in items
                            if i["page"] == page and i["kind"] == "full"),
                           key=lambda i: i["step"])
                path = outdir / ep / f"s{step:02d}_{page}_patch.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                meta = (f"captured at step {step + 1} of 12, page {page} "
                        f"('{TITLES[page]}'), revision {obs['revision_id']}: "
                        f"UPDATE PATCH — crop of the changed field row "
                        f"(y={y0}..{y1} of the step-{step + 1} screenshot, "
                        f"x={ROW_X[0]}..{ROW_X[1]}); applies on top of the "
                        f"snapshot from step {base['step'] + 1} (JPEG "
                        f"q{PATCH_JPEG_Q}, {len(payload)} bytes, clock "
                        f"{obs['clock']})")
                it = _item("GVM", budget, ep, obs, "patch", str(path),
                           len(payload), meta, base_step=base["step"],
                           region=[ROW_X[0], y0, ROW_X[1], y1],
                           region_src=region_src, detect_dy=d["dy"],
                           detect_score=round(d["score"], 2))
                items.append(it)
                total += it["bytes"]
                recorder.store("GVM", budget, ep, step, it)
                ref_png[page] = obs["png"]

        # ---- running cap enforcement --------------------------------
        cap = budget * (step + 1)
        while total > cap:
            live = [i for i in items if i["live"] and i["step"] < step]
            by_page = {}
            for i in live:
                by_page.setdefault(i["page"], []).append(i)
            latest_patch = {p: max((i["step"] for i in v
                                    if i["kind"] == "patch"), default=None)
                            for p, v in by_page.items()}
            # 1) superseded patches, oldest first
            cand = sorted((i for i in live if i["kind"] == "patch"
                           and i["step"] != latest_patch[i["page"]]),
                          key=lambda i: i["step"])
            if cand:
                v = cand[0]
                v["live"] = False
                total -= v["bytes"]
                recorder.delete("GVM", budget, ep, step, v)
                continue
            # 2) fulls of multi-item pages, then 3) single-item pages
            for multi in (True, False):
                cand = sorted((i for i in live if i["kind"] == "full"
                               and not i["thumb"]
                               and (len(by_page[i["page"]]) >= 2) == multi),
                              key=lambda i: i["step"])
                if cand:
                    break
            if cand:
                total -= gvm_demote(cand[0], recorder, budget, ep, step)
                continue
            residual["events"] += 1
            residual["max_overage"] = max(residual["max_overage"],
                                          total - cap)
            recorder.ev("GVM", budget, ep, step, "over_cap_residual",
                        total_bytes=total, cap=cap, overage=total - cap)
            break
    return items, total, {"input_gate": input_gate, "detection": det,
                          "over_cap_residual": residual}


def replay_text(rec_ep, recorder):
    ep = rec_ep["episode_id"]
    items, total = [], 0
    for obs in rec_ep["observations"]:
        page = obs["page"]
        lines = [f"page title: {TITLES[page]}"]
        lines += [f"{LABELS[page][fid]}: {v}"
                  for fid, v in obs["visible_values"].items()]
        text = "\n".join(lines)
        nbytes = len(text.encode("utf-8"))
        meta = (f"perfect-OCR text dump captured at step {obs['step'] + 1} "
                f"of 12, page {page} ('{TITLES[page]}'), revision "
                f"{obs['revision_id']}, scroll {obs['scroll']}px, clock "
                f"{obs['clock']} ({nbytes} bytes)")
        it = _item("text_only", "-", ep, obs, "text", None, nbytes, meta,
                   text=text)
        items.append(it)
        total += nbytes
        recorder.store("text_only", "-", ep, obs["step"], it)
    return items, total, {}


# ---------------------------------------------------------------------------

def memory_items_for(items, page):
    out = []
    for it in sorted((i for i in items if i["live"] and i["page"] == page),
                     key=lambda i: (i["step"], i["kind"])):
        if it["kind"] == "text":
            out.append({"kind": "text", "text": it["text"], "meta": it["meta"]})
        else:
            out.append({"kind": "image", "path": it["path"], "meta": it["meta"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    episodes = [json.loads(l) for l in open(MANIFEST)]
    pngs = [o["png"] for e in episodes for o in e["observations"]]
    print(f"{len(episodes)} episodes, {len(pngs)} observations")

    # ---- parallel AVIF budget-encode precompute -------------------------
    tasks = [(p, b) for p in pngs for b in (8192, 24576)]
    cache = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for png, tgt, res in ex.map(_encode_task, tasks, chunksize=4):
            cache[(png, tgt)] = res
    need64 = [p for p in pngs if not cache[(p, 24576)]["feasible"]]
    cache64 = {}
    if need64:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for png, tgt, res in ex.map(
                    _encode_task, [(p, KEEP_FALLBACK_CAP) for p in need64]):
                cache64[(png, tgt)] = res
    n_inf8 = sum(not cache[(p, 8192)]["feasible"] for p in pngs)
    n_inf24 = sum(not cache[(p, 24576)]["feasible"] for p in pngs)
    print(f"encode cache ready: infeasible@8192={n_inf8}/{len(pngs)}, "
          f"infeasible@24576={n_inf24}/{len(pngs)} (-> 65536 fallback)")

    if STORE.exists():
        shutil.rmtree(STORE)
    recorder = Recorder()
    det_cache = {}
    ledger = {p: {} for p in POLICIES}
    stores = {}          # (policy, budget) -> {ep: items}
    extras = {}

    # budget-independent policies replay once
    keep_items, keep_extra = {}, {}
    text_items = {}
    for e in episodes:
        ep = e["episode_id"]
        it, tot, ex_ = replay_keep(e, cache | cache64, recorder,
                                   STORE / "keep_everything")
        keep_items[ep] = (it, tot)
        for k, v in ex_.items():
            keep_extra[k] = keep_extra.get(k, 0) + v
        it, tot, _ = replay_text(e, recorder)
        text_items[ep] = (it, tot)

    for budget in BUDGETS:
        for e in episodes:
            ep = e["episode_id"]
            it, tot, ex_ = replay_uniform(
                e, budget, cache, recorder, STORE / f"uniform_b{budget}")
            stores[("uniform", budget, ep)] = (it, tot)
            k = ("uniform", budget)
            extras.setdefault(k, {"store_failures": 0})
            extras[k]["store_failures"] += ex_["store_failures"]

            it, tot, ex_ = replay_gvm(
                e, budget, cache, cache64, recorder,
                STORE / f"gvm_b{budget}", det_cache)
            stores[("GVM", budget, ep)] = (it, tot)
            k = ("GVM", budget)
            if k not in extras:
                extras[k] = ex_
            else:
                for grp in ("input_gate", "detection"):
                    for kk in ex_[grp]:
                        extras[k][grp][kk] += ex_[grp][kk]
                extras[k]["over_cap_residual"]["events"] += \
                    ex_["over_cap_residual"]["events"]
                extras[k]["over_cap_residual"]["max_overage"] = max(
                    extras[k]["over_cap_residual"]["max_overage"],
                    ex_["over_cap_residual"]["max_overage"])

    # ---- ledger ---------------------------------------------------------
    def ev_count(policy, budget, etype):
        return sum(1 for v in recorder.events
                   if v["policy"] == policy and v["type"] == etype
                   and (v["budget"] == budget or v["budget"] == "-"))

    for policy in POLICIES:
        for budget in BUDGETS:
            if policy == "no_memory":
                per_ep = {e["episode_id"]: 0 for e in episodes}
            elif policy == "keep_everything":
                per_ep = {ep: t for ep, (i, t) in keep_items.items()}
            elif policy == "text_only":
                per_ep = {ep: t for ep, (i, t) in text_items.items()}
            else:
                per_ep = {e["episode_id"]:
                          stores[(policy, budget, e["episode_id"])][1]
                          for e in episodes}
            entry = {"total_bytes": sum(per_ep.values()),
                     "per_episode_bytes": per_ep,
                     "store_events": ev_count(policy, budget, "store"),
                     "demote_events": ev_count(policy, budget, "demote"),
                     "delete_events": ev_count(policy, budget, "delete"),
                     "dedup_events": ev_count(policy, budget, "dedup"),
                     "store_failure_events":
                         ev_count(policy, budget, "store_failure"),
                     "over_cap_residual_events":
                         ev_count(policy, budget, "over_cap_residual"),
                     "thrash_count": 0}
            if (policy, budget) in extras:
                for k, v in extras[(policy, budget)].items():
                    if k not in ("over_cap_residual",):
                        entry[k] = v
                if "over_cap_residual" in extras[(policy, budget)]:
                    entry["over_cap_residual"] = \
                        extras[(policy, budget)]["over_cap_residual"]
            if policy == "keep_everything":
                entry.update(keep_extra)
            ledger[policy][str(budget)] = entry
    # thrash is tracked globally per key; report the global count on each
    for policy in POLICIES:
        for budget in BUDGETS:
            ledger[policy][str(budget)]["thrash_count_global"] = \
                recorder.thrash

    # ---- read manifest ---------------------------------------------------
    rows = []
    for e in episodes:
        ep = e["episode_id"]
        for q in e["questions"]:
            for policy in POLICIES:
                for budget in BUDGETS:
                    if policy == "no_memory":
                        mi = []
                    elif policy == "keep_everything":
                        mi = memory_items_for(keep_items[ep][0], q["page"])
                    elif policy == "text_only":
                        mi = memory_items_for(text_items[ep][0], q["page"])
                    else:
                        mi = memory_items_for(
                            stores[(policy, budget, ep)][0], q["page"])
                    rows.append({
                        "episode_id": ep, "question_id": q["question_id"],
                        "policy": policy, "budget": budget,
                        "page": q["page"], "field": q["field"],
                        "label": q["label"], "revised": q["revised"],
                        "question_text": q["question_text"],
                        "gold": q["gold"],
                        "stale_values": q["stale_values"],
                        "memory_items": mi})
    with open(READ_MANIFEST, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(EVENTS, "w") as f:
        for v in recorder.events:
            f.write(json.dumps(v) + "\n")

    # ---- sanity checks ---------------------------------------------------
    print("\n== sanity checks ==")
    for budget in BUDGETS:
        g = ledger["GVM"][str(budget)]["total_bytes"]
        k = ledger["keep_everything"][str(budget)]["total_bytes"]
        assert g < k, f"GVM {g} !< keep {k} at B={budget}"
        print(f"  (b) B={budget}: GVM {g} < keep_everything {k} "
              f"({100 * g / k:.1f}%)")
    for r in rows:
        if r["policy"] == "keep_everything":
            assert len(r["memory_items"]) >= 1, r["question_id"]
    for e in episodes:
        for q in e["questions"]:
            assert q["gold"] == q["value_history"][-1]["value"], \
                q["question_id"]
    print(f"  (c) all keep_everything rows have >=1 item; "
          f"gold == last value_history entry for all "
          f"{sum(len(e['questions']) for e in episodes)} questions")

    # (a) 3 random changed revisits: detector fired + patch crop content
    VERIFY.mkdir(parents=True, exist_ok=True)
    changed = [(e, o) for e in episodes for o in e["observations"]
               if o["mutation"] is not None]
    rnd = random.Random(SEED)
    for e, o in rnd.sample(changed, min(3, len(changed))):
        ep, step = e["episode_id"], o["step"]
        d = det_cache[(ep, step)]
        fired = d["score"] > DETECT_THRESHOLD
        m = o["mutation"]
        # the stored patch (from the 24576 store; same file content at 8192)
        src = STORE / "gvm_b24576" / ep / f"s{step:02d}_{o['page']}_patch.jpg"
        tag = f"{ep}_s{step:02d}_{o['page']}"
        if src.exists():
            shutil.copy(src, VERIFY / f"{tag}_patch.jpg")
        # aligned old-value crop from the detection reference
        prev = max((oo for oo in e["observations"]
                    if oo["page"] == o["page"] and oo["step"] < step),
                   key=lambda oo: oo["step"])
        bx1, by1, bx2, by2 = m["bbox"]
        y0 = max(HEADER_H, by1 - PATCH_MARGIN) + d["dy"]
        y1 = min(H, by2 + PATCH_MARGIN) + d["dy"]
        Image.open(prev["png"]).crop((ROW_X[0], max(0, y0), ROW_X[1],
                                      min(H, y1))).save(
            VERIFY / f"{tag}_oldref.png")
        print(f"  (a) {tag}: fired={fired} score={d['score']:.1f} "
              f"field={m['field']} {m['old_value']} -> {m['new_value']} "
              f"patch={'yes' if src.exists() else 'MISSING'}")

    det = extras[("GVM", BUDGETS[0])]["detection"]
    print(f"\ndetector vs ground truth (per budget replay): {det}")
    LEDGER.write_text(json.dumps(ledger, indent=1))
    print(f"\nwrote {READ_MANIFEST} ({len(rows)} rows)")
    print(f"wrote {LEDGER}")
    print(f"wrote {EVENTS} ({len(recorder.events)} events)")


if __name__ == "__main__":
    main()
