#!/usr/bin/env python3
"""GVM-v2 memory-policy simulator for the preregistered E2E stream benchmark v2.

Implements docs/E2E-STREAM-PREREG-V2.md.  Only the GVM policy is re-simulated
(policy id "GVM_v2") at per-observation budgets B in {8192, 24576}; the
fixture (results/discovery/gates/stream/manifest.jsonl + PNGs) and the other
four v1 policies (no_memory / keep_everything / uniform / text_only) are NOT
re-simulated -- their v1 outputs stay canonical.

Changes vs v1 GVM (sim_stream_policies.py), per the prereg:

  (a) Demotion operator: the v1 25%-area JPEG thumbnail is replaced by a
      content-adaptive 3-rung ladder per stored full snapshot:
        R0  the stored full AVIF exactly as written (v1 write rules unchanged:
            new page -> full AVIF with 8192 -> 24576 -> 65536 feasibility
            fallback; revisit unchanged -> dedup skip; revisit changed ->
            JPEG q85 patch of the changed field row, detector reference PNG
            updated).
        R1  the SAME full-resolution source screenshot re-encoded at minimum
            AVIF quality (q=1, ~13.6KB floor) PLUS an attached text dump
            ("label: value" lines for the page as of the latest STORED
            revision, patches folded in; perfect-OCR assumption, identical to
            the text_only policy's assumption; text bytes are counted).
        R2  image bytes deleted (files unlinked), text dump only (~1KB).
            R2 items are never deleted.  When a base demotes to R2 its
            patches' image bytes are dropped too; their values are already in
            the text dump (it tracks the latest stored revision).
  (b) Eviction order: FIFO -> memory strength.  When over the running cap
      (B x observations-so-far), demote ONE item by ONE rung per iteration:
        (1) superseded-revision items first (a base whose page has a stored
            patch with a newer revision), (2) then largest
            age = current_step - last_used, (3) tie -> older capture step.
      Hysteresis: an item used at the current step (stored or usage-refreshed
      this step) cannot demote this step; an item demoted at step t cannot
      demote again before step t+2.
  (c) Mid-stream questions: 2 per episode (right after manifest steps 5 and
      9, 0-based), deterministically sampled with seed 20260820 from pages
      observed so far; usage events only (no VLM answer): ALL stored items of
      the queried page get last_used reset to the current step.

Interpretation notes (points the prereg left open; fixed here BEFORE running
and reported as such -- see the ledger "notes" field):
  * "steps 5 and 9" are the manifest's 0-based step values (6th and 10th of
    the 12 observations).
  * Demotable units are full snapshots; patches ride along with their base
    (image kept at R1, folded into the text dump at R2) and never demote
    independently.  "Superseded revisions' items first" therefore selects
    bases whose page has at least one stored patch (base revision < latest
    stored revision).
  * The per-step order is: write/dedup/patch -> mid-question usage refresh
    (steps 5 and 9) -> cap enforcement, so a refresh protects the page's
    items at the very step it fires.
  * "Demoted at step t cannot demote again at step t+1" is applied as: next
    demotion allowed at step >= t+2, which also forbids dropping two rungs
    within a single step ("one rung at a time").
  * last_used is reset ONLY by mid-question usage (per the operationalized
    spec); plain revisits do not reset the base's age (its capture step
    initializes last_used).
  * The text dump accumulates visible_values from every STORED observation of
    the page (base + patch stores), i.e. field values as of the latest stored
    revision for every field that was visible at a stored observation.
    Deduped or missed (false-negative) revisits do not contribute.  If a
    patch is stored after the dump exists (base at R1/R2), the dump is
    refreshed to the new latest stored revision (event text_dump_refresh).
  * On R0->R1 the R0 AVIF file is left on disk for debugging but its bytes
    are no longer counted and the read manifest never references it; on
    R1->R2 all image files of the base and its patches are unlinked.

Outputs under results/discovery/gates/stream/:
  read_manifest_v2.jsonl  400 rows (200 final questions x 2 budgets), same
                          row schema as v1 + "page_used_midstream"
  ledger_v2.json          per-budget byte/event totals, rung distribution,
                          ratio vs keep_everything (6,894,644 bytes)
  events_v2.jsonl         every store/dedup/demote_R1/demote_R2/
                          usage_refresh/text_dump_refresh/over_cap_residual
  mid_questions.jsonl     40 rows (20 episodes x 2) for H5' bookkeeping
  store_v2/               encoded payloads (AVIF fulls, R1 re-encodes, JPEG
                          patches)

CPU only; no VLM.  Sanity phase (2 episodes, scratch store) runs first, then
the full 20-episode x 2-budget replay.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
STREAM = ROOT / "results" / "discovery" / "gates" / "stream"
MANIFEST = STREAM / "manifest.jsonl"
STORE_V2 = STREAM / "store_v2"
READ_MANIFEST_V2 = STREAM / "read_manifest_v2.jsonl"
LEDGER_V2 = STREAM / "ledger_v2.json"
EVENTS_V2 = STREAM / "events_v2.jsonl"
MIDQ_PATH = STREAM / "mid_questions.jsonl"

sys.path.insert(0, str(ROOT))
import vlm_diagnosis.core.byte_codecs  # noqa: F401,E402  (registers AVIF)

_s1 = importlib.util.spec_from_file_location(
    "sim_stream_policies_v1",
    ROOT / "vlm_diagnosis" / "scripts" / "sim_stream_policies.py")
sim1 = importlib.util.module_from_spec(_s1)
sys.modules["sim_stream_policies_v1"] = _s1.name and sim1
_s1.loader.exec_module(sim1)

# reused v1 machinery / constants
budget_avif = sim1.budget_avif
detect_change = sim1.detect_change
full_meta = sim1.full_meta
probe = sim1.probe
TITLES, LABELS = sim1.TITLES, sim1.LABELS
W, H, HEADER_H = sim1.W, sim1.H, sim1.HEADER_H
ROW_X, PATCH_MARGIN = sim1.ROW_X, sim1.PATCH_MARGIN
DETECT_THRESHOLD = sim1.DETECT_THRESHOLD
PATCH_JPEG_Q = sim1.PATCH_JPEG_Q
GVM_TIGHT_CAP = sim1.GVM_TIGHT_CAP          # 8192
GVM_FALLBACK_CAP = sim1.GVM_FALLBACK_CAP    # 24576
KEEP_FALLBACK_CAP = sim1.KEEP_FALLBACK_CAP  # 65536

SEED = sim1.SEED                            # 20260820
BUDGETS = sim1.BUDGETS                      # (8192, 24576)
POLICY = "GVM_v2"
MID_STEPS = (5, 9)                          # manifest 0-based step values
KEEP_TOTAL_BYTES = 6894644                  # v1 ledger keep_everything total
MIN_AVIF_Q = 1

NOTES = [
    "mid-question steps 5 and 9 are the manifest's 0-based step values "
    "(6th and 10th observation)",
    "demotable units are full snapshots; patches ride with their base "
    "(image kept at R1, folded into the text dump at R2), never demote "
    "independently; 'superseded first' = base with >=1 stored patch",
    "per-step order: write -> mid-question usage refresh -> cap "
    "enforcement (refresh protects the page's items that same step)",
    "hysteresis: an item demoted at step t may demote next at step >= t+2 "
    "(also enforces one rung per step per item)",
    "last_used is reset only by mid-question usage; revisits do not reset it",
    "text dump = visible_values accumulated over STORED observations of the "
    "page (values as of the latest stored revision); refreshed when a patch "
    "lands after the dump exists",
    "on R0->R1 the superseded R0 file stays on disk uncounted and "
    "unreferenced; on R1->R2 all base+patch image files are unlinked",
]


# ---------------------------------------------------------------------------
# parallel encode tasks (top-level for pickling)
# ---------------------------------------------------------------------------

def _enc_task(args):
    png, target = args
    return png, target, budget_avif(Image.open(png), target)


def _q1_task(png):
    buf = BytesIO()
    Image.open(png).convert("RGB").save(buf, format="AVIF", quality=MIN_AVIF_Q)
    return png, buf.getvalue()


# ---------------------------------------------------------------------------
# mid-stream questions (fixture-level, policy/budget independent)
# ---------------------------------------------------------------------------

def mid_questions_for(episode):
    ep = episode["episode_id"]
    out = []
    for s in MID_STEPS:
        rng = random.Random(f"{SEED}|{ep}|mid{s}")
        seen = [o for o in episode["observations"] if o["step"] <= s]
        page = rng.choice(sorted({o["page"] for o in seen}))
        fields = sorted({f for o in seen if o["page"] == page
                         for f in o["visible_values"]})
        field = rng.choice(fields)
        gold = None
        for o in seen:                        # latest value AT THAT TIME
            if o["page"] == page and field in o["visible_values"]:
                gold = o["visible_values"][field]
        out.append({"episode": ep, "step": s, "page": page, "field": field,
                    "label": LABELS[page][field], "gold": gold})
    return out


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

class Recorder:
    def __init__(self):
        self.events = []

    def ev(self, budget, ep, step, etype, **kw):
        self.events.append({"policy": POLICY, "budget": budget, "episode": ep,
                            "step": step, "type": etype, **kw})


def _dump_text(page, values):
    lines = [f"page title: {TITLES[page]}"]
    lines += [f"{LABELS[page][fid]}: {v}" for fid, v in values.items()]
    return "\n".join(lines)


def _text_meta(base, page, latest_rev, dstep, nbytes):
    return (f"OCR index of the page {page} ('{TITLES[page]}') snapshot "
            f"captured at step {base['step'] + 1} of 12: perfect-OCR text "
            f"dump of field values as of the latest stored revision "
            f"{latest_rev} (patches folded in), attached at rung-R1 demotion "
            f"at step {dstep + 1} ({nbytes} bytes)")


def _mk_item(budget, ep, obs, kind, path, nbytes, meta, **extra):
    it = sim1._item(POLICY, budget, ep, obs, kind, path, nbytes, meta, **extra)
    it.update({"rung": 0, "last_used": obs["step"], "last_demote_step": None,
               "transitions": 0, "folded": False, "src_png": obs["png"],
               "text": None, "text_bytes": 0, "text_meta": None})
    return it


def demote_r1(st, q1cache, recorder, budget, ep, step):
    """R0 -> R1: min-quality full-res re-encode + text dump.  Returns the
    signed byte delta added to the running total."""
    b = st["base"]
    payload = q1cache[b["src_png"]]
    p0 = Path(b["path"])
    p1 = p0.with_name(p0.stem + ".r1q1.avif")
    p1.write_bytes(payload)
    old_img = b["bytes"]
    b["r0_path"] = b["path"]
    b["path"] = str(p1)
    b["bytes"] = len(payload)
    b["text"] = _dump_text(b["page"], st["values"])
    b["text_bytes"] = len(b["text"].encode("utf-8"))
    b["text_meta"] = _text_meta(b, b["page"], st["latest_rev"], step,
                                b["text_bytes"])
    b["meta"] += (f"; demoted R0->R1 at step {step + 1}: same "
                  f"full-resolution screenshot re-encoded at minimum AVIF "
                  f"quality (q{MIN_AVIF_Q}, {len(payload)} bytes), paired "
                  f"with an OCR text index ({b['text_bytes']} bytes); "
                  f"now rung R1")
    b["rung"], b["last_demote_step"] = 1, step
    b["transitions"] += 1
    recorder.ev(budget, ep, step, "demote_R1", item_id=b["item_id"],
                page=b["page"], img_bytes_before=old_img,
                img_bytes_after=b["bytes"], text_bytes=b["text_bytes"],
                transitions=b["transitions"])
    return (b["bytes"] - old_img) + b["text_bytes"]


def demote_r2(st, recorder, budget, ep, step):
    """R1 -> R2: unlink base + patch image files, keep the text dump only.
    Returns the byte count freed from the running total."""
    b = st["base"]
    freed = b["bytes"]
    for p in (b["path"], b.get("r0_path")):
        if p and Path(p).exists():
            Path(p).unlink()
    n_folded = 0
    for pt in st["patches"]:
        if not pt["folded"]:
            freed += pt["bytes"]
            if Path(pt["path"]).exists():
                Path(pt["path"]).unlink()
            pt["folded"] = True
            pt["meta"] += (f"; folded into the base text dump at step "
                           f"{step + 1} (base demoted to R2)")
            n_folded += 1
    b["bytes"] = 0
    b["path"] = None
    b["meta"] += (f"; demoted R1->R2 at step {step + 1}: image bytes "
                  f"deleted, text dump only; now rung R2")
    b["text_meta"] += (f"; base demoted R1->R2 at step {step + 1}: this "
                       f"text dump is now the page's only stored item")
    b["rung"], b["last_demote_step"] = 2, step
    b["transitions"] += 1
    recorder.ev(budget, ep, step, "demote_R2", item_id=b["item_id"],
                page=b["page"], img_bytes_freed=freed,
                text_bytes=b["text_bytes"], patches_folded=n_folded,
                transitions=b["transitions"])
    return freed


def replay_gvm_v2(rec_ep, budget, cache, cache64, q1cache, recorder, outdir,
                  det_cache, mids):
    ep = rec_ep["episode_id"]
    pages = {}          # page -> {base, patches, ref, values, latest_rev,
                        #          stored_steps}
    total = 0
    input_gate = {"fit8192": 0, "fallback24576": 0, "fallback65536": 0}
    det = {"revisits": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0}
    residual = {"events": 0, "max_overage": 0}
    mid_by_step = {m["step"]: m for m in mids}

    for obs in rec_ep["observations"]:
        step, page = obs["step"], obs["page"]

        # ---- write rules (identical to v1 GVM) --------------------------
        if page not in pages:                               # new page: full
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
            meta = full_meta(obs, enc["quality"], enc["bytes"], cap) + \
                "; rung R0"
            it = _mk_item(budget, ep, obs, "full", str(path), enc["bytes"],
                          meta)
            pages[page] = {"base": it, "patches": [], "ref": obs["png"],
                           "values": dict(obs["visible_values"]),
                           "latest_rev": obs["revision_id"],
                           "stored_steps": [step]}
            total += it["bytes"]
            recorder.ev(budget, ep, step, "store", kind="full",
                        item_id=it["item_id"], bytes=it["bytes"], page=page)
        else:                                               # revisit: detect
            st = pages[page]
            det["revisits"] += 1
            dkey = (ep, step)
            if dkey not in det_cache:
                det_cache[dkey] = detect_change(st["ref"], obs["png"])
            d = det_cache[dkey]
            fired = d["score"] > DETECT_THRESHOLD
            truly_changed = obs["mutation"] is not None
            det["tp" if fired and truly_changed else
                "fn" if not fired and truly_changed else
                "fp" if fired else "tn"] += 1
            if not fired:
                recorder.ev(budget, ep, step, "dedup", page=page,
                            score=round(d["score"], 2),
                            truly_changed=truly_changed)
            else:
                if truly_changed:
                    bx1, by1, bx2, by2 = obs["mutation"]["bbox"]
                    y0 = max(HEADER_H, by1 - PATCH_MARGIN)
                    y1 = min(H, by2 + PATCH_MARGIN)
                    region_src = "true_bbox"
                else:
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
                base = st["base"]
                path = outdir / ep / f"s{step:02d}_{page}_patch.jpg"
                path.write_bytes(payload)
                meta = (f"captured at step {step + 1} of 12, page {page} "
                        f"('{TITLES[page]}'), revision {obs['revision_id']}: "
                        f"UPDATE PATCH — crop of the changed field row "
                        f"(y={y0}..{y1} of the step-{step + 1} screenshot, "
                        f"x={ROW_X[0]}..{ROW_X[1]}); applies on top of the "
                        f"snapshot from step {base['step'] + 1} (JPEG "
                        f"q{PATCH_JPEG_Q}, {len(payload)} bytes, clock "
                        f"{obs['clock']})")
                it = _mk_item(budget, ep, obs, "patch", str(path),
                              len(payload), meta, base_step=base["step"],
                              region=[ROW_X[0], y0, ROW_X[1], y1],
                              region_src=region_src, detect_dy=d["dy"],
                              detect_score=round(d["score"], 2))
                st["patches"].append(it)
                st["values"].update(obs["visible_values"])
                st["latest_rev"] = obs["revision_id"]
                st["ref"] = obs["png"]
                st["stored_steps"].append(step)
                total += it["bytes"]
                recorder.ev(budget, ep, step, "store", kind="patch",
                            item_id=it["item_id"], bytes=it["bytes"],
                            page=page)
                if base["rung"] >= 1:       # keep the dump at latest stored
                    old = base["text_bytes"]
                    base["text"] = _dump_text(page, st["values"])
                    base["text_bytes"] = len(base["text"].encode("utf-8"))
                    base["text_meta"] = _text_meta(
                        base, page, st["latest_rev"], step,
                        base["text_bytes"]) + \
                        (f"; refreshed at step {step + 1} after a new patch"
                         if base["rung"] == 1 else
                         f"; refreshed at step {step + 1} after a new patch "
                         f"(base at R2)")
                    total += base["text_bytes"] - old
                    recorder.ev(budget, ep, step, "text_dump_refresh",
                                item_id=base["item_id"], page=page,
                                text_bytes_before=old,
                                text_bytes_after=base["text_bytes"])

        # ---- mid-stream question: usage refresh -------------------------
        if step in mid_by_step:
            m = mid_by_step[step]
            st = pages[m["page"]]           # page observed by now, so stored
            n = 0
            for it in [st["base"]] + st["patches"]:
                it["last_used"] = step
                n += 1
            recorder.ev(budget, ep, step, "usage_refresh", page=m["page"],
                        field=m["field"], items_refreshed=n)

        # ---- cap enforcement: memory-strength demotion ------------------
        cap = budget * (step + 1)
        while total > cap:
            cands = []
            for st in pages.values():
                b = st["base"]
                if b["rung"] >= 2:
                    continue
                if b["last_used"] == step:                  # used this step
                    continue
                if b["last_demote_step"] is not None and \
                        step - b["last_demote_step"] < 2:   # hysteresis
                    continue
                superseded = st["latest_rev"] > b["revision_id"]
                cands.append(((0 if superseded else 1,
                               -(step - b["last_used"]), b["step"]), st))
            if not cands:
                residual["events"] += 1
                residual["max_overage"] = max(residual["max_overage"],
                                              total - cap)
                recorder.ev(budget, ep, step, "over_cap_residual",
                            total_bytes=total, cap=cap, overage=total - cap)
                break
            _, st = min(cands, key=lambda t: t[0])
            if st["base"]["rung"] == 0:
                total += demote_r1(st, q1cache, recorder, budget, ep, step)
            else:
                total -= demote_r2(st, recorder, budget, ep, step)

    # ---- invariant: running total == recomputed total -------------------
    calc = 0
    for st in pages.values():
        b = st["base"]
        calc += b["bytes"] + b["text_bytes"]
        calc += sum(p["bytes"] for p in st["patches"] if not p["folded"])
    assert calc == total, f"{ep} b{budget}: running {total} != calc {calc}"
    return pages, total, {"input_gate": input_gate, "detection": det,
                          "over_cap_residual": residual}


# ---------------------------------------------------------------------------
# read manifest
# ---------------------------------------------------------------------------

def memory_items_v2(pages, page):
    """Images in capture order, then the text dump last (nearest to the
    question).  R0: image only.  R1: low-q image + text dump.  R2: text
    only.  Folded patches are represented by the dump."""
    st = pages[page]
    b = st["base"]
    out = []
    if b["rung"] < 2:
        out.append({"kind": "image", "path": b["path"], "meta": b["meta"]})
    for p in sorted(st["patches"], key=lambda i: i["step"]):
        if not p["folded"]:
            out.append({"kind": "image", "path": p["path"], "meta": p["meta"]})
    if b["rung"] >= 1:
        out.append({"kind": "text", "text": b["text"], "meta": b["text_meta"]})
    return out


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def run_phase(episodes, store_root, cache, cache64, q1cache, det_cache,
              midqs):
    recorder = Recorder()
    stores, extras = {}, {}
    if store_root.exists():
        shutil.rmtree(store_root)
    for budget in BUDGETS:
        for e in episodes:
            ep = e["episode_id"]
            pages, total, ex = replay_gvm_v2(
                e, budget, cache, cache64, q1cache, recorder,
                store_root / f"gvm_v2_b{budget}", det_cache, midqs[ep])
            stores[(budget, ep)] = (pages, total)
            k = budget
            if k not in extras:
                extras[k] = ex
            else:
                for grp in ("input_gate", "detection"):
                    for kk in ex[grp]:
                        extras[k][grp][kk] += ex[grp][kk]
                extras[k]["over_cap_residual"]["events"] += \
                    ex["over_cap_residual"]["events"]
                extras[k]["over_cap_residual"]["max_overage"] = max(
                    extras[k]["over_cap_residual"]["max_overage"],
                    ex["over_cap_residual"]["max_overage"])
    return recorder, stores, extras


def rung_distribution(stores, episodes, budget):
    dist = {"R0": 0, "R1": 0, "R2": 0}
    patches = {"live": 0, "folded": 0}
    for e in episodes:
        pages, _ = stores[(budget, e["episode_id"])]
        for st in pages.values():
            dist[f"R{st['base']['rung']}"] += 1
            for p in st["patches"]:
                patches["folded" if p["folded"] else "live"] += 1
    return dist, patches


def thrash_stats(stores, episodes, budget):
    """Prereg thrash metric: any single item with >= 3 rung transitions."""
    max_tr, n_ge3 = 0, 0
    for e in episodes:
        pages, _ = stores[(budget, e["episode_id"])]
        for st in pages.values():
            tr = st["base"]["transitions"]
            max_tr = max(max_tr, tr)
            if tr >= 3:
                n_ge3 += 1
    return {"items_with_ge3_transitions": n_ge3,
            "max_transitions_per_item": max_tr}


def expected_dump_values(episode, page, stored_steps):
    vals = {}
    for o in episode["observations"]:
        if o["page"] == page and o["step"] in stored_steps:
            vals.update(o["visible_values"])
    return vals


def sanity_r1_inspection(episodes, stores):
    """Find a base at rung R1 at stream end; verify file, resolution, and
    text dump vs the fixture manifest."""
    for budget in BUDGETS:
        for e in episodes:
            pages, _ = stores[(budget, e["episode_id"])]
            for page, st in sorted(pages.items()):
                b = st["base"]
                if b["rung"] != 1:
                    continue
                p = Path(b["path"])
                im = Image.open(p)
                exp = expected_dump_values(e, page, st["stored_steps"])
                got = {}
                lab2fid = {v: k for k, v in LABELS[page].items()}
                for ln in b["text"].splitlines()[1:]:
                    lab, val = ln.split(": ", 1)
                    got[lab2fid[lab]] = val
                print(f"  (a) R1 item: ep={e['episode_id']} page={page} "
                      f"budget={budget}")
                print(f"      file exists={p.exists()} "
                      f"bytes={p.stat().st_size} format={im.format} "
                      f"size={im.size} (full res = {(W, H)})")
                print(f"      stored steps={st['stored_steps']} latest "
                      f"stored revision={st['latest_rev']}")
                match = exp == got
                print(f"      dump fields={len(got)} vs manifest expectation"
                      f"={len(exp)} -> values match: {match}")
                for fid in sorted(set(exp) | set(got)):
                    tag = "OK " if exp.get(fid) == got.get(fid) else "DIFF"
                    print(f"        [{tag}] {LABELS[page][fid]}: dump="
                          f"{got.get(fid)!r} manifest={exp.get(fid)!r}")
                assert p.exists() and im.size == (W, H) and \
                    im.format == "AVIF" and match
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sanity-store", default=None,
                    help="store root for the 2-episode sanity phase")
    args = ap.parse_args()

    episodes = [json.loads(l) for l in open(MANIFEST)]
    base_pngs = []
    for e in episodes:
        seen = set()
        for o in e["observations"]:
            if o["page"] not in seen:
                seen.add(o["page"])
                base_pngs.append(o["png"])
    print(f"{len(episodes)} episodes, "
          f"{sum(len(e['observations']) for e in episodes)} observations, "
          f"{len(base_pngs)} first-occurrence pages (bases)")

    # ---- encode caches (bases only; revisits are never full-stored) -----
    tasks = [(p, b) for p in base_pngs for b in (8192, 24576)]
    cache = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for png, tgt, res in ex.map(_enc_task, tasks, chunksize=2):
            cache[(png, tgt)] = res
    need64 = [p for p in base_pngs if not cache[(p, 24576)]["feasible"]]
    cache64 = {}
    if need64:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for png, tgt, res in ex.map(
                    _enc_task, [(p, KEEP_FALLBACK_CAP) for p in need64]):
                cache64[(png, tgt)] = res
    q1cache = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for png, payload in ex.map(_q1_task, base_pngs, chunksize=2):
            q1cache[png] = payload
    q1_sizes = sorted(len(v) for v in q1cache.values())
    print(f"encode caches ready: infeasible@8192="
          f"{sum(not cache[(p, 8192)]['feasible'] for p in base_pngs)}"
          f"/{len(base_pngs)}, infeasible@24576={len(need64)} (-> 65536); "
          f"q1 floor bytes min/median/max = {q1_sizes[0]}/"
          f"{q1_sizes[len(q1_sizes) // 2]}/{q1_sizes[-1]}")

    # ---- mid-stream questions (fixture-level) ---------------------------
    midqs = {e["episode_id"]: mid_questions_for(e) for e in episodes}

    det_cache = {}

    # ================= sanity phase: 2 episodes ==========================
    print("\n== sanity phase (2 episodes, both budgets) ==")
    sanity_root = Path(args.sanity_store) if args.sanity_store else \
        STORE_V2.with_name("store_v2_sanity")
    s_eps = episodes[:2]
    s_rec, s_stores, _ = run_phase(s_eps, sanity_root, cache, cache64,
                                   q1cache, det_cache, midqs)
    found = sanity_r1_inspection(s_eps, s_stores)
    if not found:
        print("  (a) no R1 item at stream end in 2 episodes -- widening")
        s_eps = episodes[:6]
        s_rec, s_stores, _ = run_phase(s_eps, sanity_root, cache, cache64,
                                       q1cache, det_cache, midqs)
        assert sanity_r1_inspection(s_eps, s_stores), "no R1 item found"
    # (b) R2 floor: every observed page still has >= 1 item; every final
    # question has >= 1 memory item
    for budget in BUDGETS:
        for e in s_eps:
            pages, _ = s_stores[(budget, e["episode_id"])]
            for page, st in pages.items():
                n = len(memory_items_v2(pages, page))
                assert n >= 1, (e["episode_id"], budget, page)
            for q in e["questions"]:
                assert len(memory_items_v2(pages, q["page"])) >= 1, \
                    (e["episode_id"], budget, q["question_id"])
    print(f"  (b) R2 floor holds: no page lost its last item; every final "
          f"question of the {len(s_eps)} sanity episodes has >=1 memory "
          f"item at both budgets")
    for budget in BUDGETS:
        print(f"  (c) thrash @B={budget}: "
              f"{thrash_stats(s_stores, s_eps, budget)}")

    # ================= full run: 20 episodes x 2 budgets =================
    print("\n== full run (all episodes, both budgets) ==")
    recorder, stores, extras = run_phase(episodes, STORE_V2, cache, cache64,
                                         q1cache, det_cache, midqs)

    # ---- mid_questions.jsonl -------------------------------------------
    with open(MIDQ_PATH, "w") as f:
        for e in episodes:
            for m in midqs[e["episode_id"]]:
                f.write(json.dumps(m) + "\n")

    # ---- read manifest v2 ----------------------------------------------
    mid_pages = {ep: {m["page"] for m in ms} for ep, ms in midqs.items()}
    rows = []
    for e in episodes:
        ep = e["episode_id"]
        for q in e["questions"]:
            for budget in BUDGETS:
                pages, _ = stores[(budget, ep)]
                rows.append({
                    "episode_id": ep, "question_id": q["question_id"],
                    "policy": POLICY, "budget": budget,
                    "page": q["page"], "field": q["field"],
                    "label": q["label"], "revised": q["revised"],
                    "question_text": q["question_text"],
                    "gold": q["gold"],
                    "stale_values": q["stale_values"],
                    "memory_items": memory_items_v2(pages, q["page"]),
                    "page_used_midstream": q["page"] in mid_pages[ep]})
    with open(READ_MANIFEST_V2, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(EVENTS_V2, "w") as f:
        for v in recorder.events:
            f.write(json.dumps(v) + "\n")

    # ---- ledger v2 ------------------------------------------------------
    def ev_count(budget, etype, **match):
        return sum(1 for v in recorder.events
                   if v["budget"] == budget and v["type"] == etype
                   and all(v.get(k) == vv for k, vv in match.items()))

    ledger = {POLICY: {}, "keep_everything_total_bytes": KEEP_TOTAL_BYTES,
              "notes": NOTES,
              "mid_questions": {"count": sum(len(v) for v in midqs.values()),
                                "path": str(MIDQ_PATH)}}
    for budget in BUDGETS:
        per_ep = {e["episode_id"]: stores[(budget, e["episode_id"])][1]
                  for e in episodes}
        total = sum(per_ep.values())
        dist, patches = rung_distribution(stores, episodes, budget)
        text_bytes = sum(st["base"]["text_bytes"]
                         for e in episodes
                         for st in stores[(budget, e["episode_id"])][0]
                         .values())
        n_mid_rows = sum(1 for r in rows if r["budget"] == budget
                         and r["page_used_midstream"])
        ledger[POLICY][str(budget)] = {
            "total_bytes": total,
            "text_bytes_included": text_bytes,
            "ratio_vs_keep_everything": round(total / KEEP_TOTAL_BYTES, 4),
            "per_episode_bytes": per_ep,
            "store_full_events": ev_count(budget, "store", kind="full"),
            "store_patch_events": ev_count(budget, "store", kind="patch"),
            "dedup_events": ev_count(budget, "dedup"),
            "demote_R1_events": ev_count(budget, "demote_R1"),
            "demote_R2_events": ev_count(budget, "demote_R2"),
            "usage_refresh_events": ev_count(budget, "usage_refresh"),
            "text_dump_refresh_events": ev_count(budget,
                                                 "text_dump_refresh"),
            "over_cap_residual_events": ev_count(budget,
                                                 "over_cap_residual"),
            "over_cap_residual": extras[budget]["over_cap_residual"],
            "thrash": thrash_stats(stores, episodes, budget),
            "final_rung_distribution": dist,
            "patches": patches,
            "input_gate": extras[budget]["input_gate"],
            "detection": extras[budget]["detection"],
            "final_questions_with_page_used_midstream": n_mid_rows,
        }
    LEDGER_V2.write_text(json.dumps(ledger, indent=1))

    # ---- final checks (e) ----------------------------------------------
    print("\n== final checks ==")
    assert len(rows) == 2 * sum(len(e["questions"]) for e in episodes), \
        len(rows)
    n_img = n_txt = 0
    for r in rows:
        assert len(r["memory_items"]) >= 1, r["question_id"]
        for mi in r["memory_items"]:
            if mi["kind"] == "image":
                assert Path(mi["path"]).exists(), mi["path"]
                n_img += 1
            else:
                assert mi["text"]
                n_txt += 1
    print(f"  (e) {len(rows)} rows; every row >=1 item; all {n_img} image "
          f"paths exist; {n_txt} text items")
    for budget in BUDGETS:
        L = ledger[POLICY][str(budget)]
        ratio = L["ratio_vs_keep_everything"]
        assert ratio < 0.5, (budget, ratio)
        print(f"  (e) B={budget}: total {L['total_bytes']} bytes = "
              f"{100 * ratio:.1f}% of keep_everything ({KEEP_TOTAL_BYTES}) "
              f"< 50%")
        print(f"      events: store_full={L['store_full_events']} "
              f"store_patch={L['store_patch_events']} "
              f"dedup={L['dedup_events']} demote_R1={L['demote_R1_events']} "
              f"demote_R2={L['demote_R2_events']} "
              f"usage_refresh={L['usage_refresh_events']} "
              f"dump_refresh={L['text_dump_refresh_events']} "
              f"residual={L['over_cap_residual_events']}")
        print(f"      rungs at stream end: {L['final_rung_distribution']} "
              f"patches: {L['patches']} thrash: {L['thrash']}")
        print(f"      detection: {L['detection']} (v1 parity expected)")
        print(f"      final questions with page_used_midstream=true: "
              f"{L['final_questions_with_page_used_midstream']}/"
              f"{len(rows) // 2}")
    n_pages_mid = sum(len(v) for v in mid_pages.values())
    print(f"  mid-questions: {sum(len(v) for v in midqs.values())} written "
          f"to {MIDQ_PATH.name}; {n_pages_mid} (episode, page) pairs got a "
          f"usage refresh")
    print(f"\nwrote {READ_MANIFEST_V2} ({len(rows)} rows)")
    print(f"wrote {LEDGER_V2}")
    print(f"wrote {EVENTS_V2} ({len(recorder.events)} events)")
    print(f"wrote {MIDQ_PATH}")


if __name__ == "__main__":
    main()
