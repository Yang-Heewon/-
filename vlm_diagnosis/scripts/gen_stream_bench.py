#!/usr/bin/env python3
"""Stream benchmark fixture generator (preregistered E2E stream, M7 candidate).

Implements the fixture half of docs/E2E-STREAM-PREREG.md:

  - 20 episodes x 12 observations, seed 20260820, deterministic per episode.
  - Each episode uses the 5 realgui templates (dashboard/inbox/settings/
    table_report/article) as 5 pages with episode-specific instance data;
    each page appears 1..4 times, counts sum to 12, order shuffled.
  - Every observation renders with headless Chrome: fixed 64px header with a
    monotonic episode clock + snapshot id (all observations differ in pixels),
    random vertical scroll in {0..600}px (0 included with prob 0.15).
  - On a revisit, with prob 0.5 exactly one field is mutated.  The mutated
    field must be fully visible in BOTH the current screenshot and the page's
    previous observation (otherwise the change would be undetectable by
    construction, matching gate_realgui_probe's pairing rule).  Ground-truth
    bbox in the new screenshot's coordinates comes from the magenta debug
    re-render diff (same technique as gate_realgui_probe).
  - After the stream, 2 questions per page (one about a revised field when
    the page has any revision, one about a never-revised field), gold =
    latest observed value.  20 episodes x 10 questions = 200 questions.

Outputs under results/discovery/gates/stream/:
    manifest.jsonl   one line per episode: observations (with mutations,
                     bboxes, visible-field value snapshots) + questions.
    png/, html/      rendered screenshots (+ debug renders for bboxes).

Reuses gate_realgui_probe.py (templates, page_html, render_many, calibrate,
diff_bbox) by import; no logic is duplicated.  CPU only; no VLM.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STREAM = ROOT / "results" / "discovery" / "gates" / "stream"
PNG_DIR = STREAM / "png"
HTML_DIR = STREAM / "html"
MANIFEST = STREAM / "manifest.jsonl"

SEED = 20260820
OBS_PER_EP = 12
SCROLL_MAX = 600
ZERO_SCROLL_PROB = 0.15
REVISION_PROB = 0.5
VIS_MARGIN = 12          # same visibility margin as gate_realgui_probe

# ---- import the realgui probe module (templates, rendering, calibration) ---
_spec = importlib.util.spec_from_file_location(
    "gate_realgui_probe",
    ROOT / "vlm_diagnosis" / "scripts" / "gate_realgui_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)
# route this script's HTML sources into the stream directory (probe.render
# writes html next to its own fixture otherwise)
probe.HTML_DIR = HTML_DIR

W, H, HEADER_H = probe.W, probe.H, probe.HEADER_H
TEMPLATES = probe.TEMPLATES


def visible(cal, tname, fid, scroll):
    """Field fully visible in the viewport at this scroll (probe's rule)."""
    y0, y1 = cal[tname][fid]
    return (y0 - scroll >= VIS_MARGIN
            and y1 - scroll <= H - HEADER_H - VIS_MARGIN)


def fmt_clock(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S UTC")


def gen_episode(ep: int, cal):
    """Deterministic episode plan: timeline, mutations, questions.

    Returns (record, render_jobs); bboxes are filled in after rendering.
    """
    rnd = random.Random(SEED * 1_000_003 + ep)
    ep_id = f"ep{ep:02d}"
    tnames = sorted(TEMPLATES)

    # page visit counts in 1..4 summing to 12
    counts = {t: 1 for t in tnames}
    for _ in range(OBS_PER_EP - len(tnames)):
        counts[rnd.choice([t for t in tnames if counts[t] < 4])] += 1
    slots = [t for t in tnames for _ in range(counts[t])]
    rnd.shuffle(slots)

    # episode-specific instance data
    vals = {t: {fid: rnd.choice(pool) for fid, _, pool in TEMPLATES[t][0]}
            for t in tnames}
    init_vals = {t: dict(vals[t]) for t in tnames}
    labels = {t: {fid: lbl for fid, lbl, _ in TEMPLATES[t][0]} for t in tnames}

    clock = (dt.datetime(2026, 8, 20, 8, 0, 0)
             + dt.timedelta(minutes=rnd.randint(0, 360)))

    observations = []
    jobs = []
    last_scroll = {}        # page -> scroll of its previous observation
    occurrence = {t: 0 for t in tnames}
    rev_id = {t: 0 for t in tnames}
    history = {t: {fid: [] for fid in vals[t]} for t in tnames}
    for t in tnames:
        for fid, v in vals[t].items():
            history[t][fid].append({"step": None, "value": v,
                                    "is_mutation": False})

    for step, tname in enumerate(slots):
        occurrence[tname] += 1
        occ = occurrence[tname]
        clock += dt.timedelta(minutes=rnd.randint(2, 20),
                              seconds=rnd.randint(0, 59))
        sync = clock - dt.timedelta(seconds=rnd.randint(5, 90))
        sid = f"{rnd.randrange(16**8):08x}"
        scroll = 0 if rnd.random() < ZERO_SCROLL_PROB \
            else rnd.randint(0, SCROLL_MAX)

        mutation = None
        mutation_skipped = False
        if occ > 1 and rnd.random() < REVISION_PROB:
            prev_s = last_scroll[tname]
            fields = TEMPLATES[tname][0]
            cands = [(fid, pool) for fid, _, pool in fields
                     if visible(cal, tname, fid, scroll)
                     and visible(cal, tname, fid, prev_s)]
            if cands:
                fid, pool = cands[rnd.randrange(len(cands))]
                old = vals[tname][fid]
                new = rnd.choice([v for v in pool if v != old])
                vals[tname][fid] = new
                rev_id[tname] += 1
                history[tname][fid].append(
                    {"step": step, "value": new, "is_mutation": True})
                mutation = {"field": fid, "label": labels[tname][fid],
                            "old_value": old, "new_value": new,
                            "bbox": None}
            else:
                mutation_skipped = True   # no field visible in both viewports

        png = PNG_DIR / f"{ep_id}_s{step:02d}_{tname}_r{rev_id[tname]}.png"
        jobs.append((probe.page_html(tname, vals[tname], scroll,
                                     fmt_clock(clock), fmt_clock(sync), sid),
                     png))
        if mutation is not None:
            dbg = PNG_DIR / "debug" / png.name.replace(".png", "_dbg.png")
            jobs.append((probe.page_html(tname, vals[tname], scroll,
                                         fmt_clock(clock), fmt_clock(sync),
                                         sid, dbg=mutation["field"]),
                         dbg))
            mutation["dbg_png"] = str(dbg)

        vis_fields = {fid: vals[tname][fid] for fid, _, _ in TEMPLATES[tname][0]
                      if visible(cal, tname, fid, scroll)}
        observations.append({
            "step": step, "page": tname, "occurrence": occ,
            "revision_id": rev_id[tname], "scroll": scroll,
            "clock": fmt_clock(clock), "png": str(png),
            "mutation": mutation, "mutation_skipped": mutation_skipped,
            "visible_values": vis_fields,
        })
        last_scroll[tname] = scroll

    # ---- questions: 2 per page ----------------------------------------
    questions = []
    page_steps = {t: [o["step"] for o in observations if o["page"] == t]
                  for t in tnames}
    for tname in tnames:
        title = TEMPLATES[tname][2]
        mutated = []
        for o in observations:
            m = o["mutation"]
            if o["page"] == tname and m and m["field"] not in mutated:
                mutated.append(m["field"])
        all_fids = [fid for fid, _, _ in TEMPLATES[tname][0]]
        never = [f for f in all_fids if f not in mutated]
        vis_all = [f for f in never
                   if all(visible(cal, tname, f, o["scroll"])
                          for o in observations if o["page"] == tname)]
        vis_last = [f for f in never
                    if visible(cal, tname, f,
                               observations[page_steps[tname][-1]]["scroll"])]

        picks = []
        if mutated:
            picks.append((rnd.choice(mutated), True, None))
        else:
            pool = vis_all or vis_last or never
            picks.append((rnd.choice(pool), False, "no_revision_on_page"))
        used = picks[0][0]
        if [f for f in vis_all if f != used]:
            pool2, fallback2 = [f for f in vis_all if f != used], None
        elif [f for f in vis_last if f != used]:
            pool2, fallback2 = [f for f in vis_last if f != used], \
                "visible_last_only"
        else:
            pool2, fallback2 = [f for f in never if f != used], \
                "not_visible_last"
        picks.append((rnd.choice(pool2), False, fallback2))

        for qi, (fid, revised, fb) in enumerate(picks):
            label = labels[tname][fid]
            questions.append({
                "question_id": f"{ep_id}_q_{tname}_{qi}",
                "page": tname, "field": fid, "label": label,
                "revised": revised,
                "question_text": (
                    f'You are given stored memory of the page "{title}" '
                    f'observed over time. Based on the MOST RECENT '
                    f'observation, what is the value of "{label}"? '
                    f'Answer with the value only.'),
                "gold": vals[tname][fid],
                "value_history": history[tname][fid],
                "stale_values": sorted({h["value"]
                                        for h in history[tname][fid][:-1]}),
                "fallback": fb,
            })

    rec = {"episode_id": ep_id, "seed": SEED,
           "pages": {t: {"count": counts[t], "title": TEMPLATES[t][2],
                         "init_vals": init_vals[t]} for t in tnames},
           "observations": observations, "questions": questions,
           "final_vals": {t: vals[t] for t in tnames}}
    return rec, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    for d in (STREAM, PNG_DIR, PNG_DIR / "debug", HTML_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("== calibration (reuses realgui calib renders) ==", flush=True)
    cal = probe.calibrate()

    print(f"== planning {args.episodes} episodes ==", flush=True)
    records, jobs = [], []
    for ep in range(args.episodes):
        rec, j = gen_episode(ep, cal)
        records.append(rec)
        jobs.extend(j)

    print(f"== rendering {len(jobs)} pages (skip existing) ==", flush=True)
    probe.render_many(jobs)

    # ground-truth bboxes from debug diffs (new screenshot coordinates)
    n_mut = 0
    for rec in records:
        for o in rec["observations"]:
            m = o["mutation"]
            if not m:
                continue
            bbox = probe.diff_bbox(o["png"], m["dbg_png"])
            assert bbox is not None, f"{rec['episode_id']} s{o['step']}: empty debug diff"
            x1, y1, x2, y2 = bbox
            assert y1 >= HEADER_H and y2 <= H and (y2 - y1) <= 100, \
                f"{rec['episode_id']} s{o['step']}: suspicious bbox {bbox}"
            m["bbox"] = bbox
            n_mut += 1

    with open(MANIFEST, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n_obs = sum(len(r["observations"]) for r in records)
    n_rev = sum(o["occurrence"] > 1 for r in records for o in r["observations"])
    n_skip = sum(o["mutation_skipped"] for r in records for o in r["observations"])
    n_q = sum(len(r["questions"]) for r in records)
    n_qrev = sum(q["revised"] for r in records for q in r["questions"])
    n_fb = sum(1 for r in records for q in r["questions"] if q["fallback"])
    print(f"episodes={len(records)} observations={n_obs} revisits={n_rev} "
          f"mutations={n_mut} mutation_skipped={n_skip}")
    print(f"questions={n_q} (revised={n_qrev}, unrevised={n_q - n_qrev}, "
          f"fallback-flagged={n_fb})")
    print("wrote", MANIFEST)


if __name__ == "__main__":
    main()
