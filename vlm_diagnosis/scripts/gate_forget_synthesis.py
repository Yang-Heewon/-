"""Step 5 — forget-gate synthesis on the scaled memory-dynamics run.

Combines:
  * measured trial accuracy per temporal memory condition
    (old_only / current_only / old+current / no_memory), 64 episodes
  * the Step-3 pixel change detector's per-episode verdicts

and reports the accuracy of the GATED policy: if the detector says the
screen changed, read the current snapshot; otherwise keep the old memory.
Because every condition was measured directly, the gated score is an exact
composition of measured trials, not an extrapolation.

Also reports the stale-answer rate: how often old_only answers with the
OLD (superseded) value on changed episodes — the "quiet lie".
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "discovery" / "gates" / "forget_gate_synthesis.json"


def main():
    # ground truth + stale answers from the read manifest
    manifest = {}
    for line in open(ROOT / "experiments/manifests/md_scaled_read.jsonl"):
        r = json.loads(line)
        manifest[r["episode_id"]] = {
            "changed": bool(r["factorial"]["state_changed"]),
            "gap": r["factorial"]["time_gap"],
            "stale_answers": [a.lower() for a in r["question"].get("stale_answers", [])],
        }

    trials = defaultdict(dict)  # episode -> condition -> trial
    for sh in (0, 1):
        p = ROOT / f"results/discovery/md_scaled_read.shard{sh}.shard{sh}.jsonl"
        for line in open(p):
            r = json.loads(line)
            if r.get("record_type") == "trial_result" and r["arm"] == "d4":
                trials[r["episode_id"]][r["memory_condition"]] = r

    # step-3 detector verdicts for the scaled fixture: rerun the cheap tile
    # detector inline (content region) to get per-episode decisions
    import sys
    sys.path.insert(0, str(ROOT / "vlm_diagnosis" / "scripts"))
    from gate_forget_detector import tile_diff, in_box, CONTENT_X, CONTENT_Y
    from PIL import Image

    THRESH = 40.0  # between measured unchanged-max 0 and changed-min 63
    detect = {}
    for ep in trials:
        old = ROOT / "data/md_scaled" / f"{ep}_target_old.png"
        cur = ROOT / "data/md_scaled" / f"{ep}_target_current.png"
        tiles = tile_diff(Image.open(old), Image.open(cur))
        content = max(
            v for (tx, ty), v in tiles.items()
            if in_box(tx, ty, (CONTENT_X[0], CONTENT_Y[0], CONTENT_X[1], CONTENT_Y[1]))
        )
        detect[ep] = content > THRESH

    conditions = ["old_only", "current_only", "old+current", "no_memory"]
    cells = ["changed", "unchanged"]

    def acc(rows):
        return round(sum(r["current_em"] for r in rows) / len(rows), 4) if rows else None

    report = {"n_episodes": len(trials), "threshold": THRESH}
    for cell in cells:
        eps = [e for e in trials if manifest[e]["changed"] == (cell == "changed")]
        row = {"n": len(eps)}
        for c in conditions:
            row[c] = acc([trials[e][c] for e in eps])
        # gated: detector-changed -> current_only, else -> old_only
        gated = [trials[e]["current_only" if detect[e] else "old_only"] for e in eps]
        row["gated"] = acc(gated)
        # stale-answer rate under old_only (the quiet lie)
        if cell == "changed":
            stale = sum(
                1 for e in eps
                if trials[e]["old_only"]["prediction"].strip().lower()
                in manifest[e]["stale_answers"]
            )
            row["old_only_stale_answer_rate"] = f"{stale}/{len(eps)}"
        report[cell] = row

    # overall
    all_eps = list(trials)
    overall = {}
    for c in conditions:
        overall[c] = acc([trials[e][c] for e in all_eps])
    overall["gated"] = acc(
        [trials[e]["current_only" if detect[e] else "old_only"] for e in all_eps]
    )
    detector_correct = sum(detect[e] == manifest[e]["changed"] for e in all_eps)
    overall["detector_accuracy"] = f"{detector_correct}/{len(all_eps)}"
    report["overall"] = overall

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
