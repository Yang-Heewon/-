"""재사용 판 판정 — 쓰기 시점 두 기준(이미지만 / 과거 질문)의 결합이 새 질문에서 얼마나 버티는가.

예산 B마다
  1) 표: 이미지 신호(kvzip / image) × alpha, 과거 질문만(alpha=0), random, oracle(새 질문을 아는 상한), FULL
  2) 짝지은 차이 (이미지 단위 bootstrap 95% CI):
       중간 alpha − 이미지만(alpha=1)   : 과거 질문 기준을 섞은 가치 (KVzip 대비)
       중간 alpha − 과거 질문만(alpha=0): 이미지 기준을 섞은 가치 (h2o 대비)
       중간 alpha − random
  3) 근거 겹침 층(짝 라벨 T2/T3/T4 등)별 표: 과거 질문과 새 질문이 같은 근거를 보는지에 따라 갈리는가

  python -m vlm_diagnosis.scripts.core_delta_reuse_analysis \
      --pattern "results/smoke/ru_q25_sqa.shard*.jsonl" --out results/smoke/ru_q25_sqa_summary
"""
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
COND_RE = re.compile(r"^(?P<kind>ru|ruT)_(?P<sig>kvzip|image|past)_a(?P<a>[0-9.]+)@B(?P<B>[0-9.]+)$")
CTRL_RE = re.compile(r"^(?P<name>random|oracle_s1)@B(?P<B>[0-9.]+)$")
PAIRS = {"ScreenQA": "experiments/manifests/screenqa_discovery_pairs.jsonl",
         "GQA": "experiments/manifests/gqa_discovery_pairs.jsonl"}


def load(pattern):
    rows, meta = [], []
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            (meta if r.get("record_type") == "run_metadata" else rows).append(r)
    return rows, meta


def load_pair_labels(dataset):
    path = PAIRS.get(dataset)
    if not path or not os.path.exists(os.path.join(ROOT, path)):
        return {}
    lab = {}
    for l in open(os.path.join(ROOT, path)):
        r = json.loads(l)
        lab[(str(r["sample_id"]), r["qA_id"], r["qB_id"])] = r.get("auto_label")
        lab[(str(r["sample_id"]), r["qB_id"], r["qA_id"])] = r.get("auto_label")
    return lab


def boot_ci(per_img, n_boot, seed=42):
    keys = list(per_img)
    if not keys:
        return None, None, None
    rng = random.Random(seed)

    def mean(lists):
        flat = [x for l in lists for x in l]
        return sum(flat) / len(flat) if flat else None
    pt = mean([per_img[k] for k in keys])
    vals = sorted(v for v in (mean([per_img[rng.choice(keys)] for _ in keys])
                              for _ in range(n_boot)) if v is not None)
    if not vals:
        return pt, None, None
    return pt, vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def fmt(d):
    if not d or d.get("diff") is None:
        return "—"
    lo, hi = d["ci"]
    return f"{d['diff']:+.3f}" if lo is None else f"{d['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def analyze(rows, metric, n_boot, base_correct_only, strata_filter=None):
    full, per_q, labels = {}, defaultdict(dict), {}
    dataset = rows[0]["dataset"] if rows else None
    pair_lab = load_pair_labels(dataset)
    for r in rows:
        key = (str(r["sample_id"]), r["question_id"])
        if r["condition_id"] == "FULL_KV":
            full[key] = r
            labels[key] = pair_lab.get((str(r["sample_id"]), r.get("past_question_id"), r["question_id"]))
        else:
            per_q[key][r["condition_id"]] = r
    keys = [k for k in per_q if k in full and (not base_correct_only or full[k]["em"] == 1)
            and (strata_filter is None or labels.get(k) in strata_filter)]
    conds = sorted({c for k in keys for c in per_q[k]})

    def stat(c):
        per_img = defaultdict(list)
        for k in keys:
            if c in per_q[k]:
                per_img[k[0]].append(per_q[k][c][metric])
        return boot_ci(per_img, n_boot)

    def paired(c, ref):
        if ref == c or not any(ref in per_q[k] for k in keys):
            return None
        per_img = defaultdict(list)
        for k in keys:
            if c in per_q[k] and ref in per_q[k]:
                per_img[k[0]].append(per_q[k][c][metric] - per_q[k][ref][metric])
        pt, lo, hi = boot_ci(per_img, n_boot)
        return {"diff": pt, "ci": [lo, hi]}

    table = {}
    for c in conds:
        m = COND_RE.match(c)
        mc = CTRL_RE.match(c)
        if not (m or mc):
            continue
        pt, lo, hi = stat(c)
        vis = [per_q[k][c].get("keep_frac_visual") for k in keys if c in per_q[k]
               and per_q[k][c].get("keep_frac_visual") is not None]
        t = {"mean": pt, "ci": [lo, hi], "keep_frac_visual": (sum(vis) / len(vis)) if vis else None}
        if m:
            t.update({"kind": m.group("kind"), "sig": m.group("sig"), "alpha": float(m.group("a")),
                      "B": float(m.group("B"))})
            B = m.group("B")
            if m.group("kind") == "ru" and 0 < t["alpha"] < 1:
                t["paired"] = {"vs_image_only": paired(c, f"ru_{t['sig']}_a1@B{B}"),
                               "vs_past_only": paired(c, f"ru_past_a0@B{B}"),
                               "vs_random": paired(c, f"random@B{B}"),
                               "vs_oracle": paired(c, f"oracle_s1@B{B}")}
            elif m.group("kind") == "ru":
                t["paired"] = {"vs_random": paired(c, f"random@B{B}")}
        else:
            t.update({"kind": mc.group("name"), "sig": None, "alpha": None, "B": float(mc.group("B"))})
        table[c] = t
    strata_counts = defaultdict(int)
    for k in keys:
        strata_counts[labels.get(k)] += 1
    return {"n_questions": len(keys), "n_images": len({k[0] for k in keys}),
            "full_kv_mean": (sum(full[k][metric] for k in keys) / len(keys)) if keys else None,
            "strata_counts": dict(strata_counts), "table": table}


def md(res, title):
    t = res["table"]
    L = [f"### {title} (질문 {res['n_questions']}, 이미지 {res['n_images']}, FULL {res['full_kv_mean']:.3f}; "
         f"층 분포 {res['strata_counts']})", ""]
    Bs = sorted({v["B"] for v in t.values()})
    for B in Bs:
        L += [f"#### 예산 B={B:g}% (나머지 삭제) — 새 질문 정답률 EM [95% CI]", "",
              "| 조건 | EM | 시각 비율 | − 이미지만(KVzip) | − 과거 질문만(h2o) | − random | − 상한(oracle) |",
              "|---|---|---|---|---|---|---|"]
        order = sorted([(c, v) for c, v in t.items() if v["B"] == B],
                       key=lambda kv: (kv[1]["kind"] != "ru", kv[1]["kind"], str(kv[1]["sig"]),
                                       -(kv[1]["alpha"] if kv[1]["alpha"] is not None else -1)))
        for c, v in order:
            p = v.get("paired", {})
            L.append(f"| {c} | {v['mean']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] | "
                     f"{(v['keep_frac_visual'] if v['keep_frac_visual'] is not None else float('nan')):.2f} | "
                     f"{fmt(p.get('vs_image_only'))} | {fmt(p.get('vs_past_only'))} | "
                     f"{fmt(p.get('vs_random'))} | {fmt(p.get('vs_oracle'))} |")
        L.append("")
    return "\n".join(L)


def strata_md(rows, metric, n_boot, base_correct_only):
    dataset = rows[0]["dataset"]
    labs = sorted({v for v in load_pair_labels(dataset).values() if v})
    if not labs:
        return ""
    L = ["### 근거 겹침 층별 (과거 질문 q0 ↔ 새 질문 짝 라벨; FULL 정답 질문만)" if base_correct_only
         else "### 근거 겹침 층별 (전체 질문)", ""]
    header_done = False
    for lab in labs:
        res = analyze(rows, metric, n_boot, base_correct_only, strata_filter={lab})
        if res["n_questions"] < 5:
            continue
        t = res["table"]
        Bs = sorted({v["B"] for v in t.values()})
        for B in Bs:
            conds = [(c, v) for c, v in t.items() if v["B"] == B]
            if not header_done:
                L += ["| 층 | B | n | " + " | ".join(c.split("@")[0] for c, _ in sorted(conds)) + " |",
                      "|---|---|---|" + "---|" * len(conds)]
                header_done = True
            L.append(f"| {lab} | {B:g}% | {res['n_questions']} | " +
                     " | ".join(f"{v['mean']:.3f}" for _, v in sorted(conds)) + " |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="em", choices=["em", "anls"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    rows, meta = load(a.pattern)
    if not rows:
        raise SystemExit(f"no rows for {a.pattern}")
    res_all = analyze(rows, a.metric, a.n_boot, False)
    res_ok = analyze(rows, a.metric, a.n_boot, True)
    title = a.title or f"Core–Delta 재사용 판 — {a.pattern}"
    text = (f"# {title}\n\nmetric = {a.metric}\n\n" + md(res_all, "전체 질문") + "\n\n"
            + md(res_ok, "FULL 정답 질문만") + "\n\n" + strata_md(rows, a.metric, max(200, a.n_boot // 10), True))
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out + ".md", "w").write(text)
    json.dump({"meta": meta, "all": res_all, "full_correct_only": res_ok},
              open(out + ".json", "w"), ensure_ascii=False, indent=1)
    print(text)
    print(f"[saved] {out}.md / .json")


if __name__ == "__main__":
    main()
