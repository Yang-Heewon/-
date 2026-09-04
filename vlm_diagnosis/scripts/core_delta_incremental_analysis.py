"""누적 선택 판 판정 — 단계 t(질문 t, 새 질문)마다 방식(keep/grow) × alpha 의 정답률과 짝지은 차이.

  - alpha=1 = 이미지만(KVzip). 각 (방식, t)에서 alpha<1 − alpha=1 = 기록 기준을 섞은 가치
  - − random(같은 크기), − oracle(같은 크기, 그 질문을 아는 상한)
  - grow 는 크기가 20·t % 로 커지므로 keep(20%)과는 크기가 다르다. 표에 크기를 같이 적는다.
  - 근거 겹침 층: 질문 t와 이전 질문들 중 가장 가까운 짝 라벨(T2 > T3 > ...)로 나눠 본다.

  python -m vlm_diagnosis.scripts.core_delta_incremental_analysis \
      --pattern "results/smoke/inc_q25_sqa.shard*.jsonl" --out results/smoke/inc_q25_sqa_summary
"""
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
COND_RE = re.compile(r"^inc_(?P<scheme>keep|grow)_a(?P<a>[0-9.]+)_t(?P<t>\d+)$")
CTRL_RE = re.compile(r"^(?P<name>random|oracle)_(?P<scheme>keep|grow)_t(?P<t>\d+)$")
PAIRS = {"ScreenQA": "experiments/manifests/screenqa_discovery_pairs.jsonl",
         "GQA": "experiments/manifests/gqa_discovery_pairs.jsonl"}
LABEL_RANK = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}


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


def closest_label(pair_lab, sample_id, qid, history):
    best = None
    for h in history:
        lab = pair_lab.get((sample_id, h, qid))
        if lab in LABEL_RANK and (best is None or LABEL_RANK[lab] < LABEL_RANK[best]):
            best = lab
    return best


def analyze(rows, metric, n_boot, base_correct_only, strata_filter=None):
    dataset = rows[0]["dataset"]
    pair_lab = load_pair_labels(dataset)
    full, per_q, labels = {}, defaultdict(dict), {}
    for r in rows:
        key = (str(r["sample_id"]), r["question_id"])
        if r["condition_id"] == "FULL_KV":
            full[key] = r
            labels[key] = closest_label(pair_lab, str(r["sample_id"]), r["question_id"],
                                        r.get("history_question_ids", []))
        else:
            per_q[key][r["condition_id"]] = r
    keys = [k for k in per_q if k in full and (not base_correct_only or full[k]["em"] == 1)
            and (strata_filter is None or labels.get(k) in strata_filter)]

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
    for c in sorted({c for k in keys for c in per_q[k]}):
        m, mc = COND_RE.match(c), CTRL_RE.match(c)
        if not (m or mc):
            continue
        pt, lo, hi = stat(c)
        n = sum(1 for k in keys if c in per_q[k])
        size = [per_q[k][c].get("size_frac") for k in keys if c in per_q[k]]
        vis = [per_q[k][c].get("keep_frac_visual") for k in keys if c in per_q[k]
               and per_q[k][c].get("keep_frac_visual") is not None]
        t = {"mean": pt, "ci": [lo, hi], "n": n, "size_frac": (sum(size) / len(size)) if size else None,
             "keep_frac_visual": (sum(vis) / len(vis)) if vis else None}
        if m:
            sch, al, st = m.group("scheme"), float(m.group("a")), int(m.group("t"))
            t.update({"kind": "inc", "scheme": sch, "alpha": al, "t": st})
            t["paired"] = {"vs_image_only": paired(c, f"inc_{sch}_a1_t{st}"),
                           "vs_random": paired(c, f"random_{sch}_t{st}"),
                           "vs_oracle": paired(c, f"oracle_{sch}_t{st}")}
        else:
            t.update({"kind": mc.group("name"), "scheme": mc.group("scheme"), "alpha": None,
                      "t": int(mc.group("t"))})
        table[c] = t
    counts = defaultdict(int)
    for k in keys:
        counts[labels.get(k)] += 1
    return {"n_questions": len(keys), "n_images": len({k[0] for k in keys}),
            "full_kv_mean": (sum(full[k][metric] for k in keys) / len(keys)) if keys else None,
            "strata_counts": dict(counts), "table": table}


def md(res, title):
    t = res["table"]
    L = [f"### {title} (질문 {res['n_questions']}, 이미지 {res['n_images']}, FULL {res['full_kv_mean']:.3f}; "
         f"층 분포 {res['strata_counts']})", ""]
    steps = sorted({v["t"] for v in t.values()})
    for st in steps:
        L += [f"#### 단계 t={st} (질문 {st}, 이전 질문 {st-1}개의 기록 사용) — EM [95% CI]", "",
              "| 조건 | 크기 | EM | 시각 비율 | − 이미지만(KVzip, 같은 방식·크기) | − random | − oracle |",
              "|---|---|---|---|---|---|---|"]
        order = sorted([(c, v) for c, v in t.items() if v["t"] == st],
                       key=lambda kv: (kv[1]["scheme"], kv[1]["kind"] != "inc", kv[1]["kind"],
                                       -(kv[1]["alpha"] if kv[1]["alpha"] is not None else -1)))
        for c, v in order:
            p = v.get("paired", {})
            L.append(f"| {c} | {v['size_frac']*100:.0f}% | {v['mean']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] | "
                     f"{(v['keep_frac_visual'] if v['keep_frac_visual'] is not None else float('nan')):.2f} | "
                     f"{fmt(p.get('vs_image_only'))} | {fmt(p.get('vs_random'))} | {fmt(p.get('vs_oracle'))} |")
        L.append("")
    return "\n".join(L)


def strata_md(rows, metric, n_boot):
    labs = sorted({v for v in load_pair_labels(rows[0]["dataset"]).values() if v})
    if not labs:
        return ""
    L = ["### 근거 겹침 층별 (질문 t와 이전 질문들 중 가장 가까운 짝 라벨; FULL 정답 질문만, t≥2)", ""]
    header = None
    for lab in labs:
        res = analyze(rows, metric, n_boot, True, strata_filter={lab})
        if res["n_questions"] < 5:
            continue
        t = {c: v for c, v in res["table"].items() if v["t"] >= 2}
        for st in sorted({v["t"] for v in t.values()}):
            conds = sorted([(c, v) for c, v in t.items() if v["t"] == st])
            if header is None:
                header = [c.rsplit("_t", 1)[0] for c, _ in conds]
                L += ["| 층 | t | n | " + " | ".join(header) + " |", "|---|---|---|" + "---|" * len(header)]
            L.append(f"| {lab} | {st} | {sum(1 for _ in conds) and res['n_questions']} | " +
                     " | ".join(f"{v['mean']:.3f}" for _, v in conds) + " |")
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
    title = a.title or f"Core–Delta 누적 선택 판 — {a.pattern}"
    text = (f"# {title}\n\nmetric = {a.metric}\n\n" + md(res_all, "전체 질문") + "\n\n"
            + md(res_ok, "FULL 정답 질문만") + "\n\n" + strata_md(rows, a.metric, max(200, a.n_boot // 10)))
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out + ".md", "w").write(text)
    json.dump({"meta": meta, "all": res_all, "full_correct_only": res_ok},
              open(out + ".json", "w"), ensure_ascii=False, indent=1)
    print(text)
    print(f"[saved] {out}.md / .json")


if __name__ == "__main__":
    main()
