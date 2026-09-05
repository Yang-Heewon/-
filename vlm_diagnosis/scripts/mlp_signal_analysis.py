"""MLP 신호 probe 판정 — 신호별 정답률(FULL 정답 / 전체), kvzip·attn1 대비 짝지은 차이(이미지 단위
bootstrap 95% CI), 신호와 kvzip의 순위 상관·상위 겹침 평균.

  python -m vlm_diagnosis.scripts.mlp_signal_analysis --pattern "results/smoke/mlp_q25_sqa.shard*.jsonl" --out results/smoke/mlp_q25_sqa_summary
"""
import argparse, glob, json, os, random
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def load(pattern):
    rows, diags = [], []
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            t = r.get("record_type")
            if t == "run_metadata":
                continue
            (diags if t == "signal_diag" else rows).append(r)
    return rows, diags


def boot(per_img, n_boot, seed=42):
    keys = list(per_img)
    if not keys:
        return None, None, None
    rng = random.Random(seed)
    def mean(ls):
        flat = [x for l in ls for x in l]
        return sum(flat) / len(flat) if flat else None
    pt = mean([per_img[k] for k in keys])
    vals = sorted(v for v in (mean([per_img[rng.choice(keys)] for _ in keys]) for _ in range(n_boot)) if v is not None)
    return pt, vals[int(.025 * len(vals))], vals[min(len(vals) - 1, int(.975 * len(vals)))]


def fmt(d):
    if not d or d[0] is None:
        return "—"
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"


def analyze(rows, metric, n_boot, full_only):
    full, per_q = {}, defaultdict(dict)
    for r in rows:
        k = (str(r["sample_id"]), r["question_id"])
        if r["condition_id"] == "FULL_KV":
            full[k] = r
        else:
            per_q[k][r["condition_id"]] = r
    keys = [k for k in per_q if k in full and (not full_only or full[k]["em"] == 1)]
    conds = sorted({c for k in keys for c in per_q[k]})
    def stat(c):
        d = defaultdict(list)
        for k in keys:
            if c in per_q[k]:
                d[k[0]].append(per_q[k][c][metric])
        return boot(d, n_boot)
    def paired(c, ref):
        if ref not in conds or c == ref:
            return None
        d = defaultdict(list)
        for k in keys:
            if c in per_q[k] and ref in per_q[k]:
                d[k[0]].append(per_q[k][c][metric] - per_q[k][ref][metric])
        return boot(d, n_boot)
    table = {}
    for c in conds:
        B = c.split("@")[1]
        vis = [per_q[k][c].get("keep_frac_visual") for k in keys if c in per_q[k] and per_q[k][c].get("keep_frac_visual") is not None]
        table[c] = {"mean": stat(c), "vs_kvzip": paired(c, f"kvzip@{B}"), "vs_attn1": paired(c, f"attn1@{B}"),
                    "keep_frac_visual": sum(vis) / len(vis) if vis else None}
    return {"n_questions": len(keys), "n_images": len({k[0] for k in keys}),
            "full_mean": sum(full[k][metric] for k in keys) / len(keys) if keys else None, "table": table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="em"); ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    rows, diags = load(a.pattern)
    if not rows:
        raise SystemExit("no rows")
    L = ["# MLP 신호 probe — " + a.pattern, ""]
    for title, full_only in [("FULL 정답 질문만", True), ("전체 질문", False)]:
        res = analyze(rows, a.metric, a.n_boot, full_only)
        L += [f"## {title} (질문 {res['n_questions']}, 이미지 {res['n_images']}, FULL {res['full_mean']:.3f})", "",
              "| 조건 | EM [95% CI] | − kvzip | − attn1 | 시각 비율 |", "|---|---|---|---|---|"]
        for c, v in sorted(res["table"].items(), key=lambda kv: -(kv[1]["mean"][0] or 0)):
            m = v["mean"]
            L.append(f"| {c} | {m[0]:.3f} [{m[1]:.3f},{m[2]:.3f}] | {fmt(v['vs_kvzip'])} | {fmt(v['vs_attn1'])} | "
                     f"{(v['keep_frac_visual'] if v['keep_frac_visual'] is not None else float('nan')):.2f} |")
        L.append("")
    if diags:
        sp, ov = defaultdict(list), defaultdict(list)
        for d in diags:
            for k, v in d.get("spearman_vs_kvzip", {}).items():
                sp[k].append(v)
            for k, v in d.get("top_overlap_vs_kvzip", {}).items():
                ov[k].append(v)
        L += [f"## 신호 진단 (화면 {len(diags)}장 평균)", "", "| 신호 | kvzip과 Spearman 순위 상관 | 상위 5% 조각 겹침 |", "|---|---|---|"]
        for k in sp:
            o = ov.get(f"{k}@B5") or ov.get(next((x for x in ov if x.startswith(k + "@")), ""), [])
            L.append(f"| {k} | {sum(sp[k])/len(sp[k]):.3f} | {(sum(o)/len(o)) if o else float('nan'):.3f} |")
    text = "\n".join(L)
    out = os.path.join(ROOT, a.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out + ".md", "w").write(text); print(text); print(f"[saved] {out}.md")


if __name__ == "__main__":
    main()
