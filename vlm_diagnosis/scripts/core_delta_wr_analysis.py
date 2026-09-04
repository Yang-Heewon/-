"""쓰기/읽기 판 판정 — (core 크기 C, delta 크기 D) 격자의 정답률과 core·delta 각각의 가치.

  1) 격자 표: 각 (C, D) 칸의 EM (전체 질문 / FULL 정답 질문만), 이미지 단위 bootstrap 95% CI
  2) core의 가치: (C, D) − (0, D)  같은 읽기 전송량에서 core가 더해 주는 정답률
  3) delta의 가치: (C, D) − (C, 0)
  4) 내용 검정: (C, D) − 무작위 core (C, D)
  5) 등가 전송량: core 없이 같은 정답률을 내려면 D가 얼마나 커야 하는가 (격자 내 보간 없이 표로)

  python -m vlm_diagnosis.scripts.core_delta_wr_analysis \
      --pattern "results/smoke/wr_q25_sqa.shard*.jsonl" --out results/smoke/wr_q25_sqa_summary
"""
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
COND_RE = re.compile(r"^(?P<kind>wr|wrR|wrT)(?:B(?P<B>[0-9.]+))?_C(?P<C>[0-9.]+)_D(?P<D>[0-9.]+)$")


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


def analyze(rows, metric, n_boot, base_correct_only):
    full, per_q = {}, defaultdict(dict)
    for r in rows:
        key = (str(r["sample_id"]), r["question_id"])
        if r["condition_id"] == "FULL_KV":
            full[key] = r
        else:
            per_q[key][r["condition_id"]] = r
    keys = [k for k in per_q if k in full and (not base_correct_only or full[k]["em"] == 1)]
    conds = sorted({c for k in keys for c in per_q[k]})
    cells = {}
    for c in conds:
        m = COND_RE.match(c)
        if not m:
            continue
        cells[c] = (m.group("kind"), float(m.group("C")), float(m.group("D")),
                    float(m.group("B")) if m.group("B") else 100.0)

    def stat(c):
        per_img = defaultdict(list)
        for k in keys:
            if c in per_q[k]:
                per_img[k[0]].append(per_q[k][c][metric])
        return boot_ci(per_img, n_boot)

    def paired(c, ref):
        per_img = defaultdict(list)
        for k in keys:
            if c in per_q[k] and ref in per_q[k]:
                per_img[k[0]].append(per_q[k][c][metric] - per_q[k][ref][metric])
        pt, lo, hi = boot_ci(per_img, n_boot)
        return {"diff": pt, "ci": [lo, hi]}

    table = {}
    for c, (kind, C, D, B) in cells.items():
        pt, lo, hi = stat(c)
        fetch = [per_q[k][c].get("fetch_triples", 0) for k in keys if c in per_q[k]]
        hot = [per_q[k][c].get("hot_triples", 0) for k in keys if c in per_q[k]]
        vis = [per_q[k][c].get("keep_frac_visual") for k in keys if c in per_q[k]
               and per_q[k][c].get("keep_frac_visual") is not None]
        table[c] = {"kind": kind, "C": C, "D": D, "B": B, "mean": pt, "ci": [lo, hi],
                    "n": len(fetch), "fetch_triples_mean": sum(fetch) / max(1, len(fetch)),
                    "hot_triples_mean": sum(hot) / max(1, len(hot)),
                    "keep_frac_visual": (sum(vis) / len(vis)) if vis else None}
        if kind == "wr":
            bt = "" if B >= 100 else f"B{B:g}"
            refs = {}
            if C > 0 and f"wr{bt}_C0_D{D:g}" in cells:
                refs["core_value_vs_delta_only"] = paired(c, f"wr{bt}_C0_D{D:g}")
            if D > 0 and f"wr_C{C:g}_D0" in cells:
                refs["delta_value_vs_core_only"] = paired(c, f"wr_C{C:g}_D0")
            if f"wrR{bt}_C{C:g}_D{D:g}" in cells:
                refs["vs_random_core"] = paired(c, f"wrR{bt}_C{C:g}_D{D:g}")
            if f"wrT{bt}_C{C:g}_D{D:g}" in cells:
                refs["head_vs_token"] = paired(c, f"wrT{bt}_C{C:g}_D{D:g}")
            if B < 100:
                if f"wr_C{B:g}_D0" in cells:            # 같은 저장량의 KVzip (B 전부 GPU)
                    refs["vs_kvzip_same_storage"] = paired(c, f"wr_C{B:g}_D0")
                if f"wr_C{C+D:g}_D0" in cells:          # 같은 GPU 사용량의 KVzip
                    refs["vs_kvzip_same_gpu"] = paired(c, f"wr_C{C+D:g}_D0")
                if f"wr_C{C:g}_D{D:g}" in cells:        # 전부 보관(B=100) 대비 삭제의 손실
                    refs["vs_keep_all"] = paired(c, f"wr_C{C:g}_D{D:g}")
            table[c]["paired"] = refs
    return {"n_questions": len(keys), "n_images": len({k[0] for k in keys}),
            "full_kv_mean": (sum(full[k][metric] for k in keys) / len(keys)) if keys else None,
            "table": table}


def fmt(d):
    if not d or d.get("diff") is None:
        return "—"
    lo, hi = d["ci"]
    return f"{d['diff']:+.3f}" if lo is None else f"{d['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def grid_md(res, title):
    t = res["table"]
    Bs = sorted({v["B"] for v in t.values() if v["kind"] == "wr"}, reverse=True)
    L = [f"### {title} (질문 {res['n_questions']}, 이미지 {res['n_images']}, "
         f"FULL {res['full_kv_mean']:.3f})", ""]
    for B in Bs:
        sub = {c: v for c, v in t.items() if v["kind"] == "wr" and (v["B"] == B or v["D"] == 0)}
        Cs = sorted({v["C"] for v in sub.values()} | {0.0})
        Ds = sorted({v["D"] for v in sub.values()} | {0.0})
        head = "전부 보관(참고)" if B >= 100 else f"DRAM 보관 B={B:g}% (나머지 삭제)"
        L += [f"#### {head}", "",
              "행 = GPU 상주 core 크기 C, 열 = 질문 시 가져오는 delta 크기 D (접두 KV 대비 %). "
              "D=0 열 = core만(KVzip 방식, B 무관). 칸 = EM [95% CI]", "",
              "| C \\ D | " + " | ".join(f"{D:g}%" for D in Ds) + " |",
              "|---|" + "---|" * len(Ds)]
        for C in Cs:
            cells = []
            for D in Ds:
                bt = "" if (B >= 100 or D == 0) else f"B{B:g}"
                v = t.get(f"wr{bt}_C{C:g}_D{D:g}")
                cells.append("—" if not v else f"{v['mean']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
            L.append(f"| **{C:g}%** | " + " | ".join(cells) + " |")
        L.append("")
    L += ["| 칸 (B/C/D) | core 가치 (C,D)−(0,D) | delta 가치 (C,D)−(C,0) | vs 무작위 core | head−token | "
          "vs KVzip 같은 저장량(B) | vs KVzip 같은 GPU(C+D) | vs 전부 보관 | 전송 세 짝 | 시각 비율 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in sorted(t.items(), key=lambda kv: (-kv[1]["B"], kv[1]["C"], kv[1]["D"])):
        if v["kind"] != "wr" or v["D"] == 0:
            continue
        p = v.get("paired", {})
        L.append(f"| B{v['B']:g}/C{v['C']:g}/D{v['D']:g} | {fmt(p.get('core_value_vs_delta_only'))} | "
                 f"{fmt(p.get('delta_value_vs_core_only'))} | {fmt(p.get('vs_random_core'))} | "
                 f"{fmt(p.get('head_vs_token'))} | {fmt(p.get('vs_kvzip_same_storage'))} | "
                 f"{fmt(p.get('vs_kvzip_same_gpu'))} | {fmt(p.get('vs_keep_all'))} | "
                 f"{v['fetch_triples_mean']:.0f} | "
                 f"{(v['keep_frac_visual'] if v['keep_frac_visual'] is not None else float('nan')):.2f} |")
    others = [(c, v) for c, v in t.items() if v["kind"] != "wr"]
    if others:
        L += ["", "| 대조 조건 | EM [CI] |", "|---|---|"]
        for c, v in sorted(others, key=lambda kv: (kv[1]["kind"], -kv[1]["B"], kv[1]["C"], kv[1]["D"])):
            L.append(f"| {c} | {v['mean']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] |")
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
    title = a.title or f"Core–Delta 쓰기/읽기 — {a.pattern}"
    md = f"# {title}\n\nmetric = {a.metric}\n\n" + grid_md(res_all, "전체 질문") + "\n\n" + \
        grid_md(res_ok, "FULL 정답 질문만")
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out + ".md", "w").write(md)
    json.dump({"meta": meta, "all": res_all, "full_correct_only": res_ok},
              open(out + ".json", "w"), ensure_ascii=False, indent=1)
    print(md)
    print(f"[saved] {out}.md / .json")


if __name__ == "__main__":
    main()
