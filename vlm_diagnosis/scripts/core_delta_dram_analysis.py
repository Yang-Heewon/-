"""DRAM 판 판정 — 두 core 신호(kvzip / image) × DRAM 보관분 B × (core C, delta D) 격자.

각 칸의 EM(전체 질문 / FULL 정답 질문만)과 이미지 단위 bootstrap 95% CI, 그리고 짝지은 차이:
  core 가치      (B,C,D) − (B,0,D)      같은 읽기 전송량에서 미리 둔 core가 더해 주는 정답률
  delta 가치     (B,C,D) − core만 C     질문 시 D를 더 가져오는 가치
  vs KVzip 저장량 (B,C,D) − core만 B    같은 저장량(B)을 전부 GPU에 올린 KVzip 대비 (GPU B→C+D 절감의 대가)
  vs KVzip GPU   (B,C,D) − core만 C+D  같은 GPU 사용량의 KVzip 대비 (DRAM B를 둔 이득)
  vs 무작위 core / head−token / kvzip 신호 − image 신호 (같은 칸)

  python -m vlm_diagnosis.scripts.core_delta_dram_analysis \
      --pattern "results/smoke/wd_q25_sqa.shard*.jsonl" --out results/smoke/wd_q25_sqa_summary
"""
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
COND_RE = re.compile(r"^(?P<kind>wd|wdR|wdT)_(?P<sig>kvzip|image)(?:B(?P<B>[0-9.]+))?"
                     r"_C(?P<C>[0-9.]+)_D(?P<D>[0-9.]+)$")


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


def name(kind, sig, B, C, D):
    bt = "" if B >= 100 else f"B{B:g}"
    return f"{kind}_{sig}{bt}_C{C:g}_D{D:g}"


def analyze(rows, metric, n_boot, base_correct_only):
    full, per_q = {}, defaultdict(dict)
    for r in rows:
        key = (str(r["sample_id"]), r["question_id"])
        if r["condition_id"] == "FULL_KV":
            full[key] = r
        else:
            per_q[key][r["condition_id"]] = r
    keys = [k for k in per_q if k in full and (not base_correct_only or full[k]["em"] == 1)]
    cells = {}
    for c in sorted({c for k in keys for c in per_q[k]}):
        m = COND_RE.match(c)
        if m:
            cells[c] = (m.group("kind"), m.group("sig"), float(m.group("B")) if m.group("B") else 100.0,
                        float(m.group("C")), float(m.group("D")))

    def stat(c):
        per_img = defaultdict(list)
        for k in keys:
            if c in per_q[k]:
                per_img[k[0]].append(per_q[k][c][metric])
        return boot_ci(per_img, n_boot)

    def paired(c, ref):
        if ref not in cells or ref == c:
            return None
        per_img = defaultdict(list)
        for k in keys:
            if c in per_q[k] and ref in per_q[k]:
                per_img[k[0]].append(per_q[k][c][metric] - per_q[k][ref][metric])
        pt, lo, hi = boot_ci(per_img, n_boot)
        return {"diff": pt, "ci": [lo, hi]}

    table = {}
    for c, (kind, sig, B, C, D) in cells.items():
        pt, lo, hi = stat(c)
        fetch = [per_q[k][c].get("fetch_triples", 0) for k in keys if c in per_q[k]]
        vis = [per_q[k][c]["keep_frac_visual"] for k in keys if c in per_q[k]
               and per_q[k][c].get("keep_frac_visual") is not None]
        t = {"kind": kind, "sig": sig, "B": B, "C": C, "D": D, "mean": pt, "ci": [lo, hi],
             "n": len(fetch), "fetch_triples_mean": sum(fetch) / max(1, len(fetch)),
             "keep_frac_visual": (sum(vis) / len(vis)) if vis else None}
        if kind == "wd" and D > 0:
            other = "image" if sig == "kvzip" else "kvzip"
            t["paired"] = {
                "core_value": paired(c, name("wd", sig, B, 0.0, D)),
                "delta_value": paired(c, name("wd", sig, 100.0, C, 0.0)),
                "vs_kvzip_same_storage": paired(c, name("wd", sig, 100.0, B, 0.0)) if B < 100 else None,
                "vs_kvzip_same_gpu": paired(c, name("wd", sig, 100.0, C + D, 0.0)),
                "vs_random_core": paired(c, name("wdR", sig, B, C, D)),
                "head_vs_token": paired(c, name("wdT", sig, B, C, D)),
                "vs_other_signal": paired(c, name("wd", other, B, C, D)),
            }
        table[c] = t
    return {"n_questions": len(keys), "n_images": len({k[0] for k in keys}),
            "full_kv_mean": (sum(full[k][metric] for k in keys) / len(keys)) if keys else None,
            "table": table}


def fmt(d):
    if not d or d.get("diff") is None:
        return "—"
    lo, hi = d["ci"]
    return f"{d['diff']:+.3f}" if lo is None else f"{d['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def md(res, title):
    t = res["table"]
    L = [f"### {title} (질문 {res['n_questions']}, 이미지 {res['n_images']}, FULL {res['full_kv_mean']:.3f})", ""]
    sigs = sorted({v["sig"] for v in t.values()})
    for sig in sigs:
        L += [f"#### core 신호 = {sig}", ""]
        cores_only = sorted((v["C"], v["mean"], v["ci"]) for v in t.values()
                            if v["sig"] == sig and v["kind"] == "wd" and v["D"] == 0)
        L += ["core만(= KVzip 방식, 전부 GPU) 크기별: " + ", ".join(
            f"{C:g}% → {m:.3f}" for C, m, _ in cores_only), ""]
        Bs = sorted({v["B"] for v in t.values() if v["sig"] == sig and v["kind"] == "wd" and v["D"] > 0},
                    reverse=True)
        for B in Bs:
            sub = [v for v in t.values() if v["sig"] == sig and v["kind"] == "wd" and v["B"] == B and v["D"] > 0]
            Cs = sorted({v["C"] for v in sub})
            Ds = sorted({v["D"] for v in sub})
            head = "전부 보관(참고)" if B >= 100 else f"DRAM 보관 B={B:g}% (나머지 삭제)"
            L += [f"**{head}** — 행 = GPU 상주 core C, 열 = 질문 시 가져오는 delta D (접두 KV 대비 %)", "",
                  "| C \\\\ D | " + " | ".join(f"{D:g}%" for D in Ds) + " |", "|---|" + "---|" * len(Ds)]
            for C in Cs:
                cells = []
                for D in Ds:
                    v = t.get(name("wd", sig, B, C, D))
                    cells.append("—" if not v else f"{v['mean']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
                L.append(f"| **{C:g}%** | " + " | ".join(cells) + " |")
            L.append("")
        L += ["| 칸 (B/C/D) | core 가치 | delta 가치 | vs KVzip 같은 저장량(B) | vs KVzip 같은 GPU(C+D) | "
              "vs 무작위 core | head−token | 이 신호 − 다른 신호 | 전송 세 짝 | 시각 비율 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c, v in sorted(t.items(), key=lambda kv: (-kv[1]["B"], kv[1]["C"], kv[1]["D"])):
            if v["sig"] != sig or v["kind"] != "wd" or v["D"] == 0:
                continue
            p = v["paired"]
            L.append(f"| B{v['B']:g}/C{v['C']:g}/D{v['D']:g} | {fmt(p['core_value'])} | {fmt(p['delta_value'])} | "
                     f"{fmt(p['vs_kvzip_same_storage'])} | {fmt(p['vs_kvzip_same_gpu'])} | "
                     f"{fmt(p['vs_random_core'])} | {fmt(p['head_vs_token'])} | {fmt(p['vs_other_signal'])} | "
                     f"{v['fetch_triples_mean']:.0f} | "
                     f"{(v['keep_frac_visual'] if v['keep_frac_visual'] is not None else float('nan')):.2f} |")
        others = [(c, v) for c, v in t.items() if v["sig"] == sig and v["kind"] != "wd"]
        if others:
            L += ["", "| 대조 조건 | EM [CI] |", "|---|---|"]
            for c, v in sorted(others):
                L.append(f"| {c} | {v['mean']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] |")
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
    title = a.title or f"Core–Delta DRAM 판 — {a.pattern}"
    text = f"# {title}\n\nmetric = {a.metric}\n\n" + md(res_all, "전체 질문") + "\n\n" + md(res_ok, "FULL 정답 질문만")
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out + ".md", "w").write(text)
    json.dump({"meta": meta, "all": res_all, "full_correct_only": res_ok},
              open(out + ".json", "w"), ensure_ascii=False, indent=1)
    print(text)
    print(f"[saved] {out}.md / .json")


if __name__ == "__main__":
    main()
