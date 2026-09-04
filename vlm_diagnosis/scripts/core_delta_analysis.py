"""Fixed-budget dual-signal selector analysis.

core_delta_sweep 결과(jsonl)를 읽어 예산·alpha별로
  1) 조건별 평균 EM / ANLS (전체 질문, 그리고 FULL_KV가 맞힌 질문만)
  2) joint/query-only(alpha=0) 및 image/core-only(alpha=1) 대비 **짝지은 차이**와
     이미지 단위 bootstrap 95% CI
  3) 두 endpoint 대비 판정: 실패 / 부분 성공 / 성공
을 계산해 markdown + json으로 남긴다.

  python -m vlm_diagnosis.scripts.core_delta_analysis \
      --pattern "results/smoke/cd_q25_sqa.shard*.jsonl" --out results/smoke/cd_q25_sqa_summary
"""
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
COND_RE = re.compile(r"^(?P<name>.+?)@B(?P<budget>[0-9.]+)$")


def load(pattern):
    rows, meta, inv = [], [], []
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            t = r.get("record_type")
            if t == "run_metadata":
                meta.append(r)
            elif t == "kv_invariance":
                inv.append(r)
            else:
                rows.append(r)
    return rows, meta, inv


def boot_ci(per_img, fn, n_boot, seed=42):
    keys = list(per_img)
    rng = random.Random(seed)
    pt = fn([per_img[k] for k in keys])
    vals = []
    for _ in range(n_boot):
        v = fn([per_img[rng.choice(keys)] for _ in keys])
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return pt, None, None
    return pt, vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def mean_of_lists(lists):
    flat = [x for l in lists for x in l]
    return sum(flat) / len(flat) if flat else None


def analyze(rows, metric="em", n_boot=10000, base_correct_only=False):
    # (sample_id, question_id) → {condition_id: value}
    full = {}
    per_q = defaultdict(dict)
    for r in rows:
        key = (str(r["sample_id"]), r["question_id"])
        if r["condition_id"] == "FULL_KV":
            full[key] = r
        else:
            per_q[key][r["condition_id"]] = r
    if base_correct_only:
        keys = [k for k in per_q if k in full and full[k]["em"] == 1]
    else:
        keys = [k for k in per_q if k in full]
    def _cond_key(c):
        # 정렬: 통제군 → cd alpha 오름차순 → wsum w 오름차순
        m = re.match(r"^(cd|wsum)_(\w+?)_[aw]([0-9.]+)@", c)
        if m:
            return (1 if m.group(1) == "cd" else 2, m.group(2), float(m.group(3)))
        return (0, c, 0.0)
    conds = sorted({c for k in keys for c in per_q[k]}, key=_cond_key)
    by_budget = defaultdict(list)
    for c in conds:
        m = COND_RE.match(c)
        if m:
            by_budget[m.group("budget")].append(c)

    out = {"n_questions": len(keys), "n_images": len({k[0] for k in keys}),
           "full_kv_mean": (sum(full[k][metric] for k in keys) / len(keys)) if keys else None,
           "budgets": {}}
    for B, cs in sorted(by_budget.items(), key=lambda x: float(x[0])):
        tab = {}
        # 평균 (이미지 단위 CI)
        for c in cs:
            per_img = defaultdict(list)
            for k in keys:
                if c in per_q[k]:
                    per_img[k[0]].append(per_q[k][c][metric])
            pt, lo, hi = boot_ci(per_img, mean_of_lists, n_boot)
            tab[c] = {"mean": pt, "ci": [lo, hi], "n": sum(len(v) for v in per_img.values()),
                      "keep_tokens_mean": (sum(per_q[k][c].get("keep_tokens", per_q[k][c].get("kept_triples", 0))
                                               for k in keys if c in per_q[k])
                                           / max(1, sum(1 for k in keys if c in per_q[k])))}
        # 짝지은 차이: alpha 격자 vs a0 (joint/query-only) / a1 (image/core-only)
        paired = {}
        cores = sorted({re.match(r"cd_(\w+?)_a", c).group(1) for c in cs if c.startswith("cd_")})
        for core in cores:
            ref_s1 = f"cd_{core}_a0@B{B}"
            ref_core = f"cd_{core}_a1@B{B}"
            for c in cs:
                if not (c.startswith(f"cd_{core}_a") or c.startswith(f"wsum_{core}_")):
                    continue
                for ref_name, ref in (("vs_s1", ref_s1), ("vs_core", ref_core)):
                    if ref not in tab or c == ref:
                        continue
                    per_img = defaultdict(list)
                    for k in keys:
                        if c in per_q[k] and ref in per_q[k]:
                            per_img[k[0]].append(per_q[k][c][metric] - per_q[k][ref][metric])
                    pt, lo, hi = boot_ci(per_img, mean_of_lists, n_boot)
                    wins = sum(1 for v in per_img.values() for d in v if d > 0)
                    losses = sum(1 for v in per_img.values() for d in v if d < 0)
                    paired.setdefault(c, {})[ref_name] = {
                        "diff": pt, "ci": [lo, hi], "wins": wins, "losses": losses}
        # overlap 진단 (core top-B ∩ query top-B) — 시각 전용 판에서만 존재
        overlap = {}
        for c in cs:
            vals = [per_q[k][c]["core_delta"]["core_query_overlap"] / max(1, per_q[k][c]["keep_tokens"])
                    for k in keys if c in per_q[k] and "core_delta" in per_q[k][c]]
            if vals:
                overlap[c] = sum(vals) / len(vals)
        # 구성 진단 (전체 KV 판): 보존분 중 시각 비율, sink 보존 비율
        comp = {}
        for c in cs:
            vs = [per_q[k][c]["keep_frac_visual"] for k in keys
                  if c in per_q[k] and "keep_frac_visual" in per_q[k][c]]
            ss = [per_q[k][c]["sink_kept_frac"] for k in keys
                  if c in per_q[k] and "sink_kept_frac" in per_q[k][c]]
            if vs:
                comp[c] = {"keep_frac_visual": sum(vs) / len(vs),
                           "sink_kept_frac": sum(ss) / max(1, len(ss))}
        # §10 Step 3 판정 (core별)
        verdicts = {}
        for core in cores:
            mids = [c for c in cs if c.startswith(f"cd_{core}_a")
                    and re.search(r"_a(0|1)@", c) is None]
            if not mids:
                continue
            best = max(mids, key=lambda c: tab[c]["mean"] if tab[c]["mean"] is not None else -1)
            d_s1 = paired.get(best, {}).get("vs_s1", {})
            d_core = paired.get(best, {}).get("vs_core", {})
            diffs = [(paired.get(c, {}).get("vs_s1", {}).get("diff") or 0) for c in mids]
            all_below_s1 = all(d <= 0 for d in diffs)
            if all_below_s1 and all(d == 0 for d in diffs):
                v = "동률: 모든 중간 alpha가 joint/query-only와 차이 0 (표본이 작거나 mask가 같음)"
            elif all_below_s1:
                v = "실패: 모든 중간 alpha가 joint/query-only 이하 → image/core branch 기각"
            elif d_s1.get("ci", [None])[0] is not None and d_s1["ci"][0] > 0 and \
                    d_core.get("ci", [None])[0] is not None and d_core["ci"][0] > 0:
                v = "성공 후보: 최선 중간 alpha가 두 endpoint 모두 CI 하한>0 으로 상회 (재현 필요)"
            else:
                v = "부분/불확정: 점추정은 S1 초과이나 CI가 0을 포함 → 표본 확대 또는 조건부 해석"
            verdicts[core] = {"best_mid_alpha_condition": best, "verdict": v,
                              "best_vs_s1": d_s1, "best_vs_core": d_core}
        out["budgets"][B] = {"table": tab, "paired": paired, "overlap_frac_mean": overlap,
                             "composition": comp, "verdicts": verdicts}
    return out


def fmt_ci(d):
    if d is None or d.get("diff") is None:
        return "—"
    lo, hi = d["ci"]
    if lo is None:
        return f"{d['diff']:+.3f}"
    return f"{d['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def to_markdown(title, res_all, res_ok, metric):
    L = [f"# {title}", "",
         f"metric = {metric}; 질문 수 {res_all['n_questions']} (이미지 {res_all['n_images']}); "
         f"FULL_KV 평균 {res_all['full_kv_mean']:.3f}; "
         f"FULL 정답 질문만: {res_ok['n_questions']}개", ""]
    for B in res_all["budgets"]:
        ra, ro = res_all["budgets"][B], res_ok["budgets"].get(B, {"table": {}, "paired": {}})
        has_comp = bool(ra.get("composition"))
        last_col = "시각비율 / sink보존" if has_comp else "overlap"
        L += [f"## 예산 B={B}% (keep_tokens 평균 "
              f"{next(iter(ra['table'].values()))['keep_tokens_mean']:.1f})", "",
              "| 조건 | 평균(전체) [CI] | Δ vs joint/S1 [CI] (win/loss) | "
              "Δ vs image/core [CI] | "
              f"평균(FULL정답만) | Δ vs joint/S1 (FULL정답만) | {last_col} |",
              "|---|---|---|---|---|---|---|"]
        for c, t in ra["table"].items():
            p = ra["paired"].get(c, {})
            po = ro["paired"].get(c, {})
            to = ro["table"].get(c)
            ws = p.get("vs_s1")
            wl = f" ({ws['wins']}/{ws['losses']})" if ws else ""
            if has_comp:
                cc = ra["composition"].get(c)
                last = (f"{cc['keep_frac_visual']:.2f} / {cc['sink_kept_frac']:.2f}"
                        if cc else "—")
            else:
                last = f"{ra['overlap_frac_mean'].get(c, float('nan')):.2f}"
            L.append(
                f"| {c} | {t['mean']:.3f} [{t['ci'][0]:.3f},{t['ci'][1]:.3f}] | "
                f"{fmt_ci(ws)}{wl} | {fmt_ci(p.get('vs_core'))} | "
                f"{(to['mean'] if to else float('nan')):.3f} | {fmt_ci(po.get('vs_s1'))} | "
                f"{last} |")
        L.append("")
        for core, v in ra["verdicts"].items():
            L += [f"**판정 (signal={core}, 전체 질문)**: {v['verdict']}  ",
                  f"최선 중간 alpha = `{v['best_mid_alpha_condition']}` — vs joint/S1 "
                  f"{fmt_ci(v['best_vs_s1'])}, vs image/core {fmt_ci(v['best_vs_core'])}", ""]
        for core, v in ro.get("verdicts", {}).items():
            L += [f"**판정 (signal={core}, FULL 정답 질문만)**: {v['verdict']}  ",
                  f"최선 = `{v['best_mid_alpha_condition']}` — vs joint/S1 "
                  f"{fmt_ci(v['best_vs_s1'])}", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", required=True, help="확장자 없는 경로 → .md/.json")
    ap.add_argument("--metric", default="em", choices=["em", "anls"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    rows, meta, inv = load(a.pattern)
    if not rows:
        raise SystemExit(f"no rows for {a.pattern}")
    res_all = analyze(rows, a.metric, a.n_boot, base_correct_only=False)
    res_ok = analyze(rows, a.metric, a.n_boot, base_correct_only=True)
    title = a.title or f"Core–Delta Phase A — {a.pattern}"
    md = to_markdown(title, res_all, res_ok, a.metric)
    if inv:
        md += "\n\n## §3 검증: 시각 KV 불변성 (질문 교체 전후)\n\n"
        for r in inv:
            ctrl = r.get("same_text_control", {})
            md += (f"- sample {r['sample_id']} (시각 {r['n_visual']} tok, {r['n_layers']}층): "
                   f"질문 교체 → max|ΔK|={r['max_abs_dK']:.3g} (|K|max {r['K_abs_max']:.3g}), "
                   f"max|ΔV|={r['max_abs_dV']:.3g} (|V|max {r['V_abs_max']:.3g}), "
                   f"상대 Frobenius ΔK {r.get('rel_fro_dK', float('nan')):.2e} / "
                   f"ΔV {r.get('rel_fro_dV', float('nan')):.2e}; "
                   f"같은 텍스트 2회 대조 ΔK {ctrl.get('rel_fro_dK', float('nan')):.2e} / "
                   f"ΔV {ctrl.get('rel_fro_dV', float('nan')):.2e} → "
                   f"bit-동일={r.get('identical_bitwise', r.get('identical'))}, "
                   f"fp16 잡음 이내={r.get('identical_up_to_fp16_noise')}\n")
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out + ".md", "w").write(md)
    json.dump({"meta": meta, "kv_invariance": inv, "all": res_all, "full_correct_only": res_ok},
              open(out + ".json", "w"), ensure_ascii=False, indent=1)
    print(md)
    print(f"[saved] {out}.md / .json")


if __name__ == "__main__":
    main()
