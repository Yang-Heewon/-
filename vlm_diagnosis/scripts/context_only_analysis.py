"""Context-only KV 결과 분석 — 엄격한 로그 검증, context 단위 paired bootstrap, 표 출력
(docs/CONTEXT-ONLY-KV-COMPRESSION.md §8).

  python -m vlm_diagnosis.scripts.context_only_analysis --pattern "results/context_only/deletion_qwen25vl_dev*.jsonl" --out results/context_only/deletion_dev_summary

지표: EM(전체), FULL-correct retention (FULL 정답 질문 중 조건도 정답), delta_NLL = 조건 NLL − FULL NLL, loyalty.
집계: context 별 질문 평균 → context 평균 (macro). bootstrap 은 context 재표집 (질문 묶음 유지).
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FULL = "full|none|none|k1|s0"


class LogError(Exception):
    pass


def load(pattern):
    runs, builds, answers, diags, parity, errors = [], [], [], [], [], []
    seen = set()
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        for ln, line in enumerate(open(p), 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                raise LogError(f"malformed JSON at {p}:{ln}")
            t = r.get("record_type")
            if t == "run":
                runs.append(r)
            elif t == "build":
                builds.append(r)
            elif t == "answer":
                key = (r["context_id"], r["question_id"], r["condition"])
                if key in seen:
                    raise LogError(f"duplicate answer record {key} at {p}:{ln}")
                seen.add(key); answers.append(r)
            elif t == "diagnostic":
                diags.append(r)
            elif t == "parity":
                parity.append(r)
            elif t == "error":
                errors.append(r)
            elif t == "context_done":
                pass
            else:
                raise LogError(f"unknown record_type {t!r} at {p}:{ln}")
    if not runs:
        raise LogError("no run metadata")
    keys = ("schema_version", "model", "manifest_sha256", "stage", "protect", "nll_rule", "max_new_tokens")
    base = {k: runs[0].get(k) for k in keys}
    for r in runs[1:]:
        if {k: r.get(k) for k in keys} != base:
            raise LogError(f"incompatible run metadata: {base} vs { {k: r.get(k) for k in keys} }")
    return runs, builds, answers, diags, parity, errors


def macro(per_ctx):
    vals = [sum(v) / len(v) for v in per_ctx.values() if v]
    return sum(vals) / len(vals) if vals else None


def boot(per_ctx, n_boot=5000, seed=42):
    ctx = [k for k, v in per_ctx.items() if v]
    if not ctx:
        return None
    rng = random.Random(seed)
    m = {k: sum(v) / len(v) for k, v in per_ctx.items() if v}
    pt = sum(m.values()) / len(m)
    bs = sorted(sum(m[rng.choice(ctx)] for _ in ctx) / len(ctx) for _ in range(n_boot))
    return pt, bs[int(.025 * n_boot)], bs[min(n_boot - 1, int(.975 * n_boot))], len(ctx)


def fmt(b, digits=3):
    if b is None:
        return "—"
    return f"{b[0]:.{digits}f} [{b[1]:.{digits}f},{b[2]:.{digits}f}]"


def fmtd(b):
    if b is None:
        return "—"
    return f"{b[0]:+.3f} [{b[1]:+.3f},{b[2]:+.3f}]"


def analyze(answers, n_boot):
    full = {}
    per = defaultdict(dict)                     # (ctx, q) -> cond -> rec
    for r in answers:
        if r.get("status") != "ok":
            continue
        k = (r["context_id"], r["question_id"])
        if r["condition"] == FULL:
            full[k] = r
        per[k][r["condition"]] = r
    keys = [k for k in per if k in full]
    missing_full = [k for k in per if k not in full]
    conds = sorted({c for k in keys for c in per[k] if c != FULL})
    out = {}
    for c in conds:
        em, ret, dnll, loy, gen = (defaultdict(list) for _ in range(5))
        for k in keys:
            r = per[k].get(c)
            if r is None:
                continue
            em[k[0]].append(r["em"])
            if full[k]["em"] == 1:
                ret[k[0]].append(r["em"])
            if r.get("nll") is not None and full[k].get("nll") is not None:
                dnll[k[0]].append(r["nll"] - full[k]["nll"])
            if r.get("loyalty") is not None:
                loy[k[0]].append(r["loyalty"])
            gen[k[0]].append(r["generated_tokens"])
        out[c] = {"em": boot(em, n_boot), "retention": boot(ret, n_boot), "delta_nll": boot(dnll, n_boot),
                  "loyalty": boot(loy, n_boot), "gen_tokens": macro(gen),
                  "n_questions": sum(len(v) for v in em.values())}
    full_em = boot({k[0]: [] for k in keys} | {k[0]: [full[j]["em"] for j in keys if j[0] == k[0]] for k in keys}, n_boot)
    return {"conds": out, "full_em": full_em, "n_pairs": len(keys), "n_contexts": len({k[0] for k in keys}),
            "missing_full": len(missing_full), "per": per, "full": full, "keys": keys}


def paired(res, c, ref, metric, n_boot):
    per, full, keys = res["per"], res["full"], res["keys"]
    d = defaultdict(list)
    for k in keys:
        a, b = per[k].get(c), per[k].get(ref)
        if a is None or b is None:
            continue
        if metric == "retention" and full[k]["em"] != 1:
            continue
        if metric == "delta_nll":
            if a.get("nll") is None or b.get("nll") is None:
                continue
            d[k[0]].append(a["nll"] - b["nll"])
        else:
            d[k[0]].append(a["em"] - b["em"])
    return boot(d, n_boot)


def parse_cond(c):
    m, d, s, k, seed = c.split("|")
    return m, d, s, float(k[1:]), int(seed[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    runs, builds, answers, diags, parity, errors = load(a.pattern)
    stage = runs[0]["stage"]
    L = [f"# Context-only KV — {stage} — {a.pattern}", "",
         f"model {runs[0]['model']} · code {runs[0]['code_revision'][:8]}{' (dirty)' if runs[0]['code_dirty'] else ''} · "
         f"split {runs[0]['split']} · protect: {runs[0]['protect']} · NLL: {runs[0]['nll_rule']}", ""]
    if errors:
        L += [f"**오류 기록 {len(errors)}건** (context 제외됨): " + ", ".join(sorted({e['context_id'] for e in errors})[:10]), ""]
    if parity:
        L += ["## 단계 1 parity (keep=100% ragged 경로 vs 일반 FULL forward)", "",
              "| context | 질문 | 위치 일치 | logits 최대 오차 | 평균 오차 | argmax 일치율 | 첫 답 token 일치 | NLL cached | NLL dense | NLL 차 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for p in parity:
            L.append(f"| {p['context_id']} | {p['question_id']} | {p['positions_match']} | {p['logit_max_abs_diff']:.4f} | "
                     f"{p['logit_mean_abs_diff']:.5f} | {p['argmax_agreement']:.3f} | {p['first_answer_token_agree']} | "
                     f"{p['nll_cached']:.4f} | {p['nll_dense']:.4f} | {p['nll_abs_diff']:.5f} |")
        L.append("")
    if diags and stage == "probe":
        L += ["## 단계 2 관측 (단일 prefill 통계)", "", "| context | token | 시각 | 잔차 검사 최대 상대오차 | R 시각/비시각 평균 | D 시각/비시각 평균 | R–D token 순위 상관 | 상위 D token |", "|---|---|---|---|---|---|---|---|"]
        for d in diags:
            top = ", ".join(f"{t['piece']}@{t['index']}" for t in d["top_tokens_by_D"][:5])
            L.append(f"| {d['context_id']} | {d['n_tokens']} | {d['n_visual']} | {max(d['residual_max_rel_err']):.2e} | "
                     f"{d['R_visual']['mean']:.3f}/{d['R_nonvisual']['mean']:.3f} | {d['D_visual']['mean']:.3f}/{d['D_nonvisual']['mean']:.3f} | "
                     f"{(d['spearman_R_vs_D_token'] or float('nan')):.3f} | {top} |")
        L.append("")
    if answers:
        res = analyze(answers, a.n_boot)
        L += [f"## 품질 (context {res['n_contexts']}, 질문 {res['n_pairs']}, FULL EM {fmt(res['full_em'])}; FULL 참조 없는 질문 {res['missing_full']}개 제외)", ""]
        conds = res["conds"]
        rand_global = [c for c in conds if parse_cond(c)[0] == "random" and parse_cond(c)[2] == "global"]
        if stage == "deletion":
            ref = rand_global[0] if rand_global else None
            L += ["기준 열 '− random': 같은 질문에서 random(seed 0) 과의 짝지은 차이, context 단위 bootstrap 95% CI.", "",
                  "| 조건 (방법·방향·선택기) | EM [95% CI] | FULL-correct 보존 | ΔNLL vs FULL | EM − random | ΔNLL − random | 생성 길이 |",
                  "|---|---|---|---|---|---|---|"]
            for c in sorted(conds, key=lambda c: -(conds[c]["em"][0] if conds[c]["em"] else 0)):
                v = conds[c]
                L.append(f"| {c} | {fmt(v['em'])} | {fmt(v['retention'])} | {fmtd(v['delta_nll'])} | "
                         f"{fmtd(paired(res, c, ref, 'em', a.n_boot)) if ref else '—'} | "
                         f"{fmtd(paired(res, c, ref, 'delta_nll', a.n_boot)) if ref else '—'} | {v['gen_tokens']:.1f} |")
            L.append("")
            # 가설 방향 표: 신호별 keep_low(높은 점수 삭제) vs keep_high(낮은 점수 삭제) vs random
            L += ["### 삭제 민감도 (낮은 점수 삭제 = keep_high, 높은 점수 삭제 = keep_low)", "",
                  "| 신호 | 낮은 점수 삭제 EM | random EM | 높은 점수 삭제 EM | 낮은−random | 높은−random | 가설 방향? |", "|---|---|---|---|---|---|---|"]
            rnd = conds[ref]["em"][0] if ref else None
            for sig in sorted({parse_cond(c)[0] for c in conds} - {"random", "recent"}):
                hi = f"{sig}|keep_high|global|k{parse_cond(ref)[3]:g}|s0" if ref else None
                lo = f"{sig}|keep_low|global|k{parse_cond(ref)[3]:g}|s0" if ref else None
                if hi not in conds or lo not in conds:
                    continue
                e_hi, e_lo = conds[hi]["em"][0], conds[lo]["em"][0]
                ok = e_hi > rnd > e_lo
                L.append(f"| {sig} | {e_hi:.3f} | {rnd:.3f} | {e_lo:.3f} | {fmtd(paired(res, hi, ref, 'em', a.n_boot))} | "
                         f"{fmtd(paired(res, lo, ref, 'em', a.n_boot))} | {'예' if ok else '아니오'} |")
            L.append("")
        else:
            ks = sorted({parse_cond(c)[3] for c in conds}, reverse=True)
            ms = sorted({parse_cond(c)[0] for c in conds})
            for metric, name in (("em", "EM"), ("retention", "FULL-correct 보존"), ("delta_nll", "ΔNLL vs FULL")):
                L += [f"### {name} (행 = 방법, 열 = 유지율)", "", "| 방법 | " + " | ".join(f"{k:g}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
                for m in ms:
                    cells = []
                    for k in ks:
                        c = f"{m}|keep_high|global|k{k:g}|s0"
                        v = conds.get(c)
                        cells.append(fmt(v[metric]) if v and v[metric] else "—")
                    L.append(f"| {m} | " + " | ".join(cells) + " |")
                L.append("")
            if rand_global:
                L += ["### EM − random (같은 유지율, 짝지은 차이)", "", "| 방법 | " + " | ".join(f"{k:g}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
                for m in ms:
                    if m == "random":
                        continue
                    cells = []
                    for k in ks:
                        c, r = f"{m}|keep_high|global|k{k:g}|s0", f"random|keep_high|global|k{k:g}|s0"
                        cells.append(fmtd(paired(res, c, r, "em", a.n_boot)) if c in conds and r in conds else "—")
                    L.append(f"| {m} | " + " | ".join(cells) + " |")
                L.append("")
        if diags and any("spearman_vs_recon_desc" in d for d in diags):
            acc = defaultdict(list)
            for d in diags:
                for m, v in d.get("spearman_vs_recon_desc", {}).items():
                    if v is not None:
                        acc[m].append(v)
            L += ["### 보조 진단: 재구성(설명문) 점수와 평균 순위 Spearman (보호 쌍 제외)", "", "| 방법 | Spearman |", "|---|---|"]
            for m, v in sorted(acc.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
                L.append(f"| {m} | {sum(v)/len(v):.3f} |")
            L.append("")
    if builds:
        by = defaultdict(list)
        for b in builds:
            by[b["condition"]].append(b)
        L += ["## 저장량·비용 (build 기록 평균)", "", "| 조건 | 쌍 유지 비율 | KV bytes | metadata bytes | prefill s | score s | select s | prune s | peak GB |", "|---|---|---|---|---|---|---|---|---|"]
        for c, bs in sorted(by.items()):
            n = len(bs)
            g = lambda k: sum(b.get(k) or 0 for b in bs) / n
            L.append(f"| {c} | {g('keep_ratio_actual'):.3f} | {g('kv_bytes')/2**20:.1f} MB | {g('metadata_bytes')/2**10:.0f} KB | "
                     f"{g('prefill_seconds'):.2f} | {g('score_seconds'):.3f} | {g('select_seconds'):.3f} | {g('prune_seconds'):.3f} | {g('peak_bytes')/2**30:.2f} |")
        L.append("")
    text = "\n".join(L)
    out = os.path.join(ROOT, a.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out + ".md", "w").write(text); print(text); print(f"[saved] {out}.md")


if __name__ == "__main__":
    main()
