"""2a 단계 2·4 분석 — 배포형 신호의 판별력(AUROC)과 자기 보정 변형.

join 키: (sample_id, subset_from, eval_question_id, budget, eval_mode)
채점: FULL(이미지 그대로) 정답 표본 조건부. 반칙 coverage(있으면)와 비교.
"""
import argparse
import bisect
import json
import glob
import random
from collections import defaultdict


def load_em_map(base_pat):
    m = {}
    for f in glob.glob(base_pat):
        for l in open(f):
            r = json.loads(l)
            m[r["question_id"]] = r["em"]
    return m


def load_records(pat):
    out = []
    for f in glob.glob(pat):
        for l in open(f):
            r = json.loads(l)
            if r.get("record_type") == "run_metadata":
                continue
            out.append(r)
    return out


def auroc(pairs):
    pos = sorted(s for s, y in pairs if y == 1)
    neg = sorted(s for s, y in pairs if y == 0)
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        lo = bisect.bisect_left(neg, p); hi = bisect.bisect_right(neg, p)
        wins += lo; ties += hi - lo
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def boot_auroc(pairs, n=2000, seed=42):
    a = auroc(pairs)
    rng = random.Random(seed); vals = []
    for _ in range(n):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        v = auroc(s)
        if v is not None:
            vals.append(v)
    vals.sort()
    return a, vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def analyze(name, base_pat, sig_pat, em_pats, cov_pat=None):
    full = load_em_map(base_pat)
    sigs = {}
    refs = {}
    for r in load_records(sig_pat):
        key = (str(r["sample_id"]), r["subset_from"], r["eval_question_id"],
               r["budget_per_question"], r["eval_mode"])
        if r["eval_mode"] == "episode_ref":
            refs[(str(r["sample_id"]), r["subset_from"],
                  r["budget_per_question"])] = r
        else:
            sigs[key] = r
    cov = {}
    if cov_pat:
        for r in load_records(cov_pat):
            cov[(str(r["sample_id"]), r["subset_from"], r["eval_question_id"],
                 r["budget_per_question"], r["eval_mode"])] = r["coverage"]
    joined = []
    for mode, pat in em_pats:
        for r in load_records(pat):
            if full.get(r["eval_question_id"]) != 1:
                continue
            key = (str(r["sample_id"]), r["subset_from"], r["eval_question_id"],
                   r["budget_per_question"], mode)
            s = sigs.get(key)
            if not s:
                continue
            ref = refs.get((key[0], key[1], key[3]))
            row = dict(em=r["em"], **{k: s[k] for k in
                       ("a1_mass", "a2_entropy", "a3_sink", "a4_margin")})
            if ref:
                row["a1_selfcal"] = s["a1_mass"] / max(ref["a1_mass"], 1e-9)
                row["a4_selfcal"] = s["a4_margin"] / max(ref["a4_margin"], 1e-9)
            c = cov.get(key)
            if c is not None:
                row["cheat_coverage"] = c
            joined.append(row)
    print(f"\n=== {name} (join {len(joined)}건, "
          f"정답 {sum(r['em'] for r in joined)}) ===")
    signals = [("a1_mass", +1), ("a2_entropy", -1), ("a3_sink", -1),
               ("a4_margin", +1), ("a1_selfcal", +1), ("a4_selfcal", +1),
               ("cheat_coverage", +1)]
    for sig, sign in signals:
        pairs = [(sign * r[sig], r["em"]) for r in joined if sig in r]
        if len(pairs) < 50:
            continue
        a, lo, hi = boot_auroc(pairs)
        tag = " (반칙 상한)" if sig == "cheat_coverage" else ""
        print(f"  {sig:<15} AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]{tag}")
    return joined


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    analyze("Qwen2.5 × 자연(GQA)",
            'results/smoke/gqa_base.shard*.jsonl',
            'results/smoke/est_gqa_q25.jsonl',
            [("cross", 'results/smoke/gqa_cross.shard*.jsonl'),
             ("heldout", 'results/smoke/gqa_heldout.shard*.jsonl')],
            'results/smoke/gqa_coverage.jsonl')
    analyze("Qwen3 × 자연(GQA)",
            'results/smoke/q3_gqa_base.shard*.jsonl',
            'results/smoke/vast/est_gqa_q3.jsonl',
            [("cross", 'results/smoke/q3_gqa_cross.shard*.jsonl'),
             ("heldout", 'results/smoke/q3_gqa_heldout.shard*.jsonl')])
    analyze("Qwen3 × GUI(SQA)",
            'results/smoke/vast/q3_sqa_base.shard*.jsonl',
            'results/smoke/vast/est_sqa_q3.jsonl',
            [("cross", 'results/smoke/vast/q3_sqa_cross.shard*.jsonl'),
             ("heldout", 'results/smoke/vast/q3_sqa_heldout.shard*.jsonl')])
