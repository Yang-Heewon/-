"""ScreenQA 교차평가 결과의 화면 단위 bootstrap 95% CI.

프로토콜 규칙(같은 화면의 질문은 독립 표본이 아님)에 따라 화면을 복원추출한다.
주장별 통계량:
  1) 재사용 붕괴: 자기 재현 − 교차 (B=5%, 20%)
  2) 근거 이동 축: 교차 쌍에서 거리-유지율 Spearman 상관 (B=5%)
  3) 원거리(>=25%) vs 근거리(<10%) 유지율 차이 (B=5%)

  python -m vlm_diagnosis.scripts.bootstrap_ci_screenqa
"""
import json
import glob
import os
import random
from collections import defaultdict

from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
N_BOOT = 10000
SEED = 42


def load(pat):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, pat))):
        for l in open(p):
            r = json.loads(l)
            if r.get("record_type") != "run_metadata":
                out.append(r)
    return out


def boot(per_screen, fn, n=N_BOOT, seed=SEED):
    keys = list(per_screen)
    rng = random.Random(seed)
    pt = fn([per_screen[k] for k in keys])
    vals = []
    for _ in range(n):
        v = fn([per_screen[rng.choice(keys)] for _ in keys])
        if v is not None:
            vals.append(v)
    vals.sort()
    return pt, vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def main():
    mf = {r["sample_id"]: r for r in map(
        json.loads, open(os.path.join(ROOT, "experiments/manifests/screenqa_transfer.jsonl")))}
    pairs = {(r["sample_id"], r["qA_id"], r["qB_id"]): r for r in map(
        json.loads, open(os.path.join(ROOT, "experiments/manifests/screenqa_transfer_pairs.jsonl")))}
    base = {(str(r["sample_id"]), r["question_id"]): r["em"]
            for r in load("results/smoke/screenqa_base.shard*.jsonl")}
    ok = {k for k, v in base.items() if v == 1}

    per = defaultdict(lambda: {"self": defaultdict(list), "cross": defaultdict(list),
                               "dist": defaultdict(list)})
    for r in load("results/smoke/sqa_cross.shard*.jsonl"):
        if r["subset_from"] == "UNION":
            continue
        sid = str(r["sample_id"])
        if (sid, r["eval_question_id"]) not in ok:
            continue
        B = round(r["budget_per_question"], 2)
        if r["is_self"]:
            per[sid]["self"][B].append(r["em"])
        else:
            per[sid]["cross"][B].append(r["em"])
            qs = mf[sid]["questions"][1:4]
            src = qs[int(r["subset_from"][3:])]["question_id"]
            pr = pairs.get((sid, src, r["eval_question_id"]))
            if pr:
                per[sid]["dist"][B].append((pr["center_distance"], r["em"]))

    def gap(B):
        def f(sample):
            s = [x for d in sample for x in d["self"][B]]
            c = [x for d in sample for x in d["cross"][B]]
            if not s or not c:
                return None
            return sum(s) / len(s) - sum(c) / len(c)
        return f

    def far_near(B):
        def f(sample):
            near = [em for d in sample for dd, em in d["dist"][B] if dd < 0.10]
            far = [em for d in sample for dd, em in d["dist"][B] if dd >= 0.25]
            if not near or not far:
                return None
            return sum(near) / len(near) - sum(far) / len(far)
        return f

    def rho(B):
        def f(sample):
            pts = [t for d in sample for t in d["dist"][B]]
            if len(pts) < 30:
                return None
            return float(spearmanr([p[0] for p in pts], [p[1] for p in pts]).statistic)
        return f

    print(f"[ScreenQA, 화면 {len(per)}개 cluster bootstrap, {N_BOOT}회]")
    for B in (0.05, 0.2):
        p, lo, hi = boot(per, gap(B))
        print(f"  재사용 붕괴 (self−cross) @B={B:.0%}: {p*100:+.1f}%p  [95% CI {lo*100:+.1f}, {hi*100:+.1f}]")
    for B in (0.05, 0.2):
        p, lo, hi = boot(per, far_near(B))
        print(f"  근거리(<10%)−원거리(>=25%) 유지율 @B={B:.0%}: {p*100:+.1f}%p  [CI {lo*100:+.1f}, {hi*100:+.1f}]")
    p, lo, hi = boot(per, rho(0.05))
    print(f"  거리↔유지율 Spearman ρ @B=5%: {p:+.3f}  [CI {lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
