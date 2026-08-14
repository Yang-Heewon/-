"""주요 격차에 대한 문서(이미지) 단위 bootstrap 95% CI.

프로토콜 규칙(같은 이미지의 질문들을 독립 표본으로 부풀리지 않는다)에 따라
문서를 복원추출로 재표집하고, 그 문서들의 질문을 모아 통계량을 재계산한다.

  python -m vlm_diagnosis.scripts.bootstrap_ci
"""
import json
import glob
import os
import random
from collections import defaultdict

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


def boot_ci(per_doc_stats, stat_fn, n_boot=N_BOOT, seed=SEED):
    """per_doc_stats: {doc: 자료}. stat_fn(자료 목록)->float. 복원추출 CI."""
    docs = list(per_doc_stats)
    rng = random.Random(seed)
    point = stat_fn([per_doc_stats[d] for d in docs])
    vals = []
    for _ in range(n_boot):
        sample = [per_doc_stats[rng.choice(docs)] for _ in docs]
        v = stat_fn(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    lo, hi = vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]
    return point, lo, hi


def fmt(name, res):
    p, lo, hi = res
    print(f"  {name}: {p*100:+.1f}%p  [95% CI {lo*100:+.1f}, {hi*100:+.1f}]")


def main():
    main_recs = load("results/smoke/m2a_track1.shard*.jsonl")
    full_ok = {(r["sample_id"], r["question_id"])
               for r in main_recs if r["condition_id"] == "FULL_KV" and r["em"] == 1}

    # ---- 1) 20% 예산: selector 간 격차 (조건부 유지율) ----
    by_doc = defaultdict(lambda: defaultdict(list))
    for r in main_recs:
        if "selector" in r and abs(r["keep_ratio_target"] - 0.2) < 1e-9 \
                and (r["sample_id"], r["question_id"]) in full_ok:
            by_doc[r["sample_id"]][r["selector"]].append(r["em"])

    def gap(a, b):
        def f(sample):
            xa = [v for d in sample for v in d.get(a, [])]
            xb = [v for d in sample for v in d.get(b, [])]
            if not xa or not xb:
                return None
            return sum(xa) / len(xa) - sum(xb) / len(xb)
        return f

    print(f"[20% 예산, 조건부 유지율, 문서 {len(by_doc)}개 bootstrap]")
    fmt("s1 − s5   (미확정 몫: '최소' 격차)", boot_ci(by_doc, gap("s1", "s5")))
    fmt("s5 − random (대리 지표가 닫은 몫)", boot_ci(by_doc, gap("s5", "random")))
    fmt("s1 − random (전체 격차)", boot_ci(by_doc, gap("s1", "random")))
    fmt("random − knorm", boot_ci(by_doc, gap("random", "knorm")))

    # ---- 2) 교차 평가: 자기 재현 − 교차 (20%) ----
    cross = load("results/smoke/m3_pilot.shard*.jsonl")
    cx = defaultdict(lambda: {"self": [], "cross": []})
    for r in cross:
        if abs(r["budget_per_question"] - 0.2) > 1e-9 or r["subset_from"] == "UNION":
            continue
        if (r["sample_id"], r["eval_question_id"]) not in full_ok:
            continue
        cx[r["sample_id"]]["self" if r["is_self"] else "cross"].append(r["em"])

    def gap2(sample):
        s = [v for d in sample for v in d["self"]]
        c = [v for d in sample for v in d["cross"]]
        if not s or not c:
            return None
        return sum(s) / len(s) - sum(c) / len(c)

    print(f"\n[교차 평가 20%, 문서 {len(cx)}개 bootstrap]")
    fmt("자기 재현 − 교차 (재사용 붕괴)", boot_ci(cx, gap2))

    # ---- 3) held-out: UNION − random / s5 − UNION (질문당 5% → ~9% 크기) ----
    ho = load("results/smoke/m3_heldout.shard*.jsonl")
    fb = {(r["sample_id"], r["question_id"]): r["em"]
          for r in map(json.loads, open(os.path.join(
              ROOT, "results/smoke/m3_heldout_full_baseline.jsonl")))}
    hd = defaultdict(lambda: defaultdict(list))
    for r in ho:
        if abs(r["budget_per_question"] - 0.05) > 1e-9:
            continue
        if fb.get((r["sample_id"], r["eval_question_id"])) != 1:
            continue
        key = r["subset_from"] if r["subset_from"] in (
            "UNION", "RANDOM_MATCHED", "S5_MATCHED") else None
        if key:
            hd[r["sample_id"]][key].append(r["em"])

    print(f"\n[held-out ~9% 크기, FULL 정답 질문 조건부, 문서 {len(hd)}개 bootstrap]")
    fmt("UNION − random (과거 질문 3개의 초과 가치)",
        boot_ci(hd, gap("UNION", "RANDOM_MATCHED")))
    fmt("s5 − UNION (범용 신호 − 과거 질문 pool)",
        boot_ci(hd, gap("S5_MATCHED", "UNION")))


if __name__ == "__main__":
    main()
