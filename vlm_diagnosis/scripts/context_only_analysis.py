"""Context-only KV 결과 분석 — 엄격한 로그 검증, context 단위 paired bootstrap, 표 출력
(docs/CONTEXT-ONLY-KV-COMPRESSION.md §8).

  python -m vlm_diagnosis.scripts.context_only_analysis --pattern "results/context_only/deletion_qwen25vl_dev*.jsonl" --out results/context_only/deletion_dev_summary

지표: EM(전체), FULL-correct retention (FULL 정답 질문 중 조건도 정답), delta_NLL = 조건 NLL − FULL NLL, loyalty.
집계: context 별 질문 평균 → context 평균 (macro). bootstrap 은 context 재표집 (질문 묶음 유지).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FULL = "full|none|none|k1|s0"
SCHEMAS = {"context_only_v1", "context_only_v2"}
STAGES = {"full", "probe", "deletion", "sweep", "profile"}
_META_FIELDS = ("schema_version", "stage", "model", "model_id", "code_revision", "code_dirty",
                "transformers", "torch", "dtype", "attn_backend", "manifest", "manifest_sha256",
                "split", "dev_contexts", "n_contexts", "shard", "nshards", "keep_ratios", "methods",
                "random_seeds", "eps", "protect", "protected_special_ids", "granularity", "storage",
                "budget_rule", "tie_rule", "nll_rule", "questions", "brief", "max_new_tokens", "decode")
# Everything else is compared too, including newly added model/tokenizer revision,
# source fingerprints, precision/backend options and profiling methodology.
_RUN_LOCAL = {"record_type", "run_id", "started_at", "device", "n_contexts", "shard", "nshards",
              "expected_conditions", "expected_question_ids_by_context", "expected_context_ids"}


class LogError(Exception):
    pass


def _fail(message):
    raise LogError(message)


def _id(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        _fail(f"invalid {name}: {value!r}")
    return str(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"invalid {name}: {value!r}")
    return value


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        _fail(f"invalid {name}: {value!r}")
    return value


def _unique(values, name, allow_empty=False):
    if not isinstance(values, list) or (not values and not allow_empty):
        _fail(f"{name} must be a {'possibly empty ' if allow_empty else 'nonempty '}list")
    normalized = [_id(v, name) for v in values]
    if len(set(normalized)) != len(normalized):
        _fail(f"duplicate {name}")
    return normalized


def _metadata(r):
    _id(r.get("run_id"), "run_id")
    for key in _META_FIELDS:
        if key not in r:
            _fail(f"run metadata missing {key}")
    if r["schema_version"] not in SCHEMAS:
        _fail(f"unsupported schema_version {r['schema_version']!r}")
    if r["stage"] not in STAGES or r["split"] not in {"dev", "eval", "all"}:
        _fail("invalid stage/split in run metadata")
    for key in ("model", "model_id", "code_revision", "transformers", "torch", "dtype", "attn_backend",
                "manifest", "manifest_sha256", "protect", "granularity", "storage", "budget_rule", "tie_rule",
                "nll_rule", "questions", "decode"):
        if not isinstance(r[key], str) or not r[key].strip():
            _fail(f"run metadata {key} must be nonempty text")
    if not isinstance(r["brief"], str) or not isinstance(r["code_dirty"], bool):
        _fail("invalid brief/code_dirty metadata")
    for key, minimum in (("n_contexts", 1), ("nshards", 1), ("shard", 0), ("dev_contexts", 0), ("max_new_tokens", 1)):
        _integer(r[key], key, minimum)
    if r["shard"] >= r["nshards"]:
        _fail("shard must be smaller than nshards")
    if _number(r["eps"], "eps") == 0:
        _fail("eps must be positive")
    for key in ("methods", "random_seeds"):
        _unique(r[key], key)
    if not isinstance(r["keep_ratios"], list) or not r["keep_ratios"]:
        _fail("keep_ratios must be a nonempty list")
    for ratio in r["keep_ratios"]:
        if not 0 < _number(ratio, "keep ratio") <= 1:
            _fail("keep ratio outside (0,1]")
    if len(set(r["keep_ratios"])) != len(r["keep_ratios"]):
        _fail("duplicate keep_ratios")
    for seed in r["random_seeds"]:
        _integer(seed, "random seed")
    _unique(r["protected_special_ids"], "protected_special_ids", allow_empty=True)
    if r["schema_version"] == "context_only_v2":
        for key in ("model_revision", "tokenizer_revision", "implementation_sha256"):
            if not isinstance(r.get(key), str) or not r[key].strip():
                _fail(f"v2 metadata requires nonempty {key}")
        tolerances = r.get("parity_tolerances")
        if not isinstance(tolerances, dict) or not tolerances:
            _fail("v2 metadata requires parity_tolerances")
        for key, value in tolerances.items():
            _number(value, f"parity_tolerances.{key}")
        _integer(r.get("question_start"), "question_start")
        _integer(r.get("questions_per_context"), "questions_per_context", 1)
        if r["stage"] == "profile":
            _integer(r.get("profile_repeats"), "profile_repeats", 1)
            _integer(r.get("profile_warmup"), "profile_warmup")
            if r.get("profile_isolated") is not True:
                _fail("v2 profile must be isolated")
        conditions = _unique(r.get("expected_conditions"), "expected_conditions", allow_empty=r["stage"] == "probe")
        for condition in conditions:
            parse_cond(condition)
        qmap = r.get("expected_question_ids_by_context")
        if not isinstance(qmap, dict) or len(qmap) != r["n_contexts"]:
            _fail("expected_question_ids_by_context must enumerate all declared contexts")
        for cid, qids in qmap.items():
            _id(cid, "context_id")
            _unique(qids, "expected question IDs", allow_empty=r["stage"] == "probe")
        if r["stage"] in {"full", "deletion", "sweep"} and FULL not in conditions:
            _fail("expected_conditions is missing FULL")
        if r["stage"] == "probe" and conditions:
            _fail("probe must not declare answer conditions")
        if r["stage"] == "full" and conditions != [FULL]:
            _fail("full stage requires exactly the FULL condition")


def _signature(r):
    ignored = _RUN_LOCAL | ({"profile_method"} if r["stage"] == "profile" else set())
    return {key: value for key, value in r.items() if key not in ignored}


def _legacy_conditions(r, records):
    stage = r["stage"]
    if stage == "probe":
        return []
    if stage == "full":
        return [FULL]
    if stage == "profile":
        # v1 omitted profile_method from metadata. It is not recoverable beyond
        # the actual build condition, so this path is explicitly legacy/unverified.
        conditions = {x["condition"] for x in records if x["record_type"] == "build"}
        if len(conditions) != 1:
            _fail("legacy profile run must have exactly one build condition")
        return sorted(conditions)
    def cond(m, d, s, k, seed=0):
        return f"{m}|{d}|{s}|k{k:g}|s{seed}"
    result = [FULL]
    if stage == "sweep":
        result += [cond(m, "keep_high", "global", k) for k in r["keep_ratios"] for m in r["methods"]]
    else:
        k = max(r["keep_ratios"])
        result += [cond("random", "keep_high", "global", k, seed) for seed in r["random_seeds"]]
        result.append(cond("recent", "keep_high", "global", k))
        signals = ("mlp_norm", "r", "d", "k_norm", "v_norm", "r_std", "hidden_rel", "hidden_cos", "d_shuffle", "d_anchor")
        result += [cond(m, d, "global", k) for m in signals for d in ("keep_high", "keep_low")]
        result += [cond(m, "keep_high", "layer_matched", k) for m in ("mlp_norm", "r", "d", "random")]
        result += [cond(m, "keep_high", "boundary", k) for m in ("d", "random")]
    if len(result) != len(set(result)):
        _fail("duplicate conditions implied by legacy metadata")
    return result


def _legacy_questions(r, contexts):
    """v1 did not log question IDs: only recover from its hash-verified manifest."""
    path = os.path.join(ROOT, r["manifest"])
    try:
        with open(path, "rb") as handle:
            payload = handle.read()
    except OSError as exc:
        raise LogError("legacy completeness requires the original manifest") from exc
    if hashlib.sha256(payload).hexdigest() != r["manifest_sha256"]:
        _fail("legacy manifest SHA256 mismatch")
    match = re.fullmatch(r"manifest questions\[1:1\+(\d+)\] \+ BRIEF", r["questions"])
    if not match:
        _fail("unsupported legacy question selection rule")
    n = int(match[1])
    if r["stage"] == "full":
        n = min(n, 2)
    try:
        entries = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if r["split"] == "dev":
            entries = entries[:r["dev_contexts"]]
        elif r["split"] == "eval":
            entries = entries[r["dev_contexts"]:]
        entries = entries[r["shard"]::r["nshards"]]
        mapping = {}
        for item in entries:
            cid = _id(item["sample_id"], "manifest context_id")
            if cid in mapping:
                _fail("duplicate context ID in legacy manifest")
            mapping[cid] = [] if r["stage"] == "probe" else [
                _id(q["question_id"], "manifest question_id") for q in item["questions"][1:1+n]]
    except (KeyError, TypeError, ValueError) as exc:
        raise LogError("malformed legacy manifest") from exc
    if not contexts <= mapping.keys():
        _fail("record context outside declared legacy split/shard")
    if r["stage"] != "probe" and any(not mapping[cid] for cid in contexts):
        _fail("legacy context has no evaluation questions")
    return {cid: mapping[cid] for cid in contexts}


def _validate_answer(r):
    if r.get("status") != "ok":
        _fail("answer status is not ok")
    if _number(r.get("em"), "EM") not in (0, 1):
        _fail("EM must be 0 or 1")
    _integer(r.get("generated_tokens"), "generated_tokens")
    n = _integer(r.get("n_answer_tokens"), "n_answer_tokens")
    if r.get("nll") is None:
        if n != 0:
            _fail("null NLL must have zero answer tokens")
    else:
        _number(r["nll"], "NLL")
        if n < 1:
            _fail("NLL requires positive answer token count")
    if not isinstance(r.get("prediction"), str) or not isinstance(r.get("gold"), list) or not r["gold"]:
        _fail("answer requires prediction and gold")
    if r.get("loyalty") is not None and _number(r["loyalty"], "loyalty") not in (0, 1):
        _fail("loyalty must be 0 or 1")


def _complete(r, records):
    contexts = {x["context_id"] for x in records}
    if len(contexts) != r["n_contexts"]:
        _fail(f"incomplete run {r['run_id']}: expected {r['n_contexts']} contexts, got {len(contexts)}")
    conditions = r.get("expected_conditions") if r["schema_version"] == "context_only_v2" else _legacy_conditions(r, records)
    qmap = ({str(cid): [_id(q, "question_id") for q in qs] for cid, qs in r["expected_question_ids_by_context"].items()}
            if r["schema_version"] == "context_only_v2" else _legacy_questions(r, contexts))
    if contexts != set(qmap):
        _fail("records do not match declared context IDs")
    for cid in contexts:
        rows = [x for x in records if x["context_id"] == cid]
        by = defaultdict(list)
        for row in rows:
            by[row["record_type"]].append(row)
        if len(by["context_done"]) != 1 or rows[-1]["record_type"] != "context_done":
            _fail(f"incomplete context {cid}: missing/followed context_done")
        stage = r["stage"]
        expected_types = {"context_done", "diagnostic"} if stage == "probe" else {"context_done", "build"}
        expected_types |= {"parity"} if stage == "full" else ({"answer"} if stage != "probe" else set())
        if stage in {"deletion", "sweep"}:
            expected_types.add("diagnostic")
        if set(by) != expected_types:
            _fail(f"incomplete/invalid record types for {stage} context {cid}: {set(by)}")
        if "diagnostic" in expected_types and len(by["diagnostic"]) != 1:
            _fail(f"duplicate/missing diagnostic for {cid}")
        repeats = r.get("profile_repeats", 1) if stage == "profile" else 1
        expected_builds = {(c, i) for c in conditions for i in range(repeats)}
        actual_builds = {(x["condition"], x.get("repetition", 0)) for x in by["build"]}
        if actual_builds != expected_builds or len(by["build"]) != len(expected_builds):
            _fail(f"missing/extra build condition or repetition for {cid}")
        qids = set(qmap[cid])
        if stage == "full":
            if {x["question_id"] for x in by["parity"]} != qids or len(by["parity"]) != len(qids):
                _fail(f"incomplete parity questions for {cid}")
        elif stage != "probe":
            expected = {(c, q) for c in conditions for q in qids}
            actual = {(x["condition"], x["question_id"]) for x in by["answer"]}
            if actual != expected or len(by["answer"]) != len(expected):
                _fail(f"missing/extra condition × question answers for {cid}")
            for qid in qids:
                answers = [x for x in by["answer"] if x["question_id"] == qid]
                reference = answers[0]
                if any(x["gold"] != reference["gold"] or x["n_answer_tokens"] != reference["n_answer_tokens"]
                       or (x["nll"] is None) != (reference["nll"] is None) for x in answers):
                    _fail(f"incompatible answer targets/NLL availability for {cid}/{qid}")


def load(pattern):
    runs, builds, answers, diags, parity, errors = [], [], [], [], [], []
    seen, run_map, records = set(), {}, defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        with open(p, encoding="utf-8") as handle:
            lines = list(handle)
        for ln, line in enumerate(lines, 1):
            try:
                r = json.loads(line, parse_constant=lambda value: _fail(f"nonfinite JSON value {value}"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise LogError(f"malformed JSON at {p}:{ln}")
            if not isinstance(r, dict):
                _fail(f"record must be an object at {p}:{ln}")
            t = r.get("record_type")
            if t == "run":
                _metadata(r)
                if r["run_id"] in run_map:
                    _fail(f"duplicate run metadata {r['run_id']}")
                if runs and _signature(r) != _signature(runs[0]):
                    differing = sorted(k for k in _signature(r).keys() | _signature(runs[0]).keys()
                                       if _signature(r).get(k) != _signature(runs[0]).get(k))
                    _fail(f"incompatible run metadata: {', '.join(differing)}")
                run_map[r["run_id"]] = r
                runs.append(r)
                continue
            if t not in {"build", "answer", "diagnostic", "parity", "error", "context_done"}:
                _fail(f"unknown record_type {t!r} at {p}:{ln}")
            if r.get("run_id") not in run_map:
                _fail(f"orphan record with unknown run_id at {p}:{ln}")
            if t == "error":
                _fail(f"error record at {p}:{ln}: {r.get('error')}")
            r["context_id"] = _id(r.get("context_id"), "context_id")
            if t in {"answer", "parity"}:
                r["question_id"] = _id(r.get("question_id"), "question_id")
            if t in {"answer", "build"}:
                parse_cond(r.get("condition"))
            key = (t, r["context_id"], r.get("question_id"), r.get("condition"),
                   r["run_id"] if t == "context_done" else None,
                   r.get("repetition", 0) if t == "build" else None)
            if key in seen:
                _fail(f"duplicate {t} record at {p}:{ln}")
            seen.add(key)
            if t == "answer":
                _validate_answer(r)
            if t == "build" and run_map[r["run_id"]]["schema_version"] == "context_only_v2":
                isolated = run_map[r["run_id"]]["stage"] == "profile"
                if r.get("costs_valid_for_method") is not isolated:
                    _fail("build costs_valid_for_method disagrees with stage")
                if r.get("peak_scope") != ("context_build" if isolated else "shared_evaluation_harness"):
                    _fail("build peak_scope disagrees with stage")
                if isolated:
                    _integer(r.get("repetition"), "profile repetition")
            records[r["run_id"]].append(r)
            destination = {"build": builds, "answer": answers, "diagnostic": diags, "parity": parity}
            if t in destination:
                destination[t].append(r)
    if not runs:
        raise LogError("no run metadata")
    for run in runs:
        _complete(run, records[run["run_id"]])
    return runs, builds, answers, diags, parity, errors


def macro(per_ctx):
    vals = [sum(v) / len(v) for v in per_ctx.values() if v]
    return sum(vals) / len(vals) if vals else None


def boot(per_ctx, n_boot=5000, seed=42):
    _integer(n_boot, "n_boot", 1)
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
        _validate_answer(r)
        k = (r["context_id"], r["question_id"])
        if r["condition"] in per[k]:
            _fail("duplicate answer in quality analysis")
        if r["condition"] == FULL:
            full[k] = r
        per[k][r["condition"]] = r
    keys = [k for k in per if k in full]
    missing_full = [k for k in per if k not in full]
    if missing_full:
        _fail("quality analysis requires FULL for every question")
    conds = sorted({c for k in keys for c in per[k] if c != FULL})
    if any(set(per[k]) != set(conds) | {FULL} for k in keys):
        _fail("quality analysis requires a common complete condition × question set")
    out = {}
    for c in conds:
        em, ret, dnll, loy, gen = (defaultdict(list) for _ in range(5))
        for k in keys:
            r = per[k].get(c)
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
    try:
        m, d, s, k, seed = c.split("|")
        if not m or d not in {"none", "keep_high", "keep_low"} or s not in {"none", "global", "boundary", "layer_matched"}:
            raise ValueError
        if not k.startswith("k") or not seed.startswith("s"):
            raise ValueError
        ratio, selection_seed = float(k[1:]), int(seed[1:])
        if not math.isfinite(ratio) or not 0 < ratio <= 1 or selection_seed < 0:
            raise ValueError
        return m, d, s, ratio, selection_seed
    except (AttributeError, TypeError, ValueError) as exc:
        raise LogError(f"malformed condition {c!r}") from exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    if a.n_boot < 1:
        ap.error("n-boot must be positive")
    out = os.path.join(ROOT, a.out) + ".md"
    if os.path.exists(out):
        ap.error(f"output already exists: {out}; choose a new --out")
    runs, builds, answers, diags, parity, errors = load(a.pattern)
    stage = runs[0]["stage"]
    L = [f"# Context-only KV — {stage} — {a.pattern}", "",
         f"model {runs[0]['model']} · code {runs[0]['code_revision'][:8]}{' (dirty)' if runs[0]['code_dirty'] else ''} · "
         f"split {runs[0]['split']} · protect: {runs[0]['protect']} · NLL: {runs[0]['nll_rule']}", ""]
    legacy = runs[0]["schema_version"] == "context_only_v1"
    if legacy:
        L += ["Legacy v1: 효과 로그의 완료·질문·조건 정합성은 검증했지만 비용은 legacy/unverified입니다. "
              "기존 timing/peak 값을 방법별 비용 근거로 재사용하지 않습니다.", ""]
    if stage == "profile":
        L += ["Profile은 비용 전용 실행입니다. 독립 FULL 정답 기준이 없으므로 품질 비교표를 생성하지 않습니다.", ""]
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
    if answers and stage != "profile":
        res = analyze(answers, a.n_boot)
        L += [f"## 품질 (완료된 공통 context {res['n_contexts']}, 질문 {res['n_pairs']}, FULL EM {fmt(res['full_em'])})", ""]
        excluded_nll = sum(res["full"][k].get("nll") is None for k in res["keys"])
        if excluded_nll:
            L += [f"모든 조건에서 동일하게 NLL이 정의되지 않은 질문 {excluded_nll}개는 NLL 집계에서 제외합니다.", ""]
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
        valid_costs = not legacy and stage == "profile"
        if valid_costs:
            L += ["## 격리된 profile 저장량·비용 (build 반복 평균)", "",
                  "| 조건 | 쌍 유지 비율 | KV MiB | metadata KiB | build wall s | build peak over model GiB | resident over model MiB |",
                  "|---|---|---|---|---|---|---|"]
        else:
            L += ["## 논리 저장량 (build 기록 평균)", "",
                  "비용 열은 생략합니다: legacy/unverified 또는 FULL seed가 함께 존재하는 shared evaluation harness입니다.", "",
                  "| 조건 | 쌍 유지 비율 | KV MiB | metadata KiB |", "|---|---|---|---|"]
        for c, bs in sorted(by.items()):
            n = len(bs)
            g = lambda k: sum(b.get(k) or 0 for b in bs) / n
            line = f"| {c} | {g('keep_ratio_actual'):.3f} | {g('kv_bytes')/2**20:.1f} | {g('metadata_bytes')/2**10:.0f} |"
            if valid_costs:
                line += f" {g('build_wall_seconds'):.3f} | {g('build_peak_bytes_over_model')/2**30:.3f} | {g('resident_bytes_over_model_after_build')/2**20:.2f} |"
            L.append(line)
        L.append("")
    text = "\n".join(L)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x", encoding="utf-8") as handle:
        handle.write(text)
    print(text); print(f"[saved] {out}")


if __name__ == "__main__":
    main()
