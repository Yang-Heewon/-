"""M2-A Track 2 — answer-aware probe (02_M2A §4): 용량 상한 진단.

절차 (진단 전용 — 배포 불가, gold answer는 '선택'에만 쓰고 '생성'에는 안 넣음):
  1. 시각 토큰을 2×2 공간 패치 그룹으로 묶는다 (M2A-03).
  2. leave-group-out: 그룹 하나를 가리고 gold answer의 teacher-forced logp 하락을
     잰다 → 그룹 중요도. (그룹 수만큼 forward — M2A-04 상한 기록)
  3. 중요도 순으로 예산 k토큰을 채워 subset을 만든다.
  4. gold 없이 생성해 공식 metric(ANLS/EM)으로 채점한다.

한 질문당 ranking은 1회이고 모든 예산이 그 순위의 앞부분을 공유한다.

실행:
  python -m vlm_diagnosis.exps.m2a_answer_probe --limit 3 --device cuda:0
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_qwen25vl
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.masked_eval import (
    causal_mask_4d, evict_columns, mrope_position_ids)
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
from vlm_diagnosis.core.metrics import anls, exact_match
from vlm_diagnosis.core.kv_baselines import KVShape, dense_storage, max_keep_for_budget
from vlm_diagnosis.core import signals as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAX_PIXELS = 1280 * 28 * 28
BRIEF = " Answer with a single word or phrase."
N_LAYERS, N_KV_HEADS, HEAD_DIM = 28, 4, 128


def patch_groups(grid_thw, n_vis, patch=2):
    """merged grid(R×C) 위 patch×patch 그룹 → [ [시각토큰 순번, ...], ... ]"""
    t, h, w = (int(x) for x in grid_thw[0])
    R, C = h // 2, w // 2                       # 2×2 merge 후 토큰 격자
    assert R * C * t == n_vis, f"grid {R}x{C}x{t} != n_vis {n_vis}"
    groups = {}
    for i in range(n_vis):
        r, c = (i % (R * C)) // C, i % C
        groups.setdefault((r // patch, c // patch), []).append(i)
    return list(groups.values())


@torch.no_grad()
def _merged_embeds(model, full_ids, pixel_values, grid_thw, vis):
    """vision tower를 1회만 실행해 text+visual 결합 embedding을 만든다."""
    emb = model.get_input_embeddings()(full_ids).clone()      # (1,L,H)
    tower = model.model.visual if hasattr(model.model, "visual") else model.visual
    vis_emb = tower(pixel_values.to(model.dtype), grid_thw=grid_thw)
    emb[0, vis] = vis_emb.to(emb.dtype)
    return emb


@torch.no_grad()
def _batched_answer_logp(model, emb, masks, pos, labels):
    """같은 embedding에 서로 다른 4D mask를 배치로 걸어 gold logp를 병렬 계산.
    logits은 정답 구간(+1)만 유지해 메모리를 줄인다."""
    bs = masks.shape[0]
    n_ans = labels.shape[0]
    out = model.model(
        inputs_embeds=emb.expand(bs, -1, -1),
        attention_mask=masks,
        position_ids=pos.expand(3, bs, -1),
        use_cache=False)
    hidden = out.last_hidden_state[:, -(n_ans + 1):-1]        # 정답 예측 위치
    logits = model.lm_head(hidden).float()
    logp = torch.log_softmax(logits, -1)
    tok = logp[:, torch.arange(n_ans), labels]                # (bs, n_ans)
    return tok.sum(-1)


@torch.no_grad()
def rank_groups(model, ctx, groups, batch=8):
    """leave-group-out Δlogp 랭킹 (배치·embedding 재사용). 중요도 내림차순."""
    emb, pos, m4 = ctx["emb"], ctx["pos"], ctx["m4"]
    labels = ctx["labels"]
    lp_full = float(_batched_answer_logp(model, emb, m4, pos, labels)[0])
    drops, n_fwd = [], 1
    for s in range(0, len(groups), batch):
        chunk = groups[s:s + batch]
        masks = torch.cat([
            evict_columns(m4, torch.tensor(
                [int(ctx["vis"][o]) for o in g], device=ctx["dev"]),
                ctx["vis_end"] + 1)
            for g in chunk], dim=0)
        lps = _batched_answer_logp(model, emb, masks, pos, labels)
        drops.extend((lp_full - lps).tolist())
        n_fwd += 1
    order = sorted(range(len(groups)), key=lambda i: -drops[i])
    return [groups[i] for i in order], [drops[i] for i in order], n_fwd


def fill_budget(ranked_groups, k):
    keep = []
    for g in ranked_groups:
        if len(keep) >= k:
            break
        keep.extend(g[:max(0, k - len(keep))])
    return set(keep)


def _group_mask(ctx, groups, kept_group_ids):
    kept = {o for gi in kept_group_ids for o in groups[gi]}
    cols = torch.tensor(
        [int(ctx["vis"][o]) for o in range(len(ctx["vis"])) if o not in kept],
        device=ctx["dev"])
    return evict_columns(ctx["m4"], cols, ctx["vis_end"] + 1)


@torch.no_grad()
def swap_search(model, ctx, groups, k_tokens, s1_scores,
                batch=8, passes=2, max_forwards=200):
    """s1 상위로 초기화한 뒤, (약한 in-그룹 ↔ out-그룹) 교환을 gold logp로 수락하는
    hill-climb. 낱개 하락값이 아니라 '집합 전체의 logp'만 본다 (비가산성 회피)."""
    gm = [sum(float(s1_scores[o]) for o in g) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: -gm[i])
    in_set, tok = set(), 0
    for gi in order:
        if tok >= k_tokens:
            break
        in_set.add(gi)
        tok += len(groups[gi])
    init_set = set(in_set)
    cur_lp = float(_batched_answer_logp(
        model, ctx["emb"], _group_mask(ctx, groups, in_set),
        ctx["pos"], ctx["labels"])[0])
    init_lp, n_fwd = cur_lp, 1
    out_list = [i for i in range(len(groups)) if i not in in_set]
    for _ in range(passes):
        improved = False
        for s in range(0, len(out_list), batch):
            if n_fwd >= max_forwards:
                break
            weakest = min(in_set, key=lambda i: gm[i])
            cands = [c for c in out_list[s:s + batch] if c not in in_set]
            if not cands:
                continue
            masks = torch.cat([
                _group_mask(ctx, groups, (in_set - {weakest}) | {c})
                for c in cands])
            lps = _batched_answer_logp(model, ctx["emb"], masks,
                                       ctx["pos"], ctx["labels"])
            n_fwd += 1
            best = int(torch.argmax(lps))
            if float(lps[best]) > cur_lp:
                in_set.remove(weakest)
                in_set.add(cands[best])
                cur_lp = float(lps[best])
                improved = True
        if not improved or n_fwd >= max_forwards:
            break
    return init_set, init_lp, in_set, cur_lp, n_fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/m2a_diagnostic.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budgets", default="0.05,0.1,0.2,0.4")
    ap.add_argument("--search", choices=["lgo", "swap"], default="lgo",
                    help="lgo: 한 그룹씩 가려보기(비가산성 취약) / "
                         "swap: s1 초기화 + 집합 단위 교환 hill-climb")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-forwards-per-question", type=int, default=400)
    ap.add_argument("--out", default="results/smoke/m2a_answer_probe.jsonl")
    a = ap.parse_args()

    budgets = [float(x) for x in a.budgets.split(",")]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))]
    rows = rows[a.shard::a.nshards][:a.limit]
    if a.nshards > 1:
        a.out = a.out.replace(".jsonl", f".shard{a.shard}.jsonl")
    model, processor = load_qwen25vl(device=a.device, max_pixels=MAX_PIXELS)
    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    run_id = f"m2a-probe-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    with open(out_path, "w") as f:
        f.write(json.dumps({
            "record_type": "run_metadata", "schema_version": "1.1",
            "run_id": run_id, "stage": "M2A", "run_kind": "smoke",
            "condition": "target_answer_aware_probe (diagnostic only)",
            "search": "leave_group_out, spatial_2x2",
            "budgets_keep": budgets,
            "started_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False) + "\n")
        for di, row in enumerate(rows):
            img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
            q = row["questions"][1]              # track1과 같은 첫 평가 질문
            q_text = q["question"] + BRIEF
            golds = q["answers"]
            t0 = time.time()
            ins = S.vlm_inputs(processor, img, q_text, a.device)
            ans_ids = processor.tokenizer(golds[0], add_special_tokens=False,
                                          return_tensors="pt").input_ids.to(a.device)
            full = torch.cat([ins["input_ids"], ans_ids], 1)
            sp = token_spans(full, model.config)
            vis, vis_end, L = sp["visual"], sp["vis_end"], sp["L"]
            P = ins["input_ids"].shape[1]
            attn2d = torch.ones(1, L, dtype=torch.long, device=a.device)
            pos = mrope_position_ids(model, full, ins["image_grid_thw"], attn2d)
            n_vis = len(vis)
            groups = patch_groups(ins["image_grid_thw"].cpu(), n_vis)
            if 1 + len(groups) > a.max_forwards_per_question:
                print(f"skip {row['sample_id']}: groups {len(groups)} > cap")
                continue
            ctx = {"m4": causal_mask_4d(L, a.device), "pos": pos, "vis": vis,
                   "vis_end": vis_end, "dev": a.device,
                   "emb": _merged_embeds(model, full, ins["pixel_values"],
                                         ins["image_grid_thw"], vis),
                   "labels": full[0, P:]}
            shape = KVShape(layers=N_LAYERS, batch=1, kv_heads=N_KV_HEADS,
                            tokens=n_vis, head_dim=HEAD_DIM)
            e = dense_storage(shape)
            full_bytes = e.payload_bytes + e.metadata_bytes + e.position_bytes

            if a.search == "lgo":
                ranked, drops, n_fwd = rank_groups(model, ctx, groups)
            else:
                s1_scores = S.score_s1(model, processor, img, q_text,
                                       a.device).cpu()

            for B in budgets:
                k = max_keep_for_budget(shape, int(B * full_bytes), "sparse")
                if a.search == "lgo":
                    variants = [("lgo", fill_budget(ranked, k), None, n_fwd,
                                 [round(d, 2) for d in drops[:5]])]
                else:
                    ini, ini_lp, fin, fin_lp, nf = swap_search(
                        model, ctx, groups, k, s1_scores)
                    n_fwd = nf
                    variants = [
                        ("s1_init", {o for gi in ini for o in groups[gi]},
                         ini_lp, nf, None),
                        ("swap_final", {o for gi in fin for o in groups[gi]},
                         fin_lp, nf, None)]
                for tag, keep, gold_lp, nf_used, top_drops in variants:
                    evict = torch.tensor(
                        [int(vis[o]) for o in range(n_vis) if o not in keep],
                        device=a.device)
                    pred = greedy_generate_masked(
                        model, processor, ins, max_new_tokens=a.max_new_tokens,
                        evict_cols=evict, row_start=vis_end + 1)
                    f.write(json.dumps({
                        "run_id": run_id, "dataset": row["dataset"],
                        "split": "smoke", "sample_id": row["sample_id"],
                        "question_id": q["question_id"], "gold": golds,
                        "condition_id": f"probe_{tag}@B{int(B*100)}",
                        "selection_timing": "diagnostic",
                        "search": a.search,
                        "keep_ratio_target": B, "keep_tokens": len(keep),
                        "n_visual": n_vis, "n_groups": len(groups),
                        "search_forwards": nf_used,
                        "gold_logp_of_subset": gold_lp,
                        "top_group_drops": top_drops,
                        "prediction": pred,
                        "anls": anls(pred, golds),
                        "em": exact_match(pred, golds)},
                        ensure_ascii=False) + "\n")
                    f.flush()
            print(f"[{di+1}/{len(rows)}] {row['sample_id']} groups={len(groups)} "
                  f"fwd={n_fwd} {time.time()-t0:.0f}s", flush=True)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
