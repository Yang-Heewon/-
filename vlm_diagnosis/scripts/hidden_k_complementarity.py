"""Stage 0C — 은닉 상태(초기/중기/후기) token 기하와 head 별 K 기하가 보완적인가.
late_mlp_semantic_probe 와 같은 20+20 장, 시각 token 만, prefill 1회.

hidden 묶음 H_g (early 1–9 / mid 10–18 / late 19–27): F_i = normalize(Σ_l normalize(x_l,i)) → PCA64
K 대상: (a) 후기 층 9 × head 4 = 36 개, (b) 전 층 표본 (층 0..27 중 7개 × head 4 = 28 개)
지표 (hidden 대 각 K, head 평균): 쌍별 코사인 Spearman, kNN@10 Jaccard, k-center 20% 선택 Jaccard,
  NovelCoverage = |S_H − S_K| / |S_H| (K 커버가 안 뽑은 token 을 hidden 커버가 뽑는 비율),
  그리고 반대 방향 |S_K − S_H| / |S_K|.
통과 기준(사용자): ρ < 0.8 이고 선택 Jaccard < 0.6 이며 hidden 의 구조(0A)가 K 와 비슷하거나 더 좋음.

  python -m vlm_diagnosis.scripts.hidden_k_complementarity --manifest experiments/manifests/gqa_discovery.jsonl --limit 20 --device cuda:0
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.mlp_dynamics import language_layers
from vlm_diagnosis.core.signals import vlm_inputs
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS
from vlm_diagnosis.scripts.late_mlp_semantic_probe import GROUPS, nrm, pca, kcenter, knn_sets, redundancy

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SAMPLE_LAYERS = (0, 4, 9, 13, 18, 22, 27)


class HiddenCapture:
    def __init__(self, model):
        self.layers = language_layers(model); self.x, self.h = {}, []
    def __enter__(self):
        for li, layer in enumerate(self.layers):
            self.h.append(layer.register_forward_pre_hook(lambda mod, a, kw, li=li: self.x.__setitem__(li, (a[0] if a else kw["hidden_states"])[0].half().cpu()), with_kwargs=True))
        return self
    def __exit__(self, *e):
        for h in self.h: h.remove()
        return False


def novel(zH, zK, frac=0.2):
    T = zH.shape[0]; n = int(frac * T)
    SH, SK = kcenter(zH, n), kcenter(zK, n)
    return len(SH - SK) / len(SH), len(SK - SH) / len(SK), len(SH & SK) / len(SH | SK)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/gqa_discovery.jsonl")
    ap.add_argument("--limit", type=int, default=20); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dom = os.path.basename(a.manifest).split("_")[0]
    out = a.out or f"results/context_only/hidden_k_{dom}_summary.md"
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))][:a.limit]
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    vis_mod = model.model.visual if hasattr(model.model, "visual") else None
    if vis_mod is not None and hasattr(vis_mod, "config"): vis_mod.config._attn_implementation = "sdpa"
    L, Hkv, _ = kv_dims(model)
    agg = defaultdict(list)
    for ii, row in enumerate(rows):
        img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
        ins = vlm_inputs(processor, img, "x", a.device)
        sp = token_spans(ins["input_ids"], model.config); P = int(sp["vis_end"]) + 2
        ids = ins["input_ids"][:, :P]; pos = mrope_position_ids(model, ids, ins["image_grid_thw"], torch.ones_like(ids))
        with HiddenCapture(model) as cap:
            o = model(input_ids=ids, position_ids=pos, attention_mask=torch.ones_like(ids), pixel_values=ins["pixel_values"],
                      image_grid_thw=ins["image_grid_thw"], use_cache=True)
        kv = [(k[0].float().cpu(), v[0].float().cpu()) for k, v in o.past_key_values.to_legacy_cache()]; del o
        vis = sp["visual"].cpu()
        zH = {g: pca(nrm(torch.stack([nrm(cap.x[l].float()[vis]) for l in lr]).sum(0)), 64) for g, lr in GROUPS.items()}
        K_late = [kv[l][0][h][vis] for l in GROUPS["late"] for h in range(Hkv)]
        K_all = [kv[l][0][h][vis] for l in SAMPLE_LAYERS for h in range(Hkv)]
        for g, z in zH.items():
            for tag, Ks in (("K_late", K_late), ("K_all", K_all)):
                r = redundancy(z, Ks)
                nov = [novel(z, k) for k in Ks]
                r["novel_H_minus_K"] = float(np.mean([n[0] for n in nov])); r["novel_K_minus_H"] = float(np.mean([n[1] for n in nov]))
                for k, v in r.items(): agg[(g, tag, k)].append(v)
        # hidden 묶음끼리 (early vs late) 도 참고
        r = redundancy(zH["early"], [zH["late"]])
        for k, v in r.items(): agg[("early", "hidden_late", k)].append(v)
        print(f"[{ii+1}/{len(rows)}] {row['sample_id']} Tv={vis.numel()} " + " ".join(
            f"{g}:ρ{agg[(g,'K_late','pair_spearman')][-1]:.2f}/J{agg[(g,'K_late','kcenter_jaccard')][-1]:.2f}/nov{agg[(g,'K_late','novel_H_minus_K')][-1]:.2f}" for g in GROUPS), flush=True)
        del cap
    M = [f"# Stage 0C — hidden ↔ K 보완성 ({a.manifest}, 이미지 {len(rows)}, 시각 token)", "",
         "| hidden 묶음 | K 대상 | 쌍별 코사인 Spearman | kNN@10 Jaccard | k-center 20% Jaccard | NovelCoverage |S_H−S_K|/|S_H| | |S_K−S_H|/|S_K| |", "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for tag in ("K_late", "K_all"):
            v = lambda k: np.mean(agg[(g, tag, k)])
            M.append(f"| {g} | {tag} | {v('pair_spearman'):.3f} | {v('knn_jaccard'):.3f} | {v('kcenter_jaccard'):.3f} | {v('novel_H_minus_K'):.3f} | {v('novel_K_minus_H'):.3f} |")
    v = lambda k: np.mean(agg[("early", "hidden_late", k)])
    M += [f"| early | hidden late (참고) | {v('pair_spearman'):.3f} | {v('knn_jaccard'):.3f} | {v('kcenter_jaccard'):.3f} | — | — |", "",
          "통과 기준: ρ < 0.8 이고 k-center Jaccard < 0.6 (그리고 0A 에서 hidden 의 구조 ≥ K)."]
    text = "\n".join(M); open(os.path.join(ROOT, out), "w").write(text); print(text)


if __name__ == "__main__":
    main()
