"""Stage 0 진단 — 후기 MLP 직교 갱신(m⊥)이 (A) 의미 구조를 갖는가, (B) K 공간과 다른 정보를 갖는가.
압축은 하지 않는다. prefill 1회, 시각 token 만 분석.

특징 공간 (층마다 token 벡터, 모두 L2 정규화):
  hidden      x_l (층 입력 잔차)                          — 대조: 그냥 은닉 상태
  m_hat       m_l / ‖m_l‖                                — MLP 출력 방향
  m_perp      m_l − (m_l·r̂_l) r̂_l  정규화                — 잔차와 직교하는 새 방향
  m_perp_ds   m_perp 에서 sink 방향(sink token 들의 m 의 주성분)을 빼고 정규화
층 묶음: early 1–9, mid 10–18, late 19–27, adaptive (effective-rank 골짜기 다음 층부터 27).
묶음 특징 F_i = normalize(Σ_l feat_l,i) → PCA 64 → z_i.

(A) 의미성: k-means 16 군집의 공간 응집도(격자 이웃 같은 군집 비율 / 무작위 기대), 물체 상자 안/밖 코사인,
    물체 kNN recall@10, 위치 누출(10-NN 중 격자 거리 ≤ 2 비율).
(B) K 공간과의 중복 (late m_perp_ds vs 후기 층 각 head 의 K): 쌍별 코사인 Spearman, kNN@10 Jaccard,
    k-center(20%) 선택 집합 Jaccard. 셋이 모두 높으면 (ρ>0.9, J>0.8, J>0.8) 중복으로 판정.

  python -m vlm_diagnosis.scripts.late_mlp_semantic_probe --manifest experiments/manifests/gqa_discovery.jsonl --limit 20 --device cuda:0
"""
import argparse, json, math, os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.mlp_dynamics import language_layers
from vlm_diagnosis.core.signals import vlm_inputs
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.static_pair_select import massive_activation_positions
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
GROUPS = {"early": range(1, 10), "mid": range(10, 19), "late": range(19, 28)}


class RawCapture:
    """층마다 x(입력), r(MLP 직전), m(MLP 출력) 을 fp16 CPU 로 보관 (진단 전용)."""
    def __init__(self, model):
        self.layers = language_layers(model); self.x, self.r, self.m, self.h = {}, {}, {}, []
    def __enter__(self):
        for li, layer in enumerate(self.layers):
            self.h.append(layer.register_forward_pre_hook(lambda mod, a, kw, li=li: self.x.__setitem__(li, (a[0] if a else kw["hidden_states"])[0].half().cpu()), with_kwargs=True))
            self.h.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda mod, a, li=li: self.r.__setitem__(li, a[0][0].half().cpu())))
            self.h.append(layer.mlp.register_forward_hook(lambda mod, a, o, li=li: self.m.__setitem__(li, o[0].half().cpu())))
        return self
    def __exit__(self, *e):
        for h in self.h: h.remove()
        return False


def nrm(x, eps=1e-6):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def erank(X):
    Xc = X.float() - X.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc); p = s / s.sum(); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def pca(z, k=64):
    z = z.float() - z.float().mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(z, q=min(k, z.shape[1], z.shape[0] - 1), center=False)
    return z @ V


def kmeans(z, k=16, iters=25, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = z[torch.randperm(z.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        lab = torch.cdist(z, c).argmin(1)
        for j in range(k):
            if (lab == j).any(): c[j] = z[lab == j].mean(0)
    return lab


def kcenter(z, n, start=0):
    T = z.shape[0]; zn = nrm(z); sel = [start]; mind = 1 - zn @ zn[start]
    for _ in range(n - 1):
        mind[sel] = -1; nxt = int(mind.argmax()); sel.append(nxt); mind = torch.minimum(mind, 1 - zn @ zn[nxt])
    return set(sel)


def knn_sets(z, k=10):
    S = nrm(z) @ nrm(z).T; S.fill_diagonal_(-2)
    return S.topk(k, dim=1).indices


def spatial_tests(z, rows, cols, boxes_tok, k=10, n_clusters=16):
    T = z.shape[0]; out = {}
    lab = kmeans(z, n_clusters)
    # 격자 이웃 (상하좌우) 같은 군집 비율 / 무작위 라벨 기대
    idx = torch.arange(T); adj = []
    for i in range(T):
        r, c = int(rows[i]), int(cols[i])
        for dr, dc in ((0, 1), (1, 0)):
            j = ((rows == r + dr) & (cols == c + dc)).nonzero()
            if j.numel(): adj.append((i, int(j[0])))
    adj = torch.tensor(adj)
    same = float((lab[adj[:, 0]] == lab[adj[:, 1]]).float().mean())
    perm = lab[torch.randperm(T)]
    base = float((perm[adj[:, 0]] == perm[adj[:, 1]]).float().mean())
    out["spatial_coherence_ratio"] = same / max(base, 1e-6)
    nn = knn_sets(z, k)
    d = (rows[nn] - rows[:, None]).abs() + (cols[nn] - cols[:, None]).abs()
    out["knn_grid_dist_le2"] = float((d <= 2).float().mean())
    if boxes_tok:
        obj = torch.full((T,), -1, dtype=torch.long)
        for oi, toks in enumerate(boxes_tok):
            obj[toks] = oi
        inobj = obj >= 0
        if int(inobj.sum()) >= 4 and len(boxes_tok) >= 2:
            zn = nrm(z[inobj]); S = zn @ zn.T; o = obj[inobj]
            samem = (o[:, None] == o[None, :]) & ~torch.eye(len(o), dtype=torch.bool)
            diffm = o[:, None] != o[None, :]
            out["cos_same_object"] = float(S[samem].mean()); out["cos_diff_object"] = float(S[diffm].mean())
            # 물체 kNN recall: 물체 token 의 k 이웃(전체 token 중) 가운데 같은 물체 비율
            nn_all = knn_sets(z, k)
            rec = [(obj[nn_all[i]] == obj[i]).float().mean() for i in range(T) if obj[i] >= 0]
            out["object_knn_recall"] = float(torch.stack(rec).mean())
            # 무작위 기대: 같은 물체 token 비율
            out["object_knn_recall_chance"] = float(torch.stack([(obj == obj[i]).float().sum() / T for i in range(T) if obj[i] >= 0]).mean())
    return out, lab


def redundancy(zF, zK_list, frac=0.2, k=10, n_pairs=5000, seed=0):
    T = zF.shape[0]; g = torch.Generator().manual_seed(seed)
    ii = torch.randint(0, T, (n_pairs,), generator=g); jj = torch.randint(0, T, (n_pairs,), generator=g)
    SF = (nrm(zF)[ii] * nrm(zF)[jj]).sum(1).numpy()
    nnF = knn_sets(zF, k); selF = kcenter(zF, int(frac * T))
    rho, jk, js = [], [], []
    for zK in zK_list:
        SK = (nrm(zK)[ii] * nrm(zK)[jj]).sum(1).numpy()
        rho.append(spearmanr(SF, SK).correlation)
        nnK = knn_sets(zK, k)
        jk.append(float(np.mean([len(set(nnF[i].tolist()) & set(nnK[i].tolist())) / len(set(nnF[i].tolist()) | set(nnK[i].tolist())) for i in range(T)])))
        selK = kcenter(zK, int(frac * T)); js.append(len(selF & selK) / len(selF | selK))
    return {"pair_spearman": float(np.mean(rho)), "knn_jaccard": float(np.mean(jk)), "kcenter_jaccard": float(np.mean(js))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/gqa_discovery.jsonl")
    ap.add_argument("--limit", type=int, default=20); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="qwen25vl"); ap.add_argument("--out", default=None); ap.add_argument("--n-vis", type=int, default=4)
    a = ap.parse_args()
    dom = os.path.basename(a.manifest).split("_")[0]
    out = a.out or f"results/context_only/late_mlp_{dom}"
    os.makedirs(os.path.join(ROOT, out), exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))][:a.limit]
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    vis_mod = model.model.visual if hasattr(model.model, "visual") else None
    if vis_mod is not None and hasattr(vis_mod, "config"): vis_mod.config._attn_implementation = "sdpa"
    L, Hkv, hd = kv_dims(model)
    agg = defaultdict(list); erank_curves = []; red_all = defaultdict(list); valleys = []
    for ii, row in enumerate(rows):
        img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
        ins = vlm_inputs(processor, img, "x", a.device)
        sp = token_spans(ins["input_ids"], model.config); P = int(sp["vis_end"]) + 2
        ids = ins["input_ids"][:, :P]; pos = mrope_position_ids(model, ids, ins["image_grid_thw"], torch.ones_like(ids))
        with RawCapture(model) as cap:
            o = model(input_ids=ids, position_ids=pos, attention_mask=torch.ones_like(ids), pixel_values=ins["pixel_values"],
                      image_grid_thw=ins["image_grid_thw"], use_cache=True)
        kv = [(k[0].float().cpu(), v[0].float().cpu()) for k, v in o.past_key_values.to_legacy_cache()]; del o
        vis = sp["visual"].cpu(); Tv = vis.numel()
        t, gh, gw = [int(x) for x in ins["image_grid_thw"][0]]; mh, mw = gh // 2, gw // 2
        assert mh * mw == Tv, (mh, mw, Tv)
        rows_g = torch.arange(Tv) // mw; cols_g = torch.arange(Tv) % mw
        # 물체 상자 → token 집합 (원본 좌표 → 병합 격자)
        sx, sy = (gw * 14) / row["image_width"], (gh * 14) / row["image_height"]
        boxes = {}
        for q in row["questions"]:
            b = q.get("evidence_bbox")
            if b and q.get("evidence_object_ids"):
                boxes[q["evidence_object_ids"][0]] = b
        # token → 그 token 을 포함하는 가장 작은 상자 (겹치는 상자는 작은 쪽 우선), 상자당 token ≥ 3 인 것만
        cx, cy = cols_g + 0.5, rows_g + 0.5
        owner = torch.full((Tv,), -1, dtype=torch.long); area = torch.full((Tv,), float("inf"))
        blist = list(boxes.values())
        for bi, b in enumerate(blist):
            x0, y0, x1, y1 = b[0] * sx / 28, b[1] * sy / 28, b[2] * sx / 28, b[3] * sy / 28
            inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
            ar = max(1e-6, (x1 - x0) * (y1 - y0))
            take = inside & (ar < area)
            owner[take] = bi; area[take] = ar
        boxes_tok = [torch.nonzero(owner == bi).flatten() for bi in range(len(blist)) if int((owner == bi).sum()) >= 3]
        # sink 방향: 앞 4 token ∪ MLP 스파이크 token
        R = torch.stack([cap.m[l].float().norm(dim=-1) / (cap.r[l].float().norm(dim=-1) + 1e-6) for l in range(L)])
        sink = torch.zeros(P, dtype=torch.bool); sink[:4] = True; sink |= massive_activation_positions(R, 10.0)
        feats = {k: {} for k in ("hidden", "m_hat", "m_perp", "m_perp_ds")}
        er = []
        for l in range(L):
            x, r, m = cap.x[l].float(), cap.r[l].float(), cap.m[l].float()
            er.append(erank(x[vis]))
            rh = nrm(r); mp = m - (m * rh).sum(-1, keepdim=True) * rh
            s = torch.linalg.svd(m[sink] - m[sink].mean(0, keepdim=True), full_matrices=False).Vh[0]
            mpd = mp - (mp @ s)[:, None] * s
            feats["hidden"][l] = nrm(x[vis]); feats["m_hat"][l] = nrm(m[vis]); feats["m_perp"][l] = nrm(mp[vis]); feats["m_perp_ds"][l] = nrm(mpd[vis])
        erank_curves.append(er)
        valley = int(np.argmin(er[1:27])) + 1; valleys.append(valley)
        groups = dict(GROUPS); groups["adaptive"] = range(valley + 1, L)
        zs = {}
        for fname, per_l in feats.items():
            for gname, lr in groups.items():
                F = nrm(torch.stack([per_l[l] for l in lr]).sum(0)); z = pca(F, 64); zs[(fname, gname)] = z
                res, lab = spatial_tests(z, rows_g, cols_g, boxes_tok)
                for k, v in res.items(): agg[(fname, gname, k)].append(v)
                if fname == "m_perp_ds" and gname == "late" and ii < a.n_vis:
                    try:
                        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        ax[0].imshow(img.resize((gw * 14, gh * 14))); ax[0].set_title(f"{row['sample_id']}"); ax[0].axis("off")
                        ax[1].imshow(lab.view(mh, mw).numpy(), cmap="tab20", interpolation="nearest"); ax[1].set_title("late m_perp_ds: k-means 16"); ax[1].axis("off")
                        for b in boxes.values():
                            ax[1].add_patch(plt.Rectangle((b[0] * sx / 28 - .5, b[1] * sy / 28 - .5), (b[2] - b[0]) * sx / 28, (b[3] - b[1]) * sy / 28, fill=False, color="red", lw=1.5))
                        fig.tight_layout(); fig.savefig(os.path.join(ROOT, out, f"{row['sample_id']}_clusters.png"), dpi=110); plt.close(fig)
                    except Exception as e: print("fig err", e)
        # (B) 중복: late m_perp_ds vs 후기 층 head 별 K (시각 token), 그리고 vs hidden late
        zF = zs[("m_perp_ds", "late")]
        Ks = [kv[l][0][h][vis] for l in GROUPS["late"] for h in range(Hkv)]
        red = redundancy(zF, Ks); red_h = redundancy(zF, [zs[("hidden", "late")]])
        for k, v in red.items(): red_all[("vsK_late", k)].append(v)
        for k, v in red_h.items(): red_all[("vsHidden_late", k)].append(v)
        # K 자체가 위치·물체 구조를 얼마나 갖는지 (대조)
        zK = torch.cat([nrm(k) for k in Ks], dim=1); zK = pca(zK, 64)
        resK, _ = spatial_tests(zK, rows_g, cols_g, boxes_tok)
        for k, v in resK.items(): agg[("K_late", "late", k)].append(v)
        print(f"[{ii+1}/{len(rows)}] {row['sample_id']} Tv={Tv} grid={mh}x{mw} objs={len(boxes_tok)} valley={valley} "
              f"coh(late m_perp_ds)={agg[('m_perp_ds','late','spatial_coherence_ratio')][-1]:.2f} redK={red}", flush=True)
        del cap
    # ---- 요약
    M = ["# Stage 0 — 후기 MLP 직교 갱신의 의미성(A)과 K 공간과의 중복(B)", "",
         f"manifest {a.manifest}, 이미지 {len(rows)}, 시각 token 만. effective-rank 골짜기 층(평균) = {np.mean(valleys):.1f} (개별: {valleys})", "",
         "## (A) 의미성 — 값은 이미지 평균", "",
         "| 특징 | 층 묶음 | 공간 응집도 (무작위=1) | 10-NN 중 격자거리≤2 (위치 누출) | 같은 물체 cos | 다른 물체 cos | 물체 kNN recall@10 | 우연 수준 |", "|---|---|---|---|---|---|---|---|"]
    def g(f, gr, k):
        v = agg.get((f, gr, k)); return f"{np.mean(v):.3f}" if v else "—"
    for f in ("hidden", "m_hat", "m_perp", "m_perp_ds"):
        for gr in ("early", "mid", "late", "adaptive"):
            M.append(f"| {f} | {gr} | {g(f,gr,'spatial_coherence_ratio')} | {g(f,gr,'knn_grid_dist_le2')} | {g(f,gr,'cos_same_object')} | {g(f,gr,'cos_diff_object')} | {g(f,gr,'object_knn_recall')} | {g(f,gr,'object_knn_recall_chance')} |")
    M.append(f"| K (late, head 연결) | late | {g('K_late','late','spatial_coherence_ratio')} | {g('K_late','late','knn_grid_dist_le2')} | {g('K_late','late','cos_same_object')} | {g('K_late','late','cos_diff_object')} | {g('K_late','late','object_knn_recall')} | {g('K_late','late','object_knn_recall_chance')} |")
    M += ["", "## (B) late m_perp_ds 와 다른 공간의 중복 (이미지 평균; 중단 기준: 쌍별 ρ>0.9 & kNN J>0.8 & k-center J>0.8)", "",
          "| 비교 대상 | 쌍별 코사인 Spearman | kNN@10 Jaccard | k-center 20% 선택 Jaccard |", "|---|---|---|---|"]
    for tgt in ("vsK_late", "vsHidden_late"):
        M.append(f"| {tgt} | {np.mean(red_all[(tgt,'pair_spearman')]):.3f} | {np.mean(red_all[(tgt,'knn_jaccard')]):.3f} | {np.mean(red_all[(tgt,'kcenter_jaccard')]):.3f} |")
    ec = np.mean(erank_curves, 0)
    M += ["", "## 층별 effective rank (시각 token 은닉 상태, 이미지 평균)", "", " ".join(f"{l}:{v:.0f}" for l, v in enumerate(ec)), ""]
    text = "\n".join(M); open(os.path.join(ROOT, out, "summary.md"), "w").write(text); print(text)
    json.dump({"agg": {"|".join(k): v for k, v in agg.items()}, "red": {"|".join(k): v for k, v in red_all.items()}, "erank": erank_curves, "valleys": valleys},
              open(os.path.join(ROOT, out, "raw.json"), "w"))


if __name__ == "__main__":
    main()
