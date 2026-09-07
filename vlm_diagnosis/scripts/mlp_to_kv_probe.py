"""MLP → 다음 층 K/V 투영 효과 진단 (압축 없음, 진단 전용).

질문: 층 l 의 MLP 갱신 m_l 이 다음 층 l+1 의 head 별 K/V 를 얼마나 바꾸는가, 그 변화량이
 (1) 미래 질문의 attention (평가 라벨, 점수 계산에 사용 안 함), (2) 재구성 중요도, (3) 다음 층 attention 변화와 관계있는가.

한 층짜리 반사실: x_cf_{l+1} = r_l (attention 까지는 그대로, MLP 갱신만 제거). 실제 input_layernorm_{l+1} 과 k_proj/v_proj 로
K_cf, V_cf 를 만들어 실제 K, V 와 비교. 점수는 (KV 층 L=l+1, KV head h, token i) 마다:
  dK_norm, dK_rel, dK_dir(1−cos), dV_norm, dV_rel, dV_dir.  선형 대조: PK = W_K m_l, PV = W_V m_l (norm 무시).
검산: (a) x_{l+1} ≈ r_l + m_l, (b) hook 의 k_proj/v_proj 출력 = k_proj(norm(x_{l+1})), (c) 캐시의 RoPE 후 K = rope(hook K).
Attention 재계산 (KV 층 L): 조건1 Q 실제 고정 + K_cf(RoPE 후) → ΔA; 조건2 Q_cf, K_cf 모두 반사실 → ΔA.
RoPE 는 위치별 회전이라 ‖ΔK‖·cos 는 RoPE 전후가 같다 (검산으로 확인). 점수는 RoPE 전, attention 은 RoPE 후.

  python -m vlm_diagnosis.scripts.mlp_to_kv_probe --manifest experiments/manifests/gqa_discovery.jsonl --limit 20 --device cuda:0
"""
import argparse, json, math, os, time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _q25

from vlm_diagnosis.core.loader import load_vlm, kv_dims
from vlm_diagnosis.core.masked_eval import mrope_position_ids
from vlm_diagnosis.core.mlp_dynamics import language_layers
from vlm_diagnosis.core.signals import vlm_inputs
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.core.attnstat import QKCapture
from vlm_diagnosis.core.kv_select import per_head_column_stats
from vlm_diagnosis.core.static_pair_select import protected_positions, massive_activation_positions
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS, BRIEF
from vlm_diagnosis.exps.core_delta_dram import REPEAT_PROMPT

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCORES = ["mlp_norm", "R", "D", "hidden_rel", "hidden_cos", "knorm", "vnorm", "attn1",
          "dK_norm", "dK_rel", "dK_dir", "dV_norm", "dV_rel", "dV_dir", "PK_norm", "PV_norm", "dM_cond1", "dM_cond2"]
NEW = ["dK_norm", "dK_rel", "dK_dir", "dV_norm", "dV_rel", "dV_dir"]


class FullCapture:
    """x_l, r_l, m_l, x_{l+1}, k_proj/v_proj 출력(RoPE 전), RoPE 인자(cos, sin, section), RoPE 후 q/k 를 기록."""
    def __init__(self, model):
        self.layers = language_layers(model); self.h = []
        self.x, self.r, self.m, self.xn, self.kpre, self.vpre, self.qrot, self.krot = ({} for _ in range(8))
        self.rope = None
    def __enter__(self):
        for li, layer in enumerate(self.layers):
            self.h.append(layer.register_forward_pre_hook(lambda mod, a, kw, li=li: self.x.__setitem__(li, (a[0] if a else kw["hidden_states"])[0]), with_kwargs=True))
            self.h.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda mod, a, li=li: self.r.__setitem__(li, a[0][0])))
            self.h.append(layer.mlp.register_forward_hook(lambda mod, a, o, li=li: self.m.__setitem__(li, o[0])))
            self.h.append(layer.register_forward_hook(lambda mod, a, o, li=li: self.xn.__setitem__(li, (o[0] if isinstance(o, (tuple, list)) else o)[0])))
            self.h.append(layer.self_attn.k_proj.register_forward_hook(lambda mod, a, o, li=li: self.kpre.__setitem__(li, o[0])))
            self.h.append(layer.self_attn.v_proj.register_forward_hook(lambda mod, a, o, li=li: self.vpre.__setitem__(li, o[0])))
        self._orig = _q25.apply_multimodal_rotary_pos_emb
        cap = self
        def wrapped(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
            qr, kr = cap._orig(q, k, cos, sin, mrope_section, unsqueeze_dim)
            li = len(cap.qrot); cap.qrot[li] = qr[0].detach(); cap.krot[li] = kr[0].detach()
            if cap.rope is None: cap.rope = (cos, sin, list(mrope_section), unsqueeze_dim)
            return qr, kr
        _q25.apply_multimodal_rotary_pos_emb = wrapped
        return self
    def __exit__(self, *e):
        for h in self.h: h.remove()
        _q25.apply_multimodal_rotary_pos_emb = self._orig
        return False


def sp(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 8 or a.std() == 0 or b.std() == 0: return np.nan
    return float(spearmanr(a, b).correlation)


def rope_k(cap, k_bhd):
    """k (Hkv, T, d) → RoPE 후 (Hkv, T, d), 실제 함수·실제 cos/sin 사용."""
    cos, sin, sec, ud = cap.rope
    dummy = k_bhd[None]
    _, kr = cap._orig(dummy, k_bhd[None], cos, sin, sec, ud)
    return kr[0]


@torch.no_grad()
def attention_received(q, k, scaling, group, rows=None):
    """q (Hq,T,d) k (Hkv,T,d) → causal softmax → 열마다 받은 attention 질량 (Hkv, T): 행 합, 그룹 내 q head 평균. 전체 행렬 (Hq,T,T) 도 반환."""
    Hq, T, d = q.shape
    kk = k.repeat_interleave(group, dim=0)
    w = (q.float() * scaling) @ kk.float().transpose(-1, -2)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
    w.masked_fill_(mask[None], float("-inf"))
    A = w.softmax(-1)
    recv = A.sum(1).view(-1, group, T).mean(1)
    return A, recv


@torch.no_grad()
def collect(model, processor, row, device, L, Hkv, hd):
    img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
    ins = vlm_inputs(processor, img, "x", device)
    sp_ = token_spans(ins["input_ids"], model.config); P = int(sp_["vis_end"]) + 2
    ids = ins["input_ids"][:, :P]; pos = mrope_position_ids(model, ids, ins["image_grid_thw"], torch.ones_like(ids))
    layers = language_layers(model)
    with FullCapture(model) as cap:
        out = model(input_ids=ids, position_ids=pos, attention_mask=torch.ones_like(ids), pixel_values=ins["pixel_values"],
                    image_grid_thw=ins["image_grid_thw"], use_cache=True)
    cache_k = [k[0] for k, _ in out.past_key_values.to_legacy_cache()]   # (Hkv, P, d) RoPE 후
    del out
    vis = sp_["visual"].cpu(); vis_mask = torch.zeros(P, dtype=torch.bool); vis_mask[vis] = True
    Hq = cap.qrot[0].shape[0]; group = Hq // Hkv; scaling = layers[0].self_attn.scaling
    checks = defaultdict(list)
    S = {k: torch.full((L, Hkv, P), float("nan")) for k in SCORES}
    lin_cos = torch.full((L, Hkv, P), float("nan"))
    dA_stats = defaultdict(list)
    # 기존 통계 (층 l 기준) → KV 층 l+1 에 정렬
    R_all = torch.stack([cap.m[l].float().norm(dim=-1) / (cap.r[l].float().norm(dim=-1) + 1e-6) for l in range(L)]).cpu()
    for l in range(L - 1):
        Ln = l + 1
        x_real = cap.xn[l].float(); r = cap.r[l].float(); m = cap.m[l].float(); x_in = cap.x[l].float()
        # (a) x_{l+1} ≈ r + m ; 다음 층 입력과 동일 여부
        checks["resid_rel_err"].append(float(((x_real - (r + m)).abs().amax(-1) / (x_real.abs().amax(-1) + 1e-6)).max()))
        checks["next_input_eq"].append(float((x_real - cap.x[Ln].float()).abs().max()))
        nxt = layers[Ln]; attn = nxt.self_attn
        z_real = nxt.input_layernorm(x_real.to(cap.x[Ln].dtype)).float()
        K_rec = (z_real @ attn.k_proj.weight.float().T + attn.k_proj.bias.float()).view(P, Hkv, hd).transpose(0, 1)
        V_rec = (z_real @ attn.v_proj.weight.float().T + attn.v_proj.bias.float()).view(P, Hkv, hd).transpose(0, 1)
        K_hook = cap.kpre[Ln].float().view(P, Hkv, hd).transpose(0, 1); V_hook = cap.vpre[Ln].float().view(P, Hkv, hd).transpose(0, 1)
        # (b) 재현 검산
        e = (K_rec - K_hook).abs()
        checks["K_rec_max_abs"].append(float(e.max())); checks["K_rec_mean_abs"].append(float(e.mean()))
        checks["K_rec_rel"].append(float(e.norm() / (K_hook.norm() + 1e-6)))
        checks["K_rec_cos"].append(float(torch.nn.functional.cosine_similarity(K_rec.flatten(1), K_hook.flatten(1), dim=-1).mean()))
        # (c) RoPE 검산: rope(hook K) vs 캐시 K
        K_rot = rope_k(cap, K_hook.to(cap.x[Ln].dtype)).float()
        checks["rope_rel"].append(float((K_rot - cache_k[Ln].float()).norm() / (cache_k[Ln].float().norm() + 1e-6)))
        # 반사실
        x_cf = r
        z_cf = nxt.input_layernorm(x_cf.to(cap.x[Ln].dtype)).float()
        K_cf = (z_cf @ attn.k_proj.weight.float().T + attn.k_proj.bias.float()).view(P, Hkv, hd).transpose(0, 1)
        V_cf = (z_cf @ attn.v_proj.weight.float().T + attn.v_proj.bias.float()).view(P, Hkv, hd).transpose(0, 1)
        dK, dV = K_hook - K_cf, V_hook - V_cf
        S["dK_norm"][Ln] = dK.norm(dim=-1).cpu(); S["dK_rel"][Ln] = (dK.norm(dim=-1) / (K_hook.norm(dim=-1) + 1e-6)).cpu()
        S["dK_dir"][Ln] = (1 - torch.nn.functional.cosine_similarity(K_hook, K_cf, dim=-1)).cpu()
        S["dV_norm"][Ln] = dV.norm(dim=-1).cpu(); S["dV_rel"][Ln] = (dV.norm(dim=-1) / (V_hook.norm(dim=-1) + 1e-6)).cpu()
        S["dV_dir"][Ln] = (1 - torch.nn.functional.cosine_similarity(V_hook, V_cf, dim=-1)).cpu()
        # RoPE 불변 검산: ‖ΔK‖ RoPE 후
        K_cf_rot = rope_k(cap, K_cf.to(cap.x[Ln].dtype)).float()
        checks["dK_rope_invariance"].append(float(((K_rot - K_cf_rot).norm(dim=-1) - dK.norm(dim=-1)).abs().max() / (dK.norm(dim=-1).max() + 1e-6)))
        # 선형 대조: W_K m, W_V m (bias 는 차이에서 상쇄)
        PK = (m @ attn.k_proj.weight.float().T).view(P, Hkv, hd).transpose(0, 1); PV = (m @ attn.v_proj.weight.float().T).view(P, Hkv, hd).transpose(0, 1)
        S["PK_norm"][Ln] = PK.norm(dim=-1).cpu(); S["PV_norm"][Ln] = PV.norm(dim=-1).cpu()
        lin_cos[Ln] = torch.nn.functional.cosine_similarity(PK, dK, dim=-1).cpu()
        # 기존 통계 정렬 (층 l 의 MLP → KV 층 l+1)
        S["mlp_norm"][Ln] = m.norm(dim=-1).cpu()[None].expand(Hkv, P)
        S["R"][Ln] = R_all[l][None].expand(Hkv, P)
        S["D"][Ln] = ((R_all[l] - R_all[l - 1]).abs() if l >= 1 else torch.zeros(P))[None].expand(Hkv, P)
        S["hidden_rel"][Ln] = ((x_real - x_in).norm(dim=-1) / (x_in.norm(dim=-1) + 1e-6)).cpu()[None].expand(Hkv, P)
        S["hidden_cos"][Ln] = (1 - torch.nn.functional.cosine_similarity(x_in, x_real, dim=-1)).cpu()[None].expand(Hkv, P)
        S["knorm"][Ln] = K_hook.norm(dim=-1).cpu(); S["vnorm"][Ln] = V_hook.norm(dim=-1).cpu()
        # attention 재계산: 조건1 (Q 실제, K 반사실), 조건2 (Q, K 모두 반사실)
        q_real = cap.qrot[Ln]
        A_real, recv_real = attention_received(q_real, cache_k[Ln], scaling, group)
        A_c1, recv_c1 = attention_received(q_real, K_cf_rot, scaling, group)
        Q_cf = (z_cf @ attn.q_proj.weight.float().T + attn.q_proj.bias.float()).view(P, Hq, hd).transpose(0, 1)
        cos, sin, sec, ud = cap.rope
        Q_cf_rot, _ = cap._orig(Q_cf[None].to(cap.x[Ln].dtype), K_cf[None].to(cap.x[Ln].dtype), cos, sin, sec, ud)
        A_c2, recv_c2 = attention_received(Q_cf_rot[0], K_cf_rot, scaling, group)
        S["dM_cond1"][Ln] = (recv_real - recv_c1).abs().cpu(); S["dM_cond2"][Ln] = (recv_real - recv_c2).abs().cpu()
        dA_stats["cond1_mean_abs"].append(float((A_real - A_c1).abs().mean())); dA_stats["cond1_row_tv"].append(float(0.5 * (A_real - A_c1).abs().sum(-1).mean()))
        dA_stats["cond2_mean_abs"].append(float((A_real - A_c2).abs().mean())); dA_stats["cond2_row_tv"].append(float(0.5 * (A_real - A_c2).abs().sum(-1).mean()))
        # attn1 (같은 prefill, 시각 행이 받은 attention, KV 층 Ln)
        S["attn1"][Ln] = recv_real.cpu() / max(1, P)
        if l == 0:
            fig6 = {"layer": None}
        del A_real, A_c1, A_c2
    # 그림 6 용: 조건1 ΔA 가 가장 큰 층의 받은 질량 프로파일 (head 0)
    best = int(np.argmax(dA_stats["cond1_row_tv"])) + 1
    fig6 = {"layer": best, "recv_real": None, "recv_cf": None}
    q_real = cap.qrot[best]; nxt = layers[best]; attn = nxt.self_attn
    z_cf = nxt.input_layernorm(cap.r[best - 1]).float()
    K_cf = (z_cf @ attn.k_proj.weight.float().T + attn.k_proj.bias.float()).view(P, Hkv, hd).transpose(0, 1)
    _, rr = attention_received(q_real, cache_k[best], scaling, group); _, rc = attention_received(q_real, rope_k(cap, K_cf.to(cap.x[best].dtype)).float(), scaling, group)
    fig6["recv_real"] = rr[0].cpu().tolist(); fig6["recv_cf"] = rc[0].cpu().tolist()
    # ---- 라벨 (점수 계산에 쓰지 않음): 재구성, 미래 질문 attention
    ins1 = vlm_inputs(processor, img, REPEAT_PROMPT, device)
    gen = model.generate(**{k: v for k, v in ins1.items()}, max_new_tokens=96, do_sample=False)
    P1, Lg = ins1["input_ids"].shape[1], gen.shape[1]
    am = torch.ones(1, Lg, dtype=torch.long, device=device); pg = mrope_position_ids(model, gen, ins1["image_grid_thw"], am)
    with QKCapture() as qc:
        model(input_ids=gen, attention_mask=am, position_ids=pg, pixel_values=ins1["pixel_values"], image_grid_thw=ins1["image_grid_thw"], use_cache=False)
        _, recon = per_head_column_stats(qc.qk, P1, Lg)
    recon = recon[:, :, :P]; del qc
    Aq = []
    for q in row["questions"][1:4]:
        insq = vlm_inputs(processor, img, q["question"] + BRIEF, device); Lq = insq["input_ids"].shape[1]
        assert token_spans(insq["input_ids"], model.config)["vis_end"] + 2 == P
        pq = mrope_position_ids(model, insq["input_ids"], insq["image_grid_thw"], torch.ones_like(insq["input_ids"]))
        with QKCapture() as qc:
            model(input_ids=insq["input_ids"], attention_mask=torch.ones_like(insq["input_ids"]), position_ids=pq, pixel_values=insq["pixel_values"], image_grid_thw=insq["image_grid_thw"], use_cache=False)
            mq, _ = per_head_column_stats(qc.qk, P, Lq)
        Aq.append(mq[:, :, :P]); del qc
    Aq = torch.stack(Aq)
    prot = protected_positions(ids.cpu(), set(int(s) for s in processor.tokenizer.all_special_ids) - {int(model.config.image_token_id)}, 4)
    spike = massive_activation_positions(R_all, 10.0)
    return {"P": P, "vis": vis_mask, "prot": prot, "spike": spike, "S": {k: v.half() for k, v in S.items()}, "lin_cos": lin_cos.half(),
            "recon": recon.half(), "A_mean": Aq.mean(0).half(), "A_max": Aq.amax(0).half(), "checks": dict(checks), "dA": dict(dA_stats), "fig6": fig6,
            "sample_id": row["sample_id"]}


def analyze(data, L, Hkv, out_dir, dom):
    labels = {"A_mean": "future-query attention (mean of 3)", "A_max": "future-query attention (max of 3)", "recon": "reconstruction importance"}
    # per (layer, head) correlations, averaged over images; global pooled per image then averaged
    lh = {(s, lab): np.full((L, Hkv), np.nan) for s in SCORES for lab in labels}
    cnt = {(s, lab): np.zeros((L, Hkv)) for s in SCORES for lab in labels}
    glob = defaultdict(list); glob_prot = defaultdict(list); glob_nospike = defaultdict(list)
    for d in data:
        m = d["vis"] & ~d["prot"]; m_all = d["vis"] | d["prot"]; m_ns = m & ~d["spike"]
        for s in SCORES:
            X = d["S"][s].float()
            for lab, key in (("A_mean", "A_mean"), ("A_max", "A_max"), ("recon", "recon")):
                Y = d[key].float()
                for l in range(1, L):
                    for h in range(Hkv):
                        v = sp(X[l, h][m], Y[l, h][m])
                        if not np.isnan(v):
                            lh[(s, lab)][l, h] = np.nansum([lh[(s, lab)][l, h], v]) if not np.isnan(lh[(s, lab)][l, h]) else v; cnt[(s, lab)][l, h] += 1
                glob[(s, lab)].append(sp(X[1:][:, :, m], Y[1:][:, :, m]))
                glob_prot[(s, lab)].append(sp(X[1:][:, :, m_all], Y[1:][:, :, m_all]))
                glob_nospike[(s, lab)].append(sp(X[1:][:, :, m_ns], Y[1:][:, :, m_ns]))
    for k in lh:
        lh[k] = lh[k] / np.maximum(cnt[k], 1)
    # score-score matrix (global, visual non-protected, layers 1..)
    ss = np.zeros((len(SCORES), len(SCORES)))
    for d in data:
        m = d["vis"] & ~d["prot"]
        flat = {s: d["S"][s].float()[1:][:, :, m].numpy().ravel() for s in SCORES}
        idx = np.random.RandomState(0).choice(flat["dK_rel"].size, min(20000, flat["dK_rel"].size), replace=False)
        for i, a in enumerate(SCORES):
            for j, b in enumerate(SCORES):
                ss[i, j] += sp(flat[a][idx], flat[b][idx]) / len(data)
    lin = float(np.nanmean([d["lin_cos"].float()[1:][:, :, d["vis"] & ~d["prot"]].mean() for d in data]))
    checks = {k: float(np.max([np.max(d["checks"][k]) for d in data])) for k in data[0]["checks"]}
    dA = {k: [float(np.mean([d["dA"][k][l] for d in data])) for l in range(L - 1)] for k in data[0]["dA"]}
    G = lambda dic, s, lab: float(np.nanmean(dic[(s, lab)]))
    M = [f"# MLP → 다음 층 K/V 투영 효과 진단 — {dom} (이미지 {len(data)}, 시각 token, 보호 token 제외)", "",
         "## 검산 (이미지·층 최대)", "", "| 항목 | 값 |", "|---|---|"]
    for k, v in checks.items(): M.append(f"| {k} | {v:.3e} |")
    M += ["", f"선형 대조 cos(W_K m, ΔK) 평균 = {lin:.3f} (1 이면 ΔK 가 RMSNorm 상호작용 없이 W_K m 로 설명됨)", "",
          "## 전역 Spearman (이미지별 pooled → 평균; 층 1..27, 시각·비보호 token)", "",
          "| 점수 | vs 미래 질문 attention (평균) | (최대) | vs 재구성 | 보호 token 포함 시 (평균) | 스파이크 제외 시 (평균) |", "|---|---|---|---|---|---|"]
    for s in SCORES:
        M.append(f"| {s} | {G(glob,s,'A_mean'):+.3f} | {G(glob,s,'A_max'):+.3f} | {G(glob,s,'recon'):+.3f} | {G(glob_prot,s,'A_mean'):+.3f} | {G(glob_nospike,s,'A_mean'):+.3f} |")
    M += ["", "## 층별 (head 평균) Spearman vs 미래 질문 attention(평균) — 층 1..27", ""]
    for s in ("mlp_norm", "R", "D", "dK_rel", "dK_dir", "dV_rel", "dV_dir", "dM_cond1", "knorm", "vnorm", "attn1"):
        M.append(f"- {s}: " + " ".join(f"{np.nanmean(lh[(s,'A_mean')][l]):+.2f}" for l in range(1, L)))
    M += ["", "## 층별 (head 평균) Spearman vs 재구성 — 층 1..27", ""]
    for s in ("mlp_norm", "dK_rel", "dK_dir", "dV_rel", "dV_dir", "dM_cond1"):
        M.append(f"- {s}: " + " ".join(f"{np.nanmean(lh[(s,'recon')][l]):+.2f}" for l in range(1, L)))
    M += ["", "## 점수끼리 Spearman (전역)", "", "| | " + " | ".join(SCORES) + " |", "|---|" + "---|" * len(SCORES)]
    for i, a in enumerate(SCORES): M.append(f"| {a} | " + " | ".join(f"{ss[i,j]:+.2f}" for j in range(len(SCORES))) + " |")
    M += ["", "## 다음 층 attention 변화 (층 1..27 평균; 행 total variation = 0.5·Σ|ΔA|)", "",
          "- 조건1 (Q 실제 고정, K 반사실) 행 TV: " + " ".join(f"{v:.3f}" for v in dA["cond1_row_tv"]),
          "- 조건2 (Q·K 모두 반사실) 행 TV: " + " ".join(f"{v:.3f}" for v in dA["cond2_row_tv"]), ""]
    open(os.path.join(out_dir, f"{dom}_summary.md"), "w").write("\n".join(M)); print("\n".join(M))
    json.dump({"global": {f"{s}|{lab}": G(glob, s, lab) for s in SCORES for lab in labels}, "layer_head": {f"{s}|{lab}": lh[(s, lab)].tolist() for s in SCORES for lab in labels},
               "score_score": ss.tolist(), "score_names": SCORES, "checks": checks, "lin_cos": lin, "dA": dA},
              open(os.path.join(out_dir, f"{dom}_projected_effect_stats.json"), "w"))
    # ---- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for s, name in (("dK_rel", "layer_head_corr_attention_dK"), ("dV_rel", "layer_head_corr_attention_dV")):
        fig, ax = plt.subplots(figsize=(4, 8)); im = ax.imshow(lh[(s, "A_mean")][1:], aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=0.5)
        ax.set_title(f"{dom}: Spearman({s}, future-query attn)"); ax.set_xlabel("KV head"); ax.set_ylabel("KV layer (1..27)"); fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{dom}_{name}.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    keys = ["mlp_norm", "R", "D", "hidden_cos", "knorm", "vnorm", "attn1", "dK_norm", "dK_rel", "dK_dir", "dV_norm", "dV_rel", "dV_dir", "dM_cond1"]
    for a, lab, ttl in ((ax[0], "A_mean", "vs future-query attention (mean)"), (ax[1], "recon", "vs reconstruction importance")):
        a.bar(range(len(keys)), [G(glob, s, lab) for s in keys]); a.set_xticks(range(len(keys))); a.set_xticklabels(keys, rotation=60, ha="right"); a.axhline(0, color="k", lw=.5); a.set_title(f"{dom}: global Spearman {ttl}")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{dom}_global_corr_bars.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for s in ("mlp_norm", "dK_rel", "dV_rel", "dK_dir", "dM_cond1"):
        ax.plot(range(1, L), [np.nanmean(lh[(s, "A_mean")][l]) for l in range(1, L)], label=s)
    ax.axhline(0, color="k", lw=.5); ax.set_xlabel("KV layer"); ax.set_title(f"{dom}: per-layer (head-mean) Spearman vs future-query attention"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{dom}_layer_curves.png"), dpi=110); plt.close(fig)
    f6 = data[0]["fig6"]; fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(f6["recv_real"], lw=.8, label="real"); ax.plot(f6["recv_cf"], lw=.8, label="MLP removed (cond 1)")
    ax.set_title(f"{dom} {data[0]['sample_id']}: received attention mass per token, KV layer {f6['layer']}, head 0"); ax.set_yscale("log"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{dom}_attention_counterfactual.png"), dpi=110); plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/manifests/gqa_discovery.jsonl"); ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--model", default="qwen25vl"); ap.add_argument("--out", default="results/context_only/mlp_to_kv_probe")
    a = ap.parse_args()
    dom = os.path.basename(a.manifest).split("_")[0]
    out_dir = os.path.join(ROOT, a.out); os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))][:a.limit]
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    vis_mod = model.model.visual if hasattr(model.model, "visual") else None
    if vis_mod is not None and hasattr(vis_mod, "config"): vis_mod.config._attn_implementation = "sdpa"
    L, Hkv, hd = kv_dims(model)
    layer = language_layers(model)[0]
    log = [f"model {type(model).__name__}; decoder layers {L}; kv heads {Hkv}; head_dim {hd}; q heads {model.config.num_attention_heads if hasattr(model.config,'num_attention_heads') else model.config.text_config.num_attention_heads}",
           f"pre-norm: input_layernorm({type(layer.input_layernorm).__name__}) → self_attn(q/k/v_proj bias={layer.self_attn.k_proj.bias is not None}) → +residual → post_attention_layernorm → mlp → +residual",
           "RoPE: apply_multimodal_rotary_pos_emb after k_proj/q_proj view+transpose; scores use pre-RoPE K (rotation-invariant, verified), attention recompute uses post-RoPE"]
    print("\n".join(log)); open(os.path.join(out_dir, f"{dom}_structure.log"), "w").write("\n".join(log) + "\n")
    data = []
    for i, row in enumerate(rows):
        t0 = time.time(); d = collect(model, processor, row, a.device, L, Hkv, hd); data.append(d)
        c = d["checks"]
        print(f"[{i+1}/{len(rows)}] {row['sample_id']} P={d['P']} K_rec_rel={max(c['K_rec_rel']):.1e} rope_rel={max(c['rope_rel']):.1e} resid={max(c['resid_rel_err']):.1e} "
              f"dK_rope_inv={max(c['dK_rope_invariance']):.1e} {time.time()-t0:.0f}s", flush=True)
        if max(c["K_rec_rel"]) > 2e-2 or max(c["rope_rel"]) > 2e-2 or max(c["resid_rel_err"]) > 5e-3:
            raise SystemExit("verification failed; stopping before analysis")
    torch.save(data, os.path.join(out_dir, f"{dom}_raw.pt"))
    analyze(data, L, Hkv, out_dir, dom)


if __name__ == "__main__":
    main()
