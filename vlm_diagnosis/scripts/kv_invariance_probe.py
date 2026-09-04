"""§3 전제 진단 probe — 시각 위치 K/V가 뒤에 오는 텍스트에 (수치적으로) 얼마나 의존하는가.

core_delta.visual_kv_invariance 가 Qwen3-VL에서 큰 ΔV(원소 단위로 |V|max 크기)를
보고했다. 인과 마스크상 시각 hidden state는 뒤 텍스트에 의존할 수 없으므로 남는 후보는
(1) 시퀀스 길이에 따른 fp16 kernel(GEMM tiling/softmax) 반올림 차이, (2) 그 차이가
근사-0 norm token의 RMSNorm 등에서 증폭되는 불안정. 이를 가르기 위해 텍스트 쌍을 나눠 잰다.

  A  = 평가 질문 q1                (기준)
  E  = A 반복                      (결정성 대조: 0이어야 함)
  L  = 같은 token 길이의 무의미 텍스트("x x x ...")   → 길이 같고 내용 다름
  B  = "x"                         → 길이 다르고 내용 다름 (core_delta_sweep와 동일)
  D  = 다른 질문 q2                → 길이·내용 모두 다름

길이가 같을 때(A vs L) 차이가 사라지면 원인은 길이 의존 kernel 수치이고, 남으면
(불가능하지만) 내용 의존이다. 층별·token별 분포로 "어느 token이 뒤집히는지"도 기록한다.

  python -m vlm_diagnosis.scripts.kv_invariance_probe --model qwen3vl --device cuda:0 \
      --out results/smoke/kv_invariance_probe_q3.json
"""
import argparse
import json
import os

import torch
from PIL import Image

from vlm_diagnosis.core.loader import load_vlm
from vlm_diagnosis.core.signals import vlm_inputs
from vlm_diagnosis.core.spans import token_spans
from vlm_diagnosis.exps.m2a_fixed_budget import MAX_PIXELS, BRIEF

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


@torch.no_grad()
def visual_kv(model, processor, img, text, device):
    ins = vlm_inputs(processor, img, text, device)
    out = model(input_ids=ins["input_ids"], attention_mask=torch.ones_like(ins["input_ids"]),
                pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"],
                use_cache=True, output_hidden_states=False)
    sp = token_spans(ins["input_ids"], model.config)
    pk = out.past_key_values
    n_layers = len(pk.layers) if hasattr(pk, "layers") else len(pk)
    Ks, Vs = [], []
    for li in range(n_layers):
        if hasattr(pk, "layers"):
            k, v = pk.layers[li].keys, pk.layers[li].values
        elif hasattr(pk, "key_cache"):
            k, v = pk.key_cache[li], pk.value_cache[li]
        else:
            k, v = pk[li]
        vis = sp["visual"].to(k.device)
        Ks.append(k[0, :, vis].float().cpu())      # (Hkv, n_vis, d)
        Vs.append(v[0, :, vis].float().cpu())
    logits_last = out.logits[0, -1].float().cpu()
    return {"K": Ks, "V": Vs, "L": int(sp["L"]), "n_vis": int(sp["visual"].numel()),
            "logit_argmax": int(logits_last.argmax())}


def same_length_filler(processor, img, ref_text, device):
    """ref_text와 input_ids 길이가 같은 'x x x ...' 텍스트를 찾는다."""
    L_ref = vlm_inputs(processor, img, ref_text, device)["input_ids"].shape[1]
    n = 1
    for _ in range(64):
        t = " ".join(["x"] * n)
        L = vlm_inputs(processor, img, t, device)["input_ids"].shape[1]
        if L == L_ref:
            return t, L
        n += max(1, L_ref - L)
    return None, None


def compare(A, B, top=5):
    per_layer = []
    tok_dv = torch.zeros(A["n_vis"])
    tok_v = torch.zeros(A["n_vis"])
    for ka, kb, va, vb in zip(A["K"], B["K"], A["V"], B["V"]):
        dk, dv = (ka - kb), (va - vb)
        per_layer.append({"rel_fro_dK": float(dk.norm() / ka.norm().clamp(min=1e-12)),
                          "rel_fro_dV": float(dv.norm() / va.norm().clamp(min=1e-12)),
                          "max_abs_dK": float(dk.abs().max()), "max_abs_dV": float(dv.abs().max())})
        tok_dv += dv.pow(2).sum(dim=(0, 2))       # token별 ΔV 제곱합 (전 층·헤드 누적)
        tok_v += va.pow(2).sum(dim=(0, 2))
    tok_rel = (tok_dv.sqrt() / tok_v.sqrt().clamp(min=1e-12))
    order = torch.argsort(tok_rel, descending=True)[:top]
    nK = sum(float((ka - kb).norm() ** 2) for ka, kb in zip(A["K"], B["K"])) ** 0.5
    nV = sum(float((va - vb).norm() ** 2) for va, vb in zip(A["V"], B["V"])) ** 0.5
    fK = sum(float(ka.norm() ** 2) for ka in A["K"]) ** 0.5
    fV = sum(float(va.norm() ** 2) for va in A["V"]) ** 0.5
    return {"len_A": A["L"], "len_B": B["L"], "rel_fro_dK": nK / max(fK, 1e-12),
            "rel_fro_dV": nV / max(fV, 1e-12),
            "max_abs_dK": max(p["max_abs_dK"] for p in per_layer),
            "max_abs_dV": max(p["max_abs_dV"] for p in per_layer),
            "n_tokens_rel_dV_gt_0.1": int((tok_rel > 0.1).sum()),
            "n_tokens_rel_dV_gt_0.01": int((tok_rel > 0.01).sum()),
            "median_token_rel_dV": float(tok_rel.median()),
            "top_tokens": [{"vis_idx": int(i), "rel_dV": float(tok_rel[i]),
                            "V_norm": float(tok_v[i].sqrt())} for i in order],
            "per_layer": per_layer,
            "same_next_token": A["logit_argmax"] == B["logit_argmax"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl", choices=["qwen25vl", "qwen3vl"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--manifest", default="experiments/manifests/screenqa_discovery.jsonl")
    ap.add_argument("--n-images", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, a.manifest))][:a.n_images]
    model, processor = load_vlm(a.model, device=a.device, max_pixels=MAX_PIXELS)
    report = {"model": a.model, "images": []}
    for row in rows:
        img = Image.open(os.path.join(ROOT, row["image"])).convert("RGB")
        qs = row["questions"]
        tA = qs[1]["question"] + BRIEF
        tD = qs[2]["question"] + BRIEF
        tL, L_L = same_length_filler(processor, img, tA, a.device)
        A = visual_kv(model, processor, img, tA, a.device)
        pairs = {"E_repeat_same_text": tA, "L_same_length_diff_content": tL,
                 "B_x_diff_length": "x", "D_other_question": tD}
        res = {"sample_id": row["sample_id"], "n_vis": A["n_vis"], "len_A": A["L"], "pairs": {}}
        for name, t in pairs.items():
            if t is None:
                res["pairs"][name] = None
                continue
            Bv = visual_kv(model, processor, img, t, a.device)
            c = compare(A, Bv)
            c["text_B"] = t[:60]
            res["pairs"][name] = c
            print(f"[{row['sample_id']}] {name:28s} lenA={c['len_A']} lenB={c['len_B']} "
                  f"relK={c['rel_fro_dK']:.2e} relV={c['rel_fro_dV']:.2e} "
                  f"maxdV={c['max_abs_dV']:.3g} tok>0.1={c['n_tokens_rel_dV_gt_0.1']} "
                  f"top={[(t['vis_idx'], round(t['rel_dV'], 3)) for t in c['top_tokens'][:3]]}",
                  flush=True)
            del Bv
        report["images"].append(res)
        del A
        torch.cuda.empty_cache()
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
