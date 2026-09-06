"""Context-only KV 압축: 단일 prefill → 점수 → 전역 B 선택 → 실제 삭제 → 독립 분기 평가
(docs/CONTEXT-ONLY-KV-COMPRESSION.md §3, §5, 단계 1–3; VLM 화면 표본용 구현).

    memory, report = compress_context(model, processor, image, method, keep_ratio, seed, device)
    branch = memory.clone_owned(); pred = answer_from_cache(model, branch, question)
    branch = memory.clone_owned(); nll  = answer_nll(model, branch, question, gold)

- 압축기는 image(context) 만 받는다. 질문·정답은 받지 않는다.
- context prefill 은 build 당 1회. 재구성·설명 생성·context 재입력 없음 (reconstruction 은 별도 명시 method).
- 선택 단위는 (layer, KV head, logical token) 의 K/V 한 쌍. 미선택 쌍은 RaggedKVCache 에서 실제로 삭제된다.
- master memory 는 평가로 바뀌지 않는다. 질문마다 clone_owned() 로 독립 분기를 만든다.
- 남은 token 의 logical 위치와 RoPE 는 그대로 보존된다 (재번호화 없음).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time

import torch
from transformers.cache_utils import Cache

from .loader import assert_finite_logits
from .masked_eval import mrope_position_ids
from .mlp_dynamics import MLPDynamicsCollector
from .ragged_kv import HeadKV, RaggedAttention, RaggedKVCache
from .session_adapters import QwenImageTemplate, QwenPairAdapter, SessionInput
from .spans import token_spans
from .signals import vlm_inputs
from . import static_pair_select as SEL

CONTEXT_ONLY_METHODS = ("full", "random", "recent", "k_norm", "v_norm", "mlp_norm", "r", "d", "d_shift",
                        "r_std", "hidden_rel", "hidden_cos", "d_shuffle", "d_anchor")
COST_FLAGGED_METHODS = ("attn1",)          # context-only 이지만 attention 재계산 비용이 붙는 방법
EXTERNAL_METHODS = ("recon_desc",)         # 재구성(설명문 생성) — 단일 prefill 계약 밖, 별도 명시 실행에서만


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class BuildReport:
    context_id: str
    method: str
    mapping: str
    direction: str
    keep_ratio: float
    n_tokens: int
    n_layers: int
    n_heads: int
    head_dim: int
    n_pairs_initial: int
    n_pairs_kept: int
    n_protected_pairs: int
    keep_ratio_actual: float
    kv_bytes: int
    metadata_bytes: int
    per_layer_counts: list
    selection_digest: str
    prefill_calls: int
    prefill_seconds: float
    score_seconds: float
    select_seconds: float
    prune_seconds: float
    build_seconds: float
    peak_bytes: int
    residual_max_rel_err: float | None
    next_position: int
    extra: dict


class CompressedMemory:
    """생존 K/V 만 소유하는 context 메모리 + 질문 suffix 를 붙이기 위한 template."""

    def __init__(self, cache: RaggedKVCache, template: QwenImageTemplate, next_position: int, prefix_len: int):
        self.cache, self.template = cache, template
        self.next_position, self.prefix_len = next_position, prefix_len
        self.adapter = QwenPairAdapter()

    @property
    def kv_bytes(self):
        return self.cache.nbytes

    @property
    def metadata_bytes(self):
        return self.cache.metadata_bytes + self.adapter.metadata_bytes(self.template)

    def clone_owned(self) -> "CompressedMemory":
        c = RaggedKVCache.__new__(RaggedKVCache)
        Cache.__init__(c, layers=[])
        for k in ("n_layers", "n_heads", "head_dim", "total_seen", "pair_bytes"):
            setattr(c, k, getattr(self.cache, k))
        c.heads = [HeadKV(h.key.clone(), h.value.clone(), h.token_ids.clone()) for h in self.cache.heads]
        c.backend_active, c._next_layer, c.query_ids = False, 0, None
        return CompressedMemory(c, self.template, self.next_position, self.prefix_len)

    def resident_ids(self):
        return [h.token_ids.clone() for h in self.cache.heads]


# ---------------------------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def prefill_context(model, processor, image, device, collect_dynamics=True, capture_qk=False, eps=1e-6):
    """context(이미지) prefill 정확히 1회. 반환 dict: kv(GPU dense tuple), prefix_ids, next_position,
    spans, dynamics(MLPDynamics|None), qk(list|None), prefill_seconds, input_seconds."""
    t_in = time.perf_counter()
    ins = vlm_inputs(processor, image, "x", device)
    sp = token_spans(ins["input_ids"], model.config)
    P = int(sp["vis_end"]) + 2
    ids = ins["input_ids"][:, :P]
    pos = mrope_position_ids(model, ids, ins["image_grid_thw"], torch.ones_like(ids))
    _sync(device)
    input_seconds = time.perf_counter() - t_in
    t0 = time.perf_counter()
    col = MLPDynamicsCollector(model, mode="stats", eps=eps) if collect_dynamics else None
    cap = None
    if capture_qk:
        from .attnstat import QKCapture
        cap = QKCapture()
    ctxs = [c for c in (col, cap) if c is not None]
    for c in ctxs:
        c.__enter__()
    try:
        out = model(input_ids=ids, position_ids=pos, attention_mask=torch.ones_like(ids),
                    pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"],
                    use_cache=True, output_attentions=False)
    finally:
        for c in reversed(ctxs):
            c.__exit__(None, None, None)
    assert_finite_logits(out.logits, "context_only_prefill")
    _sync(device)
    prefill_seconds = time.perf_counter() - t0
    kv = tuple((k.detach(), v.detach()) for k, v in out.past_key_values.to_legacy_cache())
    del out
    return {"kv": kv, "prefix_ids": ids.detach().cpu(), "next_position": int(pos.max()) + 1,
            "spans": {"visual": sp["visual"].cpu(), "vis_end": int(sp["vis_end"]), "P": P},
            "dynamics": col.result() if col is not None else None,
            "qk": cap.qk if cap is not None else None,
            "prefill_seconds": prefill_seconds, "input_seconds": input_seconds, "image_grid_thw": ins["image_grid_thw"]}


# ---------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------
def method_scores(method: str, pre: dict, n_layers: int, n_heads: int, seed: int, extra: dict | None = None):
    """method 이름 → (score (L,H,T), mapping 이름, 부가 정보). extra 는 attn1/recon_desc 처럼
    외부에서 계산된 (L,H,T) 점수를 넘길 때 사용."""
    T = pre["spans"]["P"]
    dyn = pre["dynamics"]
    info = {}
    if method == "full":
        return None, "none", info
    if method == "random":
        return SEL.random_scores((n_layers, n_heads, T), seed), "random", info
    if method == "recent":
        return SEL.recent_scores(n_layers, n_heads, T), "recent", info
    if method in ("k_norm", "v_norm"):
        return SEL.head_norm_scores(pre["kv"], method[0]), "head_norm", info
    if method in ("attn1", "recon_desc"):
        if extra is None or method not in extra:
            raise ValueError(f"{method} requires an externally computed score")
        return extra[method].float(), method, info
    if dyn is None:
        raise ValueError(f"{method} requires MLP dynamics from the prefill")
    if method == "mlp_norm":
        return SEL.map_token_signal(dyn.mlp_norm, "mlp_norm_same", n_layers, n_heads), "mlp_norm_same", info
    if method == "r":
        return SEL.map_token_signal(dyn.R, "r_same", n_layers, n_heads), "r_same", info
    if method == "d":
        return SEL.map_token_signal(dyn.D, "d_same_zero0", n_layers, n_heads), "d_same_zero0", info
    if method == "d_shift":
        return SEL.map_token_signal(dyn.D, "d_shift_prev", n_layers, n_heads), "d_shift_prev", info
    if method == "r_std":
        return SEL.map_token_signal(dyn.R.std(0, unbiased=False), "r_std_same", n_layers, n_heads), "r_std_same", info
    if method == "hidden_rel":
        return SEL.map_token_signal(dyn.hidden_rel, "hidden_rel_same", n_layers, n_heads), "hidden_rel_same", info
    if method == "hidden_cos":
        return SEL.map_token_signal(dyn.hidden_cos, "hidden_cos_same", n_layers, n_heads), "hidden_cos_same", info
    if method == "d_shuffle":
        Ds, pi = SEL.shuffled_D(dyn.R, seed)
        info["layer_permutation"] = pi
        return SEL.map_token_signal(Ds, "d_same_zero0", n_layers, n_heads), "d_same_zero0(shuffled_R)", info
    if method == "d_anchor":
        Da, js = SEL.anchor_D(dyn.R, seed)
        info["anchor_compare_layers"] = js
        return SEL.map_token_signal(Da, "d_same_zero0", n_layers, n_heads), "d_same_zero0(anchor)", info
    raise ValueError(f"unknown method {method}")


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def build_memory(model, processor, pre: dict, context_id: str, method: str, keep_ratio: float, seed: int,
                 device, direction: str = "keep_high", selector: str = "global", special_ids=(),
                 n_prefix_protect: int = 4, extra_scores: dict | None = None, boundary_seed: int = 777,
                 timing: dict | None = None):
    """prefill 결과(pre)에서 method 로 점수를 매겨 실제 삭제된 CompressedMemory 를 만든다.
    direction: keep_high(기본) | keep_low(높은 점수부터 삭제 = 민감도 대조).
    selector: global | layer_matched | boundary."""
    kv = pre["kv"]
    n_layers, n_heads = len(kv), kv[0][0].shape[1]
    T = pre["spans"]["P"]
    head_dim = kv[0][0].shape[-1]
    n_pairs = n_layers * n_heads * T
    B = SEL.budget_pairs(keep_ratio, n_layers, n_heads, T)
    protected = SEL.protected_positions(pre["prefix_ids"], special_ids, n_prefix_protect)
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device); t0 = time.perf_counter()
    score, mapping, info = method_scores(method, pre, n_layers, n_heads, seed, extra_scores)
    _sync(device); t_score = time.perf_counter() - t0
    t1 = time.perf_counter()
    if method == "full":
        keep = torch.ones((n_layers, n_heads, T), dtype=torch.bool)
        B = n_pairs
    else:
        s = score if direction == "keep_high" else -score
        if direction not in ("keep_high", "keep_low"):
            raise ValueError("direction must be keep_high or keep_low")
        if selector == "global":
            keep = SEL.select_pairs(s, protected, B, seed)
        elif selector == "layer_matched":
            keep = SEL.layer_matched_select(s, protected, B, seed)
        elif selector == "boundary":
            keep = SEL.boundary_control_select(s, protected, B, seed, boundary_seed)
        else:
            raise ValueError(f"unknown selector {selector}")
    t_select = time.perf_counter() - t1
    t2 = time.perf_counter()
    cache = RaggedKVCache(kv, SEL.keep_ids_per_head(keep), device=device)
    _sync(device); t_prune = time.perf_counter() - t2
    template = QwenImageTemplate(processor, int(model.config.image_token_id), pre["prefix_ids"])
    mem = CompressedMemory(cache, template, pre["next_position"], T)
    peak = int(torch.cuda.max_memory_allocated(device)) if torch.device(device).type == "cuda" else 0
    dyn = pre["dynamics"]
    prot_pairs = int(protected.sum()) * n_layers * n_heads
    rep = BuildReport(
        context_id=str(context_id), method=method, mapping=mapping, direction=direction if method != "full" else "none",
        keep_ratio=keep_ratio, n_tokens=T, n_layers=n_layers, n_heads=n_heads, head_dim=head_dim,
        n_pairs_initial=n_pairs, n_pairs_kept=cache.pair_count, n_protected_pairs=prot_pairs,
        keep_ratio_actual=cache.pair_count / n_pairs, kv_bytes=cache.nbytes, metadata_bytes=mem.metadata_bytes,
        per_layer_counts=[int(keep[l].sum()) for l in range(n_layers)], selection_digest=SEL.selection_digest(keep),
        prefill_calls=1, prefill_seconds=pre["prefill_seconds"], score_seconds=t_score, select_seconds=t_select,
        prune_seconds=t_prune, build_seconds=pre["input_seconds"] + pre["prefill_seconds"] + t_score + t_select + t_prune,
        peak_bytes=peak, residual_max_rel_err=float(dyn.residual_max_rel_err.max()) if dyn is not None else None,
        next_position=pre["next_position"], extra={"selector": selector, **info})
    return mem, rep, keep


@torch.no_grad()
def compress_context(model, processor, image, method: str, keep_ratio: float, seed: int, device,
                     context_id: str = "", special_ids=(), **kw):
    """배포 경로 계약: context 만 입력, prefill 1회, dense KV 는 압축 메모리 생성 직후 해제."""
    if method not in CONTEXT_ONLY_METHODS:
        raise ValueError(f"compress_context accepts context-only methods only, got {method}")
    needs_dyn = method not in ("full", "random", "recent", "k_norm", "v_norm")
    pre = prefill_context(model, processor, image, device, collect_dynamics=needs_dyn)
    mem, rep, _ = build_memory(model, processor, pre, context_id, method, keep_ratio, seed, device,
                               special_ids=special_ids, **kw)
    del pre["kv"]; pre["kv"] = None
    _sync(device)
    return mem, rep


# ---------------------------------------------------------------------------------------------
# evaluation on an owned branch
# ---------------------------------------------------------------------------------------------
def _ragged_forward(model, mem: CompressedMemory, ids: torch.Tensor, position: int, device):
    prepared = SessionInput(input_ids=ids, modality_ids=torch.ones(ids.shape[1], dtype=torch.long))
    out, nxt = mem.adapter.forward(model, prepared, mem.cache, position, device)
    return out, nxt


@torch.no_grad()
def answer_from_cache(model, branch: CompressedMemory, question: str, device, max_new_tokens: int = 32):
    """압축 context 뒤에 질문 suffix 를 먼저 처리한 뒤 greedy 생성. 첫 답 token 은 suffix 마지막 위치의 logits."""
    suffix = branch.template.suffix(question, first=True)
    eos = branch.adapter.stop_token_ids(model)
    generated, stop = [], None
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device); t0 = time.perf_counter()
    with RaggedAttention(model, branch.cache, collect=False):
        out, pos = _ragged_forward(model, branch, suffix, branch.next_position, device)
        _sync(device); t_query = time.perf_counter() - t0
        t1 = time.perf_counter()
        for _ in range(max_new_tokens):
            tok = int(out.logits[0, -1].argmax())
            if tok in eos:
                stop = tok
                break
            generated.append(tok)
            out, pos = _ragged_forward(model, branch, torch.tensor([[tok]]), pos, device)
        _sync(device); t_decode = time.perf_counter() - t1
    del out
    peak = int(torch.cuda.max_memory_allocated(device)) if torch.device(device).type == "cuda" else 0
    return {"prediction": branch.adapter.decode(branch.template.processor, generated),
            "generated_tokens": len(generated), "hit_generation_limit": stop is None,
            "suffix_tokens": int(suffix.shape[1]), "query_seconds": t_query, "decode_seconds": t_decode,
            "query_peak_bytes": peak}


@torch.no_grad()
def answer_nll(model, branch: CompressedMemory, question: str, answer_text: str, device):
    """teacher-forced 정답 본문 token 의 평균 NLL (EOS/chat 종료 token 제외). 본문 token 0개면 오류."""
    tok = branch.template.tokenizer
    suffix = branch.template.suffix(question, first=True)
    ans = tok(answer_text, add_special_tokens=False, return_tensors="pt").input_ids
    if ans.shape[1] == 0:
        raise ValueError("answer has no body tokens")
    ids = torch.cat([suffix, ans], dim=1)
    with RaggedAttention(model, branch.cache, collect=False):
        out, _ = _ragged_forward(model, branch, ids, branch.next_position, device)
    logits = out.logits[0, suffix.shape[1] - 1: ids.shape[1] - 1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(1, ans[0].to(logits.device)[:, None])[:, 0]
    del out
    return {"nll": float(-logp.mean()), "n_answer_tokens": int(ans.shape[1]), "token_logp": logp.cpu().tolist()}


@torch.no_grad()
def dense_reference_logits(model, processor, image, question: str, device, prefix_len_expected: int | None = None):
    """일반 FULL forward [context + question] 의 질문 suffix 구간 logits (parity 검사용, 평가 전용)."""
    ins = vlm_inputs(processor, image, question, device)
    sp = token_spans(ins["input_ids"], model.config)
    P = int(sp["vis_end"]) + 2
    if prefix_len_expected is not None and P != prefix_len_expected:
        raise ValueError("dense reference prefix length disagrees with cached prefix")
    ids = ins["input_ids"]
    pos = mrope_position_ids(model, ids, ins["image_grid_thw"], torch.ones_like(ids))
    out = model(input_ids=ids, position_ids=pos, attention_mask=torch.ones_like(ids),
                pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"], use_cache=False)
    return out.logits[0, P:].float(), ids[0, P:].cpu(), pos[:, 0, P:].cpu()


def report_dict(rep: BuildReport) -> dict:
    return asdict(rep)
