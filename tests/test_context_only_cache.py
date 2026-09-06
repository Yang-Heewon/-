"""Context-only 메모리: keep=100% parity, 독립 분기, 질문 순서 불변, NLL mask, 실제 저장량 (tiny Qwen2.5-VL text, CPU)."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

from vlm_diagnosis.core.context_only_cache import (
    CompressedMemory, answer_from_cache, answer_nll, build_memory, method_scores)
from vlm_diagnosis.core.mlp_dynamics import MLPDynamicsCollector
from vlm_diagnosis.core.ragged_kv import RaggedKVCache
from vlm_diagnosis.core.session_adapters import QwenImageTemplate
from vlm_diagnosis.core import static_pair_select as S


class _LM(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        cfg = Qwen2_5_VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                                   rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                                   pad_token_id=0, bos_token_id=1, eos_token_id=2)
        cfg._attn_implementation = "eager"; cfg.image_token_id = 50
        self.model = Qwen2_5_VLTextModel(cfg).eval()
        self.lm_head = nn.Linear(32, 64, bias=False)
        self.config = cfg
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.eval()

    def forward(self, **kw):
        kw.pop("pixel_values", None); kw.pop("image_grid_thw", None)
        out = self.model(**kw)
        return SimpleNamespace(logits=self.lm_head(out.last_hidden_state), past_key_values=out.past_key_values)


class _Tok:
    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        return SimpleNamespace(input_ids=torch.tensor([[int(t) for t in text.split()]], dtype=torch.long) if text else torch.zeros(1, 0, dtype=torch.long))

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(map(str, tokens))


class _Template(QwenImageTemplate):
    """suffix(question) = 질문 텍스트의 정수 token; 실제 processor 없이 protocol 모양만 유지."""
    def __init__(self, prefix_ids):
        self.prefix_ids = prefix_ids.clone(); self.anchor_ids = torch.tensor([[60]]); self.ending_ids = torch.tensor([[61]])
        self.tokenizer = _Tok(); self.processor = SimpleNamespace(tokenizer=self.tokenizer)

    def suffix(self, question, first):
        return torch.tensor([[int(t) for t in question.split()]], dtype=torch.long)


@torch.no_grad()
def prefill(model, ids):
    pos = torch.arange(ids.shape[1])[None, None].expand(3, 1, -1)
    col = MLPDynamicsCollector(model)
    with col:
        out = model(input_ids=ids, position_ids=pos, use_cache=True)
    kv = tuple((k.detach(), v.detach()) for k, v in out.past_key_values.to_legacy_cache())
    return {"kv": kv, "prefix_ids": ids, "next_position": ids.shape[1],
            "spans": {"visual": torch.tensor([1, 2, 3, 4, 5, 6]), "vis_end": 6, "P": ids.shape[1]},
            "dynamics": col.result(), "qk": None, "prefill_seconds": 0.0, "input_seconds": 0.0}


def make_memory(model, pre, method, keep, seed=0, **kw):
    mem, rep, keep_mask = build_memory(model, None, pre, "ctx", method, keep, seed, "cpu", special_ids=[7], **kw)
    return mem, rep, keep_mask


class ContextOnlyTest(unittest.TestCase):
    def setUp(self):
        self.model = _LM()
        self.ids = torch.tensor([[7, 50, 50, 50, 50, 50, 50, 8, 9, 10, 11, 12]])
        self.pre = prefill(self.model, self.ids)
        # build_memory 가 QwenImageTemplate(processor, ...) 를 만들므로 가짜 template 로 교체
        import vlm_diagnosis.core.context_only_cache as C
        self._orig = C.QwenImageTemplate
        C.QwenImageTemplate = lambda processor, tid, prefix: _Template(prefix)

    def tearDown(self):
        import vlm_diagnosis.core.context_only_cache as C
        C.QwenImageTemplate = self._orig

    @torch.no_grad()
    def test_full_memory_matches_dense_forward(self):
        mem, rep, _ = make_memory(self.model, self.pre, "full", 1.0)
        self.assertEqual(rep.n_pairs_kept, rep.n_pairs_initial)
        q = "20 21 22"
        b = mem.clone_owned()
        from vlm_diagnosis.core.context_only_cache import _ragged_forward
        from vlm_diagnosis.core.ragged_kv import RaggedAttention
        with RaggedAttention(self.model, b.cache, collect=False):
            out, _ = _ragged_forward(self.model, b, b.template.suffix(q, True), b.next_position, "cpu")
        full_ids = torch.cat([self.ids, torch.tensor([[20, 21, 22]])], 1)
        pos = torch.arange(full_ids.shape[1])[None, None].expand(3, 1, -1)
        ref = self.model(input_ids=full_ids, position_ids=pos, use_cache=False).logits[0, -3:]
        torch.testing.assert_close(out.logits[0], ref, atol=1e-5, rtol=1e-5)

    @torch.no_grad()
    def test_nll_mask_covers_answer_tokens_only(self):
        mem, _, _ = make_memory(self.model, self.pre, "full", 1.0)
        b = mem.clone_owned()
        r = answer_nll(self.model, b, "20 21", "30 31 32", "cpu")
        self.assertEqual(r["n_answer_tokens"], 3)
        full_ids = torch.cat([self.ids, torch.tensor([[20, 21, 30, 31, 32]])], 1)
        pos = torch.arange(full_ids.shape[1])[None, None].expand(3, 1, -1)
        logits = self.model(input_ids=full_ids, position_ids=pos, use_cache=False).logits[0]
        lp = torch.log_softmax(logits[-4:-1], -1).gather(1, torch.tensor([[30], [31], [32]]))[:, 0]
        self.assertAlmostEqual(r["nll"], float(-lp.mean()), places=5)
        with self.assertRaises(ValueError):
            answer_nll(self.model, mem.clone_owned(), "20", "", "cpu")

    @torch.no_grad()
    def test_physical_deletion_budget_and_independent_branches(self):
        mem, rep, keep = make_memory(self.model, self.pre, "d", 0.5)
        L, H, T = 3, 2, 12
        B = S.budget_pairs(0.5, L, H, T)
        self.assertEqual(rep.n_pairs_kept, B); self.assertEqual(mem.cache.pair_count, B)
        self.assertEqual(mem.kv_bytes, B * 2 * 8 * 4)                 # fp32 tiny: 2 * head_dim * 4 bytes
        self.assertLess(mem.kv_bytes, L * H * T * 2 * 8 * 4)
        # 보호: 앞 4 + special(7) 위치 — 모든 head 에 존재
        for h in mem.cache.heads:
            self.assertTrue(set([0, 1, 2, 3]) <= set(h.token_ids.tolist()))
        # master 는 평가로 바뀌지 않고, 분기끼리 독립
        before = [h.token_ids.clone() for h in mem.cache.heads]
        r1 = answer_from_cache(self.model, mem.clone_owned(), "20 21", "cpu", max_new_tokens=3)
        r2 = answer_from_cache(self.model, mem.clone_owned(), "20 21", "cpu", max_new_tokens=3)
        self.assertEqual(r1["prediction"], r2["prediction"])
        self.assertTrue(all(torch.equal(a, h.token_ids) for a, h in zip(before, mem.cache.heads)))
        self.assertEqual(mem.cache.pair_count, B)
        # 질문 순서를 바꿔도 질문별 결과 동일
        qa = answer_nll(self.model, mem.clone_owned(), "20 21", "30", "cpu")["nll"]
        qb = answer_nll(self.model, mem.clone_owned(), "22 23", "31", "cpu")["nll"]
        qb2 = answer_nll(self.model, mem.clone_owned(), "22 23", "31", "cpu")["nll"]
        qa2 = answer_nll(self.model, mem.clone_owned(), "20 21", "30", "cpu")["nll"]
        self.assertAlmostEqual(qa, qa2, places=6); self.assertAlmostEqual(qb, qb2, places=6)

    @torch.no_grad()
    def test_selection_is_question_independent_and_directions_differ(self):
        _, rep1, k1 = make_memory(self.model, self.pre, "r", 0.5)
        _, rep2, k2 = make_memory(self.model, self.pre, "r", 0.5)
        self.assertEqual(rep1.selection_digest, rep2.selection_digest)
        _, _, k3 = make_memory(self.model, self.pre, "r", 0.5, direction="keep_low")
        self.assertFalse(torch.equal(k1, k3))
        self.assertEqual(int(k3.sum()), int(k1.sum()))
        for m in ("random", "recent", "k_norm", "v_norm", "mlp_norm", "d", "d_shift", "r_std", "hidden_rel", "hidden_cos", "d_shuffle", "d_anchor"):
            s, mapping, info = method_scores(m, self.pre, 3, 2, 0)
            self.assertEqual(tuple(s.shape), (3, 2, 12), m)
        with self.assertRaises(ValueError):
            method_scores("attn1", self.pre, 3, 2, 0)
        # d_same_zero0: layer 0 점수 0 → layer 0 은 보호분 + 동점 해소만
        _, rep_d, kd = make_memory(self.model, self.pre, "d", 0.5)
        self.assertEqual(rep_d.mapping, "d_same_zero0")
        _, _, klm = make_memory(self.model, self.pre, "d", 0.5, selector="layer_matched")
        self.assertEqual(len(set(int(klm[l].sum()) for l in range(3))), 1)


if __name__ == "__main__":
    unittest.main()
