"""MLP dynamics collector: 수식·shape·호출 횟수·parity (tiny real Qwen2.5-VL text decoder, CPU)."""
import unittest

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

from vlm_diagnosis.core.mlp_dynamics import MLPDynamicsCollector, d_topk_mean, token_table


def tiny(layers=3):
    torch.manual_seed(0)
    cfg = Qwen2_5_VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                               rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                               pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg._attn_implementation = "eager"
    return Qwen2_5_VLTextModel(cfg).eval()


def run(model, ids, collector=None):
    pos = torch.arange(ids.shape[1])[None, None].expand(3, 1, -1)
    if collector is None:
        return model(input_ids=ids, position_ids=pos, use_cache=True)
    with collector:
        return model(input_ids=ids, position_ids=pos, use_cache=True)


class CollectorTest(unittest.TestCase):
    def test_invalid_eps_and_tolerance_rejected(self):
        model = tiny(2)
        for invalid in (0, -1e-6, float("nan"), float("inf"), -float("inf"), True):
            with self.subTest(eps=invalid), self.assertRaises(ValueError):
                MLPDynamicsCollector(model, eps=invalid)
            with self.subTest(rel_tol=invalid), self.assertRaises(ValueError):
                MLPDynamicsCollector(model, rel_tol=invalid)

    @torch.no_grad()
    def test_shapes_finite_and_residual_identity(self):
        model = tiny(3)
        ids = torch.tensor([[4, 5, 6, 7, 8]])
        col = MLPDynamicsCollector(model, rel_tol=1e-4)
        run(model, ids, col)
        s = col.result()
        self.assertEqual(tuple(s.R.shape), (3, 5)); self.assertEqual(tuple(s.D.shape), (2, 5))
        self.assertEqual(tuple(s.hidden_rel.shape), (3, 5))
        self.assertTrue(torch.isfinite(s.R).all() and torch.isfinite(s.D).all())
        self.assertLess(float(s.residual_max_rel_err.max()), 1e-4)
        torch.testing.assert_close(s.R, s.mlp_norm / (s.residual_norm + 1e-6))
        torch.testing.assert_close(s.D, (s.R[1:] - s.R[:-1]).abs())

    @torch.no_grad()
    def test_norms_match_direct_hooks(self):
        model = tiny(2)
        ids = torch.tensor([[4, 5, 6]])
        got = {}
        h1 = model.layers[1].post_attention_layernorm.register_forward_pre_hook(lambda m, a: got.__setitem__("r", a[0][0].clone()))
        h2 = model.layers[1].mlp.register_forward_hook(lambda m, a, o: got.__setitem__("m", o[0].clone()))
        col = MLPDynamicsCollector(model)
        run(model, ids, col)
        h1.remove(); h2.remove()
        s = col.result()
        torch.testing.assert_close(s.mlp_norm[1], got["m"].float().norm(dim=-1))
        torch.testing.assert_close(s.residual_norm[1], got["r"].float().norm(dim=-1))

    @torch.no_grad()
    def test_parity_with_and_without_collector(self):
        model = tiny(2)
        ids = torch.tensor([[4, 5, 6, 7]])
        ref = run(model, ids)
        col = MLPDynamicsCollector(model)
        out = run(model, ids, col)
        torch.testing.assert_close(out.last_hidden_state, ref.last_hidden_state)
        for (k1, v1), (k2, v2) in zip(ref.past_key_values.to_legacy_cache(), out.past_key_values.to_legacy_cache()):
            torch.testing.assert_close(k1, k2); torch.testing.assert_close(v1, v2)
        self.assertEqual(len(col.handles), 0)

    @torch.no_grad()
    def test_default_tolerance_rejects_corrupted_layer_output(self):
        model = tiny(2)
        col = MLPDynamicsCollector(model)
        # Run before the collector's layer-output hook, leaving the captured
        # r/m untouched. This violates the sequential residual identity.
        corrupt = model.layers[0].register_forward_hook(
            lambda module, args, out: (out[0] + 0.25, *out[1:]))
        try:
            with self.assertRaisesRegex(RuntimeError, "x_next != r \\+ m"):
                run(model, torch.tensor([[4, 5, 6]]), col)
            self.assertEqual(len(col.handles), 0)
        finally:
            corrupt.remove()
        for layer in model.layers:
            self.assertFalse(layer._forward_hooks)
            self.assertFalse(layer._forward_pre_hooks)
            self.assertFalse(layer.mlp._forward_hooks)
            self.assertFalse(layer.post_attention_layernorm._forward_pre_hooks)

    @torch.no_grad()
    def test_default_tolerance_handles_low_precision_addition(self):
        model = tiny(2)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                col = MLPDynamicsCollector(model)
                r = torch.full((3, 8), 1.0, dtype=dtype)
                m = torch.full_like(r, 0.13)
                col._x[0], col._r[0], col._m[0] = r, r, m
                col._layer_post(0, model.layers[0], (), ((r + m)[None],))
                self.assertTrue(torch.isfinite(col.mlp_norm[0]).all())
                # A material residual mismatch must fail for every dtype.
                col._x[0], col._r[0], col._m[0] = r, r, m
                with self.assertRaisesRegex(RuntimeError, "x_next != r \\+ m"):
                    col._layer_post(0, model.layers[0], (), ((r + m + 0.25)[None],))

    @torch.no_grad()
    def test_double_forward_without_reset_fails_and_hooks_removed_on_error(self):
        model = tiny(2)
        ids = torch.tensor([[4, 5]])
        col = MLPDynamicsCollector(model)
        with self.assertRaises(RuntimeError):
            with col:
                run(model, ids)
                run(model, ids)
        self.assertEqual(len(col.handles), 0)
        for layer in model.layers:
            self.assertFalse(layer._forward_pre_hooks); self.assertFalse(layer._forward_hooks)
            self.assertFalse(layer.mlp._forward_hooks)
        col.reset()
        run(model, ids, col)
        self.assertEqual(tuple(col.result().R.shape), (2, 2))

    def test_d_from_constant_and_peak(self):
        # 상수 R → D = 0 ; 한 층에서만 peak → 상승·하강 두 변화
        R = torch.ones(4, 3)
        self.assertTrue(((R[1:] - R[:-1]).abs() == 0).all())
        R[2, 1] = 5.0
        D = (R[1:] - R[:-1]).abs()
        self.assertEqual(int((D[:, 1] > 0).sum()), 2)
        self.assertEqual(int((D[:, 0] > 0).sum()), 0)
        torch.testing.assert_close(d_topk_mean(D, 3)[1], torch.tensor(8.0 / 3))
        self.assertEqual(d_topk_mean(D, 10).shape, (3,))

    def test_topk_rejects_invalid_shape_count_and_nonfinite_values(self):
        for invalid in (0, -1, True, 1.5):
            with self.subTest(k=invalid), self.assertRaises(ValueError):
                d_topk_mean(torch.ones(2, 3), invalid)
        for invalid in (torch.ones(3), torch.empty(0, 3), torch.empty(2, 0),
                        torch.tensor([[float("nan")]])):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                d_topk_mean(invalid, 1)

    @torch.no_grad()
    def test_token_table(self):
        model = tiny(2)
        ids = torch.tensor([[1, 5, 6]])
        col = MLPDynamicsCollector(model)
        run(model, ids, col)
        rows = token_table(col.result(), ids, tokenizer=None, special_ids=[1])
        self.assertEqual(len(rows), 3); self.assertTrue(rows[0]["special"]); self.assertFalse(rows[1]["special"])
        self.assertIn("D_top3_mean", rows[0])

    def test_single_layer_rejected(self):
        model = tiny(1)
        col = MLPDynamicsCollector(model)
        with torch.no_grad():
            run(model, torch.tensor([[4, 5]]), col)
        with self.assertRaises(ValueError):
            col.result()


if __name__ == "__main__":
    unittest.main()
