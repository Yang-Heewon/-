import unittest

import torch

from vlm_diagnosis.core.kv_baselines import (
    KVShape,
    VisualKVTransform,
    dense_storage,
    fake_quantize_keys,
    fake_quantize_values,
    hybrid_storage,
    max_keep_for_budget,
    merge_evicted_into_kept,
    quantized_storage,
    sparse_storage,
)


class KVBaselineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.keys = torch.randn(1, 2, 9, 8)
        self.values = torch.randn(1, 2, 9, 8)

    def test_fake_quantization_shape_finite_and_nontrivial(self):
        qk = fake_quantize_keys(self.keys, nbits=4, group_size=4)
        qv = fake_quantize_values(self.values, nbits=4, group_size=4)
        self.assertEqual(qk.shape, self.keys.shape)
        self.assertEqual(qv.shape, self.values.shape)
        self.assertTrue(torch.isfinite(qk).all())
        self.assertTrue(torch.isfinite(qv).all())
        self.assertGreater((qk - self.keys).abs().max().item(), 0)
        self.assertGreater((qv - self.values).abs().max().item(), 0)

    def test_constant_group_is_stable(self):
        x = torch.ones(1, 1, 5, 4)
        self.assertTrue(torch.equal(fake_quantize_keys(x, 2, 4), x))
        self.assertTrue(torch.equal(fake_quantize_values(x, 2, 4), x))

    def test_merge_returns_only_kept_tokens(self):
        keep = torch.tensor([0, 3, 8])
        mk, mv = merge_evicted_into_kept(self.keys, self.values, keep)
        self.assertEqual(mk.shape, (1, 2, 3, 8))
        self.assertEqual(mv.shape, (1, 2, 3, 8))
        self.assertTrue(torch.equal(mk, self.keys[:, :, keep]))
        self.assertTrue(torch.isfinite(mv).all())

    def test_full_keep_merge_is_identity(self):
        keep = torch.arange(self.keys.shape[2])
        mk, mv = merge_evicted_into_kept(self.keys, self.values, keep)
        self.assertTrue(torch.equal(mk, self.keys))
        self.assertTrue(torch.equal(mv, self.values))

    def test_transform_canonicalizes_unsorted_keep_indices(self):
        transform = VisualKVTransform(
            object(),
            visual_positions=range(9),
            keep_indices=[8, 0, 3, 3],
            merge=True,
        )
        self.assertEqual(transform.keep_indices.tolist(), [0, 3, 8])

    def test_storage_accounting_and_byte_matching(self):
        shape = KVShape(layers=4, batch=1, kv_heads=2, tokens=128, head_dim=16)
        full = dense_storage(shape).total_bytes
        sparse = sparse_storage(shape, 32).total_bytes
        quant = quantized_storage(shape, 4).total_bytes
        hybrid = hybrid_storage(shape, 64, 4).total_bytes
        self.assertLess(sparse, full)
        self.assertLess(quant, full)
        self.assertLess(hybrid, full)

        budget = int(full * 0.25)
        k_sparse = max_keep_for_budget(shape, budget, "sparse")
        k_hybrid = max_keep_for_budget(shape, budget, "hybrid", nbits=4)
        self.assertLessEqual(sparse_storage(shape, k_sparse).total_bytes, budget)
        self.assertLessEqual(hybrid_storage(shape, k_hybrid, 4).total_bytes, budget)
        self.assertGreater(k_hybrid, k_sparse)


if __name__ == "__main__":
    unittest.main()
