import unittest
from unittest.mock import patch

import torch

from vlm_diagnosis.core import static_pair_select as S


class SelectTest(unittest.TestCase):
    def test_budget_rule(self):
        self.assertEqual(S.budget_pairs(0.2, 2, 2, 10), 8)
        self.assertEqual(S.budget_pairs(1.0, 2, 2, 10), 40)
        for bad in (0.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                S.budget_pairs(bad, 2, 2, 10)

    def test_protected_positions(self):
        ids = torch.tensor([[9, 9, 9, 9, 9, 7, 9, 7]])
        m = S.protected_positions(ids, special_ids=[7], n_prefix=4)
        self.assertEqual(m.tolist(), [True, True, True, True, False, True, False, True])
        self.assertEqual(S.protected_positions(torch.tensor([[1, 2]]), [], 4).tolist(), [True, True])

    def test_exact_budget_and_protection(self):
        torch.manual_seed(0)
        score = torch.rand(2, 2, 10)
        prot = torch.zeros(10, dtype=torch.bool); prot[:2] = True
        keep = S.select_pairs(score, prot, 12, seed=0)
        self.assertEqual(int(keep.sum()), 12)
        self.assertTrue(bool(keep[:, :, :2].all()))
        # 비보호 선택은 점수 상위여야 함
        free = score.clone(); free[:, :, :2] = -1
        chosen = free[keep & ~prot[None, None].expand(2, 2, 10)]
        dropped = free[~keep]
        self.assertGreaterEqual(float(chosen.min()), float(dropped.max()))
        with self.assertRaises(ValueError):
            S.select_pairs(score, prot, 7, seed=0)      # 8 보호 쌍 > 예산
        with self.assertRaises(ValueError):
            S.select_pairs(score, prot, 41, seed=0)

    def test_tie_break_seeded_and_deterministic(self):
        score = torch.zeros(1, 1, 20)
        prot = torch.zeros(20, dtype=torch.bool)
        a = S.select_pairs(score, prot, 5, seed=1); b = S.select_pairs(score, prot, 5, seed=1)
        c = S.select_pairs(score, prot, 5, seed=2)
        self.assertTrue(torch.equal(a, b)); self.assertFalse(torch.equal(a, c))
        self.assertEqual(int(a.sum()), 5)
        # 동점이 아닐 때 seed 는 결과를 바꾸지 않는다
        s2 = torch.arange(20.).view(1, 1, 20)
        self.assertTrue(torch.equal(S.select_pairs(s2, prot, 5, 1), S.select_pairs(s2, prot, 5, 9)))

    def test_tie_ranks_stay_integer_beyond_fp32_exact_range(self):
        self.assertEqual(S._tie_permutation((1, 1, 5), 1).dtype, torch.int64)
        # Isolate two ranks from a large pair population without allocating
        # millions of score entries. FP32 would collapse these distinct ranks.
        tie = torch.tensor([[[2**24 + 1, 2**24]]], dtype=torch.int64)
        with patch.object(S, "_tie_permutation", return_value=tie):
            keep = S.select_pairs(torch.zeros(1, 1, 2), torch.zeros(2, dtype=torch.bool), 1, 0)
        self.assertEqual(keep.tolist(), [[[False, True]]])

    def test_nonfinite_rejected(self):
        score = torch.zeros(1, 1, 5); score[0, 0, 3] = float("nan")
        with self.assertRaises(ValueError):
            S.select_pairs(score, torch.zeros(5, dtype=torch.bool), 2, 0)

    def test_mappings(self):
        L, H, T = 3, 2, 4
        sig = torch.arange(float(L * T)).view(L, T)
        m = S.map_token_signal(sig, "r_same", L, H)
        self.assertEqual(tuple(m.shape), (L, H, T)); self.assertTrue(torch.equal(m[:, 0], m[:, 1]))
        D = torch.ones(L - 1, T) * 2
        z = S.map_token_signal(D, "d_same_zero0", L, H)
        self.assertTrue(bool((z[0] == 0).all()) and bool((z[1:] == 2).all()))
        sh = S.map_token_signal(D, "d_shift_prev", L, H)
        self.assertTrue(bool((sh[:2] == 0).all()) and bool((sh[2:] == 2).all()))
        with self.assertRaises(ValueError):
            S.map_token_signal(sig, "d_same_zero0", L, H)
        with self.assertRaises(ValueError):
            S.map_token_signal(sig, "nope", L, H)
        std = S.map_token_signal(torch.arange(4.), "r_std_same", L, H)
        self.assertEqual(tuple(std.shape), (L, H, T))

    def test_shifted_D_is_one_layer_later_and_omits_final_observation(self):
        D = torch.tensor([[1., 10.], [2., 20.], [99., 990.]])
        shifted = S.map_token_signal(D, "d_shift_prev", 4, 2)
        expected = torch.tensor([[0., 0.], [0., 0.], [1., 10.], [2., 20.]])
        for head in range(2):
            torch.testing.assert_close(shifted[:, head], expected)
        same = S.map_token_signal(D, "d_same_zero0", 4, 2)
        torch.testing.assert_close(shifted[1:], same[:-1])
        # With only two layers the sole D observation has no later KV layer.
        two_layers = S.map_token_signal(torch.tensor([[3., 4.]]), "d_shift_prev", 2, 1)
        self.assertTrue(bool((two_layers == 0).all()))

    def test_mapping_validates_dimensions_shapes_and_finite_signals(self):
        for name in ("d_same_zero0", "d_shift_prev"):
            with self.subTest(mapping=name), self.assertRaises(ValueError):
                S.map_token_signal(torch.empty(0, 3), name, 1, 2)
        for bad in (0, -1, True, 1.5):
            with self.subTest(heads=bad), self.assertRaises(ValueError):
                S.map_token_signal(torch.ones(2, 3), "r_same", 2, bad)
            with self.subTest(layers=bad), self.assertRaises(ValueError):
                S.map_token_signal(torch.ones(2, 3), "r_same", bad, 2)
        for invalid in (torch.ones(2), torch.ones(2, 3, 1), torch.empty(2, 0),
                        torch.tensor([[1., float("inf")], [1., 2.]])):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                S.map_token_signal(invalid, "r_same", 2, 2)

    def test_layer_matched_and_boundary(self):
        torch.manual_seed(1)
        score = torch.rand(3, 2, 10)
        prot = torch.zeros(10, dtype=torch.bool); prot[:1] = True
        keep = S.layer_matched_select(score, prot, 20, seed=0)
        self.assertEqual([int(keep[l].sum()) for l in range(3)], [7, 7, 6])
        self.assertTrue(bool(keep[:, :, 0].all()))
        kb = S.boundary_control_select(score, prot, 20, seed=0, boundary_seed=5)
        self.assertEqual(int(kb.sum()), 20)
        # layer 0 은 방법 점수와 무관: 다른 점수로도 layer 0 선택이 같다
        kb2 = S.boundary_control_select(torch.rand(3, 2, 10), prot, 20, seed=0, boundary_seed=5)
        self.assertTrue(torch.equal(kb[0], kb2[0]))
        self.assertEqual(int(kb[0].sum()), max(2, min(20, 20 - 4, round(20 / 3))))

    def test_shuffled_and_anchor_D(self):
        R = torch.rand(5, 6)
        Ds, pi = S.shuffled_D(R, 3)
        self.assertEqual(tuple(Ds.shape), (4, 6)); self.assertEqual(sorted(pi), list(range(5)))
        Ds2, pi2 = S.shuffled_D(R, 3)
        self.assertTrue(torch.equal(Ds, Ds2) and pi == pi2)
        Da, js = S.anchor_D(R, 0)
        self.assertEqual(tuple(Da.shape), (4, 6)); self.assertEqual(len(js), 4)
        for l, j in enumerate(js, start=1):
            self.assertNotEqual(l, j); self.assertTrue(0 <= j < 5)

    def test_shuffle_and_anchor_reject_invalid_R(self):
        for method in (S.shuffled_D, S.anchor_D):
            for invalid in (torch.ones(1, 3), torch.empty(0, 3), torch.empty(3, 0),
                            torch.ones(3), torch.tensor([[1., 2.], [float("nan"), 3.]])):
                with self.subTest(method=method.__name__, shape=invalid.shape), self.assertRaises(ValueError):
                    method(invalid, 0)

    def test_average_rank_and_spearman(self):
        r = S.average_rank(torch.tensor([3., 1., 3., 2.]))
        self.assertEqual(r.tolist(), [3.5, 1.0, 3.5, 2.0])
        self.assertAlmostEqual(S.spearman_avg_rank(torch.arange(10.), torch.arange(10.)), 1.0, places=6)
        self.assertAlmostEqual(S.spearman_avg_rank(torch.arange(10.), -torch.arange(10.)), -1.0, places=6)
        self.assertIsNone(S.spearman_avg_rank(torch.ones(5), torch.arange(5.)))

    def test_keep_ids_per_head(self):
        keep = torch.zeros(2, 2, 4, dtype=torch.bool); keep[1, 0, [0, 3]] = True
        ids = S.keep_ids_per_head(keep)
        self.assertEqual(len(ids), 4); self.assertEqual(ids[2].tolist(), [0, 3]); self.assertEqual(ids[0].numel(), 0)
        self.assertEqual(len(S.selection_digest(keep)), 16)


if __name__ == "__main__":
    unittest.main()


class DiversityAndSpikeTest(unittest.TestCase):
    def test_massive_activation_positions(self):
        R = torch.ones(4, 10); R[2, 3] = 50.0; R[0, 7] = 4.0
        m = S.massive_activation_positions(R, factor=10)
        self.assertEqual(m.tolist(), [i == 3 for i in range(10)])
        self.assertEqual(int(S.massive_activation_positions(R, factor=3).sum()), 2)

    def test_farthest_point_order_prefers_distinct_keys(self):
        # 3 clusters: 0,1,2 identical / 3,4 identical / 5 unique
        base = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.], [-1., 0.]])
        order = S.farthest_point_order(base[None], n_steps=3, start=0)[0]
        top3 = set(torch.topk(order, 3).indices.tolist())
        self.assertEqual(top3, {0, 5, 3} if 3 in top3 else {0, 5, 4})
        # seed_mask: 이미 뽑힌 0 을 대표로 두면 첫 선택은 가장 먼 5
        order2 = S.farthest_point_order(base[None], n_steps=1, seed_mask=torch.tensor([[True, False, False, False, False, False]]))[0]
        self.assertEqual(int(order2.argmax()), 5)

    def test_importance_diversity_budget_and_fill(self):
        torch.manual_seed(0)
        L, H, T, d = 2, 2, 12, 4
        score = torch.rand(L, H, T); keys = torch.randn(L, H, T, d)
        prot = torch.zeros(T, dtype=torch.bool); prot[:2] = True
        for frac in (0.0, 0.5, 1.0):
            keep = S.importance_diversity_select(score, keys, prot, 24, seed=0, div_frac=frac)
            self.assertEqual(int(keep.sum()), 24); self.assertTrue(bool(keep[:, :, :2].all()))
        k0 = S.importance_diversity_select(score, keys, prot, 24, 0, 0.0)
        self.assertTrue(torch.equal(k0, S.select_pairs(score, prot, 24, 0)))
        k5 = S.importance_diversity_select(score, keys, prot, 24, 0, 0.5)
        self.assertFalse(torch.equal(k0, k5))
        with self.assertRaises(ValueError):
            S.importance_diversity_select(score, keys, prot, 24, 0, 1.5)
        with self.assertRaises(ValueError):
            S.importance_diversity_select(score, keys, prot, 7, 0, 0.5)


class HiddenToKTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.L, self.H, self.T, self.d = 2, 2, 24, 4
        self.keys = torch.randn(self.L, self.H, self.T, self.d)
        self.prot = torch.zeros(self.T, dtype=torch.bool); self.prot[:2] = True
        self.z = torch.randn(self.T, 8)
        self.clusters = S.kmeans_labels(self.z, 4)

    def test_kmeans_labels(self):
        self.assertEqual(tuple(self.clusters.shape), (self.T,)); self.assertLess(int(self.clusters.max()), 4)
        self.assertTrue(torch.equal(self.clusters, S.kmeans_labels(self.z, 4)))

    def test_token_coverage_exact_budget_and_shared_tokens(self):
        for B in (32, 35, 96):
            keep = S.token_coverage_select(self.z, self.prot, B, self.L, self.H)
            self.assertEqual(int(keep.sum()), B); self.assertTrue(bool(keep[:, :, :2].all()))
        keep = S.token_coverage_select(self.z, self.prot, 32, self.L, self.H)
        self.assertTrue(all(torch.equal(keep[0, 0], keep[l, h]) for l in range(2) for h in range(2)))
        with self.assertRaises(ValueError):
            S.token_coverage_select(self.z, self.prot, 7, self.L, self.H)

    def test_cluster_quota_budget_protection_and_quota(self):
        B = 48
        keep = S.cluster_quota_select(self.keys, self.clusters, self.prot, B, seed=0, share=0.5)
        self.assertEqual(int(keep.sum()), B); self.assertTrue(bool(keep[:, :, :2].all()))
        # 각 그룹에서 모든 군집이 최소 1개는 대표됨 (예산 12/그룹, share 0.5 → 6 개를 4 군집에 배분)
        for l in range(2):
            for h in range(2):
                covered = set(self.clusters[keep[l, h]].tolist())
                self.assertEqual(covered, set(range(4)))
        # share=0 이면 순수 K farthest-point 와 같은 그룹별 절차 (군집 무시)
        k0 = S.cluster_quota_select(self.keys, self.clusters, self.prot, B, 0, share=0.0)
        self.assertEqual(int(k0.sum()), B)
        # score 를 주면 군집 안 상위 점수 선택
        score = torch.rand(self.L, self.H, self.T)
        ks = S.cluster_quota_select(self.keys, self.clusters, self.prot, B, 0, share=0.5, score=score)
        self.assertEqual(int(ks.sum()), B)
        with self.assertRaises(ValueError):
            S.cluster_quota_select(self.keys, self.clusters, self.prot, B, 0, share=1.5)
