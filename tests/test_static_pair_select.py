import unittest

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
        self.assertTrue(bool((sh == 2).all()))
        with self.assertRaises(ValueError):
            S.map_token_signal(sig, "d_same_zero0", L, H)
        with self.assertRaises(ValueError):
            S.map_token_signal(sig, "nope", L, H)
        std = S.map_token_signal(torch.arange(4.), "r_std_same", L, H)
        self.assertEqual(tuple(std.shape), (L, H, T))

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
