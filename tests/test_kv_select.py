import unittest

import torch

from vlm_diagnosis.core.kv_select import (
    build_eviction_mask, select_dual_prefill_tokens,
    select_dual_prefill_triples, select_triples, select_tokens,
    uniform_token_keep, kept_composition, per_head_column_stats, kv_bytes,
    index_bytes)


class SelectTriplesTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.L, self.H, self.P = 4, 2, 50
        self.core = torch.rand(self.L, self.H, self.P, generator=g)
        self.query = torch.rand(self.L, self.H, self.P, generator=g)
        self.core[:, :, 40:] = float("-inf")      # 질문 열: core 정의 안 됨

    def test_exact_budget_and_forced_counted(self):
        for T in [0, 1, 37, 200, 400, 1000]:
            for al in [0, 0.25, 1.0]:
                keep, sel = select_triples(self.core, self.query, T, al)
                self.assertEqual(int(keep.sum()), min(T, self.core.numel()))
                self.assertEqual(sel.kept_triples, int(keep.sum()))
        forced = torch.zeros(self.L, self.H, self.P, dtype=torch.bool)
        forced[:, :, :4] = True                    # sink 보호 = 32 세 짝
        keep, sel = select_triples(self.core, self.query, 100, 0.5, forced)
        self.assertEqual(int(keep.sum()), 100)
        self.assertTrue(bool(keep[:, :, :4].all()))
        self.assertEqual(sel.forced_triples, 32)
        self.assertEqual(sel.core_count + sel.query_count, 68)

    def test_alpha_one_never_picks_undefined_core(self):
        keep, sel = select_triples(self.core, self.query, 300, 1.0)
        # core가 -inf인 열(40..)은 정의된 320개가 남아 있는 한 선택되지 않음
        self.assertEqual(int(keep[:, :, 40:].sum()), 0)
        self.assertEqual(int(keep.sum()), 300)

    def test_alpha_zero_matches_query_topk(self):
        keep, _ = select_triples(self.core, self.query, 25, 0.0)
        top = set(torch.topk(self.query.flatten(), 25).indices.tolist())
        self.assertEqual(set(keep.flatten().nonzero(as_tuple=True)[0].tolist()), top)

    def test_token_granularity_replicates_across_heads(self):
        ct, qt = self.core.amax(dim=(0, 1)), self.query.mean(dim=(0, 1))
        keep, sel = select_tokens(ct, qt, 10, 0.5, self.L, self.H)
        self.assertEqual(int(keep.sum()), 10 * self.L * self.H)
        self.assertTrue(torch.equal(keep[0, 0], keep[3, 1]))
        self.assertEqual(sel.granularity, "token")

    def test_uniform_token_keep(self):
        keep = uniform_token_keep(50, 5, self.L, self.H)
        self.assertEqual(int(keep[0, 0].sum()), 5)
        self.assertTrue(torch.equal(keep[0, 0], keep[2, 1]))
        forced = torch.zeros(50, dtype=torch.bool); forced[:4] = True
        keep = uniform_token_keep(50, 5, self.L, self.H, forced)
        self.assertEqual(int(keep[0, 0].sum()), 5)
        self.assertTrue(bool(keep[0, 0, :4].all()))

    def test_dual_prefill_head_selection_uses_shared_prefix_only_for_image(self):
        image = torch.zeros(self.L, self.H, self.P)
        joint = torch.zeros_like(image)
        image[1, 0, 2] = 100
        image[2, 1, 45] = 1000       # must be ignored: image-only has no text suffix
        joint[3, 1, 49] = 100
        eligible = torch.zeros_like(image, dtype=torch.bool)
        eligible[:, :, :40] = True
        keep, sel = select_dual_prefill_triples(
            image, joint, 2, 0.5, image_eligible=eligible)
        self.assertTrue(bool(keep[1, 0, 2]))
        self.assertTrue(bool(keep[3, 1, 49]))
        self.assertFalse(bool(keep[2, 1, 45]))
        self.assertEqual(int(keep.sum()), 2)
        self.assertEqual(sel.image_count, 1)
        self.assertEqual(sel.joint_count, 1)

    def test_dual_prefill_token_selection_replicates_and_counts_forced(self):
        image = torch.arange(self.P, dtype=torch.float32)
        joint = torch.arange(self.P, 0, -1, dtype=torch.float32)
        forced = torch.zeros(self.P, dtype=torch.bool)
        forced[:2] = True
        eligible = torch.zeros(self.P, dtype=torch.bool)
        eligible[:40] = True
        keep, sel = select_dual_prefill_tokens(
            image, joint, 6, 0.5, self.L, self.H,
            forced_tok=forced, image_eligible_tok=eligible)
        self.assertTrue(torch.equal(keep[0, 0], keep[-1, -1]))
        self.assertEqual(int(keep[0, 0].sum()), 6)
        self.assertTrue(bool(keep[:, :, :2].all()))
        self.assertEqual(sel.kept_triples, 6 * self.L * self.H)
        self.assertEqual(sel.forced_triples, 2 * self.L * self.H)

    def test_dual_prefill_rejects_forced_set_larger_than_budget(self):
        forced = torch.zeros_like(self.core, dtype=torch.bool)
        forced[:, :, :4] = True
        with self.assertRaises(ValueError):
            select_dual_prefill_triples(
                self.core, self.query, 10, 0.5, forced=forced)


class EvictionMaskTest(unittest.TestCase):
    def test_prefill_rows_gated_and_heads_grouped(self):
        Lq = Lk = 8
        neg = torch.finfo(torch.float16).min
        base = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
        base[0, 0].masked_fill_(torch.triu(torch.ones(Lq, Lk, dtype=torch.bool), 1), neg)
        ev = torch.zeros(2, 8, dtype=torch.bool)      # H_kv=2
        ev[0, 2] = True                                # kv head 0: 열 2 차단
        ev[1, 5] = True                                # kv head 1: 열 5 차단
        m = build_eviction_mask(base, ev, row_start=6, groups=3)   # Hq = 6
        self.assertEqual(tuple(m.shape), (1, 6, Lq, Lk))
        # 행 < row_start 는 원본 인과 마스크 그대로
        self.assertTrue(torch.equal(m[0, :, :6], base[0, 0, :6].expand(6, 6, Lk)))
        # 행 >= row_start: query head 0..2 (kv 0) 는 열 2 차단, 3..5 (kv 1) 는 열 5 차단
        for h in range(3):
            self.assertEqual(float(m[0, h, 7, 2]), neg)
            self.assertEqual(float(m[0, h, 7, 5]), 0.0)
        for h in range(3, 6):
            self.assertEqual(float(m[0, h, 7, 5]), neg)
            self.assertEqual(float(m[0, h, 7, 2]), 0.0)
        # 자기 자신(대각)은 차단되지 않음
        self.assertEqual(float(m[0, 0, 7, 7]), 0.0)

    def test_decode_row_always_blocked_and_generated_cols_free(self):
        Lk = 12                                         # P=8 프롬프트 + 4 생성
        base = torch.zeros(1, 1, 1, Lk, dtype=torch.float16)
        ev = torch.zeros(1, 8, dtype=torch.bool); ev[0, 1] = True
        m = build_eviction_mask(base, ev, row_start=8, groups=4)
        neg = torch.finfo(torch.float16).min
        self.assertTrue(all(float(m[0, h, 0, 1]) == neg for h in range(4)))
        self.assertEqual(float(m[0, 0, 0, 9]), 0.0)     # 생성 열은 자유


class StatsTest(unittest.TestCase):
    def test_per_head_stats_match_global_aggregates(self):
        torch.manual_seed(0)
        L, Hq, Hkv, d = 10, 4, 2, 8
        qk = [(torch.randn(1, Hq, L, d), torch.randn(1, Hkv, L, d)) for _ in range(3)]
        mean, peak = per_head_column_stats(qk, 6, 10)
        self.assertEqual(tuple(mean.shape), (3, Hkv, L))
        # 인과성: 행 6..9 만 query 이므로 열 > 9 없음, 열 0..9 모두 질량 가능; 열 합은 1
        self.assertTrue(torch.allclose(mean.sum(-1), torch.ones(3, Hkv), atol=1e-4))
        self.assertTrue(bool((peak <= 1).all()) and bool((peak >= 0).all()))
        # 참조 구현(attnstat.recv_column_mass)과 순위 동일
        from vlm_diagnosis.core.attnstat import recv_column_mass
        ref = recv_column_mass(qk, 6, 10)
        mine = mean.mean(dim=(0, 1))
        self.assertTrue(torch.equal(torch.argsort(ref), torch.argsort(mine)))

    def test_composition_and_bytes(self):
        keep = torch.zeros(2, 2, 10, dtype=torch.bool)
        keep[:, :, 3:7] = True
        vis = torch.tensor([4, 5, 6, 7, 8])
        c = kept_composition(keep, vis, n_sink=4)
        self.assertEqual(c["kept_triples"], 16)
        self.assertAlmostEqual(c["keep_frac_visual"], 12 / 16)
        self.assertAlmostEqual(c["sink_kept_frac"], 1 / 4)
        self.assertEqual(c["per_layer_kept"], [8, 8])
        self.assertEqual(kv_bytes(16, 128), 16 * 2 * 128 * 2)
        self.assertEqual(index_bytes(28, 4, 1300, "head"), (28 * 4 * 1300 + 7) // 8)
        self.assertEqual(index_bytes(28, 4, 1300, "token"), (1300 + 7) // 8)


if __name__ == "__main__":
    unittest.main()
