import math
import unittest

import torch

from vlm_diagnosis.core.core_delta import (
    core_delta_keep, dual_prefill_union_keep, weighted_sum_keep, rank_normalize)


class CoreDeltaKeepTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.n = 200
        self.core = torch.rand(self.n, generator=g)
        self.query = torch.rand(self.n, generator=g)

    def _topk(self, s, k):
        return set(torch.topk(s, k).indices.tolist())

    def test_exact_budget_for_all_alphas(self):
        for B in [0, 1, 7, 50, 199, 200, 500]:
            for a in [0, 0.1, 0.25, 0.5, 0.75, 1.0]:
                keep, info = core_delta_keep(self.core, self.query, B, a)
                self.assertEqual(len(keep), min(B, self.n), (B, a))
                self.assertEqual(info.core_count + info.query_count, len(keep))
                self.assertEqual(info.core_count, round(a * min(B, self.n)))

    def test_alpha_extremes_match_topk(self):
        B = 30
        keep0, _ = core_delta_keep(self.core, self.query, B, 0.0)
        keep1, _ = core_delta_keep(self.core, self.query, B, 1.0)
        self.assertEqual(keep0, self._topk(self.query, B))
        self.assertEqual(keep1, self._topk(self.core, B))

    def test_core_protected_then_query_fills_rest(self):
        B, a = 40, 0.5
        keep, info = core_delta_keep(self.core, self.query, B, a)
        core_top = self._topk(self.core, 20)
        self.assertTrue(core_top <= keep)
        rest = keep - core_top
        # 나머지 20개는 core에 없는 것 중 query 상위 20개
        q = self.query.clone()
        q[list(core_top)] = -1
        self.assertEqual(rest, self._topk(q, 20))
        self.assertEqual(info.query_count, 20)

    def test_keep_zero_and_negative(self):
        keep, info = core_delta_keep(self.core, self.query, 0, 0.5)
        self.assertEqual(keep, set())
        keep, info = core_delta_keep(self.core, self.query, -3, 0.5)
        self.assertEqual(keep, set())

    def test_nan_scores_go_last(self):
        core = self.core.clone(); core[:5] = float("nan")
        query = self.query.clone(); query[:5] = -1.0        # query에서도 최하위
        query[5:10] = float("inf")
        query[10:15] = float("nan")
        keep, info = core_delta_keep(core, query, 100, 0.5)
        self.assertEqual(len(keep), 100)
        self.assertEqual(info.nan_core, 5)
        self.assertEqual(info.nan_query, 10)   # inf도 비유한값으로 최하위 처리
        # NaN core 위치(0..4)는 core 절반에도, query 절반에도 들어갈 수 없음
        self.assertTrue(set(range(5)).isdisjoint(keep))
        # 예산이 남아 -inf까지 내려가야 하는 경우엔 채워서 정확히 B를 맞춘다
        keep_all, _ = core_delta_keep(core, query, 200, 0.5)
        self.assertEqual(len(keep_all), 200)

    def test_ties_deterministic_by_index(self):
        core = torch.ones(50)
        query = torch.ones(50)
        keep, _ = core_delta_keep(core, query, 10, 0.5)
        self.assertEqual(keep, set(range(10)))
        keep2, _ = core_delta_keep(core, query, 10, 0.5)
        self.assertEqual(keep, keep2)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            core_delta_keep(self.core, self.query[:-1], 10, 0.5)
        with self.assertRaises(ValueError):
            core_delta_keep(self.core, self.query, 10, 1.5)

    def test_overlap_diagnostic(self):
        keep, info = core_delta_keep(self.core, self.core, 30, 0.5)
        self.assertEqual(info.core_query_overlap, 30)
        self.assertEqual(keep, self._topk(self.core, 30))

    def test_accepts_lists_and_numpy(self):
        keep, _ = core_delta_keep(self.core.tolist(), self.query.numpy(), 12, 0.25)
        self.assertEqual(len(keep), 12)


class WeightedSumTest(unittest.TestCase):
    def test_rank_normalize_range(self):
        r = rank_normalize(torch.tensor([3.0, 1.0, 2.0, float("nan")]))
        self.assertEqual(r.tolist(), [1.0, 1 / 3, 2 / 3, 0.0])

    def test_budget_and_extremes(self):
        g = torch.Generator().manual_seed(1)
        c, q = torch.rand(100, generator=g), torch.rand(100, generator=g)
        for w in [0.0, 0.3, 1.0]:
            self.assertEqual(len(weighted_sum_keep(c, q, 25, w)), 25)
        self.assertEqual(weighted_sum_keep(c, q, 25, 1.0),
                         set(torch.topk(c, 25).indices.tolist()))
        self.assertEqual(weighted_sum_keep(c, q, 25, 0.0),
                         set(torch.topk(q, 25).indices.tolist()))


class DualPrefillUnionTest(unittest.TestCase):
    def test_independent_union_deduplicates_and_backfills_exact_budget(self):
        image = torch.tensor([100, 90, 80, 70, 60, 50.])
        joint = torch.tensor([60, 100, 90, 80, 70, 50.])
        keep, info = dual_prefill_union_keep(image, joint, 4, image_quota=2)
        self.assertEqual(keep, {0, 1, 2, 3})
        self.assertEqual(info.initial_overlap, 1)
        self.assertEqual(info.joint_backfill, 1)
        self.assertEqual(info.image_count + info.joint_count, 4)

    def test_no_overlap_and_full_overlap(self):
        image = torch.tensor([6, 5, 4, 3, 2, 1.])
        joint = torch.tensor([1, 2, 3, 4, 6, 5.])
        keep, info = dual_prefill_union_keep(image, joint, 4, image_quota=2)
        self.assertEqual(keep, {0, 1, 4, 5})
        self.assertEqual(info.initial_overlap, 0)
        same, info = dual_prefill_union_keep(image, image, 4, image_quota=2)
        self.assertEqual(same, {0, 1, 2, 3})
        self.assertEqual(info.initial_overlap, 2)
        self.assertEqual(info.joint_backfill, 2)

    def test_endpoints_budget_clamp_and_ties(self):
        image = torch.tensor([1, 6, 2, 5, 3, 4.])
        joint = torch.tensor([6, 1, 5, 2, 4, 3.])
        joint_only, _ = dual_prefill_union_keep(image, joint, 3, image_quota=0)
        image_only, _ = dual_prefill_union_keep(image, joint, 3, image_quota=3)
        self.assertEqual(joint_only, {0, 2, 4})
        self.assertEqual(image_only, {1, 3, 5})
        empty, _ = dual_prefill_union_keep(image, joint, -1, image_quota=0)
        self.assertEqual(empty, set())
        all_items, _ = dual_prefill_union_keep(image, joint, 99, image_quota=6)
        self.assertEqual(all_items, set(range(6)))
        tied, _ = dual_prefill_union_keep(torch.ones(6), torch.ones(6), 4, image_quota=2)
        self.assertEqual(tied, {0, 1, 2, 3})

    def test_eligibility_prevents_image_branch_from_selecting_text_suffix(self):
        image = torch.tensor([3, 2, 1, 100, 99.])
        joint = torch.tensor([1, 2, 3, 5, 4.])
        eligible = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
        keep, info = dual_prefill_union_keep(
            image, joint, 4, image_quota=3, image_eligible=eligible)
        self.assertEqual(keep, {0, 1, 2, 3})
        self.assertEqual(info.image_count, 3)
        self.assertEqual(info.joint_count, 1)

    def test_invalid_scores_and_contract_errors(self):
        image = torch.tensor([5.0, 4.0, float("nan"), float("inf")])
        joint = torch.tensor([1.0, 2.0, 3.0, float("-inf")])
        keep, info = dual_prefill_union_keep(image, joint, 2, image_quota=1)
        self.assertEqual(len(keep), 2)
        self.assertEqual(info.nan_image, 2)
        self.assertEqual(info.nan_joint, 1)
        with self.assertRaises(ValueError):
            dual_prefill_union_keep(torch.ones(2, 2), torch.ones(4), 2, 1)
        with self.assertRaises(ValueError):
            dual_prefill_union_keep(torch.ones(4), torch.ones(4), 2, 3)
        with self.assertRaises(ValueError):
            dual_prefill_union_keep(
                torch.ones(4), torch.ones(4), 3, 1,
                image_eligible=torch.tensor([1, 0, 0, 0]),
                joint_eligible=torch.tensor([0, 1, 0, 0]),
            )


if __name__ == "__main__":
    unittest.main()
