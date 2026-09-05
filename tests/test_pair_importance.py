import json
import unittest

import torch

from vlm_diagnosis.core.pair_importance import PairImportance


def _selected(mask):
    return set(mask.nonzero(as_tuple=True)[0].tolist())


class PairImportanceTest(unittest.TestCase):
    def test_constructor_flattens_layer_head_token_order_and_owns_inputs(self):
        priors = torch.arange(1, 13, dtype=torch.float32).reshape(2, 2, 3)
        modalities = torch.tensor([0, 2, 1])
        state = PairImportance(priors, modalities, budget_pairs=4, n_sink=0)

        self.assertEqual((state.n_layers, state.n_heads, state.groups), (2, 2, 4))
        self.assertEqual(state.n_pairs, 12)
        self.assertEqual(
            state.group_ids.tolist(),
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        )
        self.assertEqual(
            state.token_ids.tolist(),
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        )
        torch.testing.assert_close(
            state.engine.prior_scores, priors.reshape(-1) / 12
        )
        self.assertEqual(state.engine.modality_ids.tolist(), [0, 2, 1] * 4)
        self.assertEqual(state.nbytes, 12 * 34)

        priors.zero_()
        modalities.fill_(5)
        ids = state.group_ids
        ids.fill_(99)
        self.assertEqual(float(state.engine.prior_scores[-1]), 1.0)
        self.assertEqual(state.engine.modality_ids.tolist(), [0, 2, 1] * 4)
        self.assertEqual(int(state.group_ids.max()), 3)

        snapshot = state.snapshot()
        self.assertEqual(snapshot["pairs_by_layer_head"], [[3, 3], [3, 3]])
        self.assertEqual(snapshot["pairs_by_layer"], [6, 6])
        self.assertEqual(snapshot["pairs_by_head"], [6, 6])
        self.assertEqual(snapshot["distinct_logical_tokens"], 3)
        self.assertEqual(
            snapshot["modality_pair_counts"],
            {"control": 4, "text": 4, "image": 4,
             "audio": 0, "video": 0, "sensor": 0},
        )
        json.dumps(snapshot)

    def test_global_budget_allows_unequal_groups_and_partial_token_retention(self):
        priors = torch.tensor(
            [
                [[0.0, 10.0, 0.0], [0.0, 9.0, 0.0]],
                [[0.0, 0.0, 8.0], [7.0, 0.0, 0.0]],
            ]
        )
        state = PairImportance(
            priors, torch.tensor([1, 1, 2]), budget_pairs=3, n_sink=0
        )
        keep, diagnostics = state.select()

        self.assertEqual(_selected(keep), {1, 4, 8})
        selected = state.selected_ids(keep)
        self.assertEqual([ids.tolist() for ids in selected], [[1], [1], [2], []])
        self.assertEqual(diagnostics["pairs_by_group"], [1, 1, 1, 0])
        self.assertEqual(diagnostics["pairs_by_layer"], [2, 1])
        self.assertEqual(diagnostics["pairs_by_head"], [2, 1])
        self.assertEqual(diagnostics["distinct_logical_tokens"], 2)
        self.assertEqual(diagnostics["counted_pairs"], 3)
        self.assertEqual(diagnostics["state_bytes"], 12 * 34)
        self.assertEqual(diagnostics["importance_state_bytes"], 12 * 18)

    def test_sink_pairs_are_protected_inside_the_same_global_budget(self):
        priors = torch.tensor([[[0.0, 1.0, 0.9], [0.0, 0.8, 0.7]]])
        state = PairImportance(
            priors, torch.tensor([0, 2, 1]), budget_pairs=3, n_sink=1
        )
        keep, diagnostics = state.select()
        # Pair 0 and pair 3 are the per-head sink; only one ranked slot remains.
        self.assertEqual(_selected(keep), {0, 1, 3})
        self.assertEqual(diagnostics["protected_count"], 2)
        self.assertEqual(diagnostics["selected_protected_tokens"], 2)
        self.assertEqual([x.tolist() for x in state.selected_ids(keep)], [[0, 1], [0]])

        with self.assertRaises(ValueError):
            PairImportance(
                priors, torch.tensor([0, 2, 1]), budget_pairs=1, n_sink=1
            )

    def test_retain_is_physical_then_append_never_revives_or_reuses_ids(self):
        priors = torch.tensor([[[9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 8.0, 7.0]]])
        modalities = torch.tensor([0, 1, 2, 1])
        state = PairImportance(priors, modalities, budget_pairs=3, n_sink=0)
        keep, _ = state.select()
        self.assertEqual([x.tolist() for x in state.selected_ids(keep)], [[0], [2, 3]])

        old_group = state._group_ids
        old_token = state._token_ids
        old_core = (
            state.engine._prior_scores,
            state.engine._history_state,
            state.engine._modality_ids,
            state.engine._protected,
            state.engine._ever_observed,
        )
        state.retain(keep)
        self.assertEqual(state.n_pairs, 3)
        self.assertEqual(state.nbytes, 3 * 34)
        self.assertEqual(state.group_ids.tolist(), [0, 1, 1])
        self.assertEqual(state.token_ids.tolist(), [0, 2, 3])
        self.assertNotEqual(old_group.data_ptr(), state._group_ids.data_ptr())
        self.assertNotEqual(old_token.data_ptr(), state._token_ids.data_ptr())
        for old, new in zip(
            old_core,
            (
                state.engine._prior_scores,
                state.engine._history_state,
                state.engine._modality_ids,
                state.engine._protected,
                state.engine._ever_observed,
            ),
        ):
            self.assertNotEqual(old.data_ptr(), new.data_ptr())
            self.assertEqual(new.untyped_storage().nbytes(), new.numel() * new.element_size())

        appended_modalities = torch.tensor([1, 2])
        appended_priors = torch.tensor([[[10.0, 0.0], [0.0, 6.0]]])
        state.append(appended_modalities, start_token=4, prior_scores=appended_priors)
        appended_modalities.zero_()
        appended_priors.zero_()
        self.assertEqual(state.group_ids.tolist(), [0, 1, 1, 0, 0, 1, 1])
        self.assertEqual(state.token_ids.tolist(), [0, 2, 3, 4, 5, 4, 5])
        self.assertEqual(state.engine.modality_ids.tolist(), [0, 2, 1, 1, 2, 1, 2])
        self.assertFalse(any(token == 1 for token in state.token_ids.tolist()))
        torch.testing.assert_close(
            state.engine.prior_scores[-4:], torch.tensor([10 / 9, 0.0, 0.0, 6 / 9])
        )

        before = (state.group_ids, state.token_ids, state.engine.prior_scores)
        with self.assertRaises(ValueError):
            state.append(torch.tensor([1]), start_token=5)
        self.assertTrue(torch.equal(state.group_ids, before[0]))
        self.assertTrue(torch.equal(state.token_ids, before[1]))
        self.assertTrue(torch.equal(state.engine.prior_scores, before[2]))

        # Gaps are legal global IDs, but the high-water mark advances and is
        # never derived from whichever low IDs happened to survive selection.
        state.append(torch.tensor([1]), start_token=10)
        self.assertEqual(state.token_ids[-2:].tolist(), [10, 10])
        with self.assertRaises(ValueError):
            state.append(torch.tensor([1]), start_token=6)

    def test_observe_maps_each_layer_head_vector_without_token_averaging(self):
        state = PairImportance(
            torch.zeros(1, 2, 3),
            torch.tensor([1, 1, 2]),
            budget_pairs=2,
            n_sink=0,
            prior_floor=0.0,
        )
        diagnostics = state.observe(
            [torch.tensor([0.0, 0.0, 10.0]), torch.tensor([9.0, 0.0, 0.0])]
        )
        torch.testing.assert_close(
            state.engine._history_state,
            torch.tensor([0.0, 0.0, 1.0, 0.9, 0.0, 0.0]),
        )
        keep, _ = state.select()
        self.assertEqual(_selected(keep), {2, 3})
        self.assertEqual([x.tolist() for x in state.selected_ids(keep)], [[2], [0]])
        self.assertEqual(diagnostics["pairs_by_group"], [1, 1])
        self.assertEqual(diagnostics["evidence_scale"], 10.0)

    def test_observe_accepts_empty_groups_after_global_pruning(self):
        state = PairImportance(
            torch.tensor([[[3.0, 2.0], [0.0, 0.0]]]),
            torch.tensor([1, 2]),
            budget_pairs=2,
            n_sink=0,
            prior_floor=0.0,
        )
        keep, _ = state.select()
        self.assertEqual([x.tolist() for x in state.selected_ids(keep)], [[0, 1], []])
        state.retain(keep)
        state.observe([torch.tensor([0.25, 1.0]), torch.empty(0)])
        torch.testing.assert_close(state.engine._history_state, torch.tensor([0.25, 1.0]))
        self.assertEqual(state.snapshot()["pairs_by_group"], [2, 0])

    def test_invalid_observation_and_append_are_atomic(self):
        state = PairImportance(
            torch.ones(1, 2, 2), torch.tensor([1, 2]), budget_pairs=2, n_sink=0
        )
        baseline = (
            state.engine._history_state.clone(),
            state.engine._ever_observed.clone(),
            state.group_ids,
            state.token_ids,
            state.engine.prior_scores,
        )
        invalid_masses = (
            [torch.ones(2)],
            [torch.ones(1), torch.ones(2)],
            [torch.tensor([1.0, -1.0]), torch.ones(2)],
            [torch.ones(2), torch.tensor([1.0, float("nan")])],
        )
        for masses in invalid_masses:
            with self.subTest(lengths=[x.numel() for x in masses]), self.assertRaises(ValueError):
                state.observe(masses)
            self.assertTrue(torch.equal(state.engine._history_state, baseline[0]))
            self.assertTrue(torch.equal(state.engine._ever_observed, baseline[1]))

        invalid_appends = (
            (torch.tensor([99]), 2, None),
            (torch.tensor([1]), 1, None),
            (torch.tensor([1]), 2, torch.ones(1, 1, 1)),
            (torch.tensor([1]), 2, torch.full((1, 2, 1), -1.0)),
        )
        for modalities, start, prior in invalid_appends:
            with self.subTest(start=start, prior=prior), self.assertRaises(ValueError):
                state.append(modalities, start, prior)
            self.assertTrue(torch.equal(state.group_ids, baseline[2]))
            self.assertTrue(torch.equal(state.token_ids, baseline[3]))
            self.assertTrue(torch.equal(state.engine.prior_scores, baseline[4]))

    def test_zero_budget_is_supported_only_without_protected_pairs(self):
        state = PairImportance(
            torch.ones(1, 2, 2), torch.tensor([1, 2]), budget_pairs=0, n_sink=0
        )
        keep, diagnostics = state.select()
        self.assertEqual(keep.tolist(), [False, False, False, False])
        self.assertEqual([ids.tolist() for ids in state.selected_ids()], [[], []])
        self.assertEqual(diagnostics["counted_pairs"], 0)
        state.retain(keep)
        self.assertEqual(state.n_pairs, 0)
        self.assertEqual(state.nbytes, 0)

        with self.assertRaises(ValueError):
            PairImportance(
                torch.ones(1, 2, 2), torch.tensor([1, 2]), budget_pairs=0, n_sink=1
            )


if __name__ == "__main__":
    unittest.main()
