import inspect
import json
import unittest

import torch

from vlm_diagnosis.core.recurrent_importance import (
    MultimodalImportance,
    RecurrentImportance,
)


def selected(mask):
    return set(mask.nonzero(as_tuple=True)[0].tolist())


class MultimodalImportanceTest(unittest.TestCase):
    def test_global_selection_has_no_modality_quota_and_reports_actual_text(self):
        state = MultimodalImportance(
            prior_scores=torch.tensor([0.0, 1.0, 0.9, 0.8, 0.7]),
            modality_ids=torch.tensor([0, 3, 3, 2, 1]),
            budget=3,
            protected=torch.tensor([True, False, False, False, False]),
        )
        keep, diag = state.select()
        self.assertEqual(selected(keep), {0, 1, 2})
        self.assertEqual(diag["tokens_by_modality"]["control"], 1)
        self.assertEqual(diag["tokens_by_modality"]["text"], 1)
        self.assertEqual(diag["tokens_by_modality"]["image"], 1)
        self.assertEqual(diag["tokens_by_modality"]["audio"], 2)
        self.assertEqual(diag["tokens_by_modality"]["video"], 0)
        self.assertEqual(diag["selected_tokens_by_modality"]["control"], 1)
        self.assertEqual(diag["selected_tokens_by_modality"]["audio"], 2)
        self.assertEqual(diag["selected_tokens_by_modality"]["text"], 0)
        self.assertEqual(diag["selected_tokens_by_modality"]["image"], 0)
        # Generic text is the actual text modality, not all non-image tokens.
        self.assertEqual(diag["selected_text_tokens"], 0)
        self.assertEqual(diag["selected_image_tokens"], 0)
        self.assertEqual(diag["prior_weight"], diag["image_weight"])
        self.assertEqual(diag["prior_floor"], diag["image_floor"])
        self.assertEqual(diag["prior_scale"], diag["image_prior_scale"])
        self.assertEqual(diag["state_bytes"], 5 * 18)

        ids_copy, priors_copy = state.modality_ids, state.prior_scores
        ids_copy.zero_(); priors_copy.zero_()
        self.assertEqual(state.modality_ids.tolist(), [0, 3, 3, 2, 1])
        self.assertGreater(float(state.prior_scores.sum()), 0.0)

    def test_append_update_retain_multiple_modalities_without_resurrection(self):
        state = MultimodalImportance(
            torch.tensor([4.0, 3.0, 2.0, 1.0]),
            torch.tensor([2, 3, 1, 5]),
            budget=2,
            prior_floor=0.0,
        )
        original = state.prior_scores
        state.append(
            2,
            modality_ids=torch.tensor([3, 1]),
            prior_scores=torch.tensor([2.0, 0.0]),
        )
        self.assertTrue(torch.allclose(state.prior_scores[:4], original))
        self.assertTrue(torch.allclose(
            state.prior_scores, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.5, 0.0])
        ))
        self.assertEqual(state.prior_scale, 4.0)

        diag = state.update(
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            torch.ones(6, dtype=torch.bool),
        )
        keep, _ = state.select()
        self.assertEqual(selected(keep), {0, 3})
        self.assertEqual(diag["selected_tokens_by_modality"]["sensor"], 1)
        state.retain(keep)
        self.assertEqual(state.modality_ids.tolist(), [2, 5])
        self.assertTrue(torch.allclose(state.prior_scores, torch.tensor([1.0, 0.25])))
        self.assertEqual(state.select()[1]["state_bytes"], 2 * 18)

        # Only newly appended slots exist after physical deletion. The old
        # audio/text priors cannot reappear in this object's rebased state.
        state.append(2, modality_ids=torch.tensor([1, 3]))
        self.assertEqual(state.modality_ids.tolist(), [2, 5, 1, 3])
        self.assertTrue(torch.allclose(
            state.prior_scores, torch.tensor([1.0, 0.25, 0.0, 0.0])
        ))

    def test_custom_vocabulary_unknown_ids_and_append_fail_atomically(self):
        with self.assertRaises(ValueError):
            MultimodalImportance(torch.ones(1), torch.tensor([8]), 1)
        custom = MultimodalImportance(
            torch.tensor([2.0, 1.0]), torch.tensor([8, 9]), 1,
            modality_names={8: "depth", 9: "thermal"},
        )
        self.assertEqual(custom.modality_names, {8: "depth", 9: "thermal"})
        self.assertEqual(custom.select()[1]["tokens_by_modality"],
                         {"depth": 1, "thermal": 1})
        for names in (
            {True: "bad"}, {-1: "bad"}, {1: ""}, {1: " text"},
            {1: "same", 2: "same"}, [(1, "text")],
        ):
            with self.subTest(names=names), self.assertRaises(ValueError):
                MultimodalImportance(
                    torch.ones(1), torch.tensor([1]), 1, modality_names=names
                )
        with self.assertRaises(ValueError):
            MultimodalImportance(torch.ones(1), torch.tensor([1], dtype=torch.int32), 1)

        zero = MultimodalImportance(
            torch.zeros(2), torch.tensor([2, 1]), 1
        )
        baseline = (
            zero.prior_scores, zero.modality_ids, zero._history_state.clone(),
            zero.select()[1],
        )
        bad_appends = (
            dict(n_tokens=1, modality_ids=torch.tensor([99])),
            dict(n_tokens=2, modality_ids=torch.tensor([1])),
            dict(n_tokens=1, prior_scores=torch.tensor([1.0])),
            dict(n_tokens=1, prior_scores=torch.tensor([-1.0])),
        )
        for kwargs in bad_appends:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                zero.append(**kwargs)
            self.assertTrue(torch.equal(zero.prior_scores, baseline[0]))
            self.assertTrue(torch.equal(zero.modality_ids, baseline[1]))
            self.assertTrue(torch.equal(zero._history_state, baseline[2]))
            self.assertEqual(zero.select()[1], baseline[3])

    def test_custom_selector_receives_copies_and_invalid_update_is_atomic(self):
        snapshots = {}

        def custom_selector(*, scores, protected, budget, modality_ids):
            snapshots["scores"] = scores.clone()
            snapshots["modalities"] = modality_ids.clone()
            scores.zero_(); protected.zero_(); modality_ids.zero_()
            self.assertEqual(budget, 2)
            return torch.tensor([True, False, True])

        state = MultimodalImportance(
            torch.tensor([3.0, 2.0, 1.0]), torch.tensor([0, 2, 3]), 2,
            protected=torch.tensor([True, False, False]), selector=custom_selector,
        )
        keep, _ = state.select()
        self.assertEqual(selected(keep), {0, 2})
        self.assertTrue(torch.allclose(snapshots["scores"], torch.tensor([1.0, 2/3, 1/3])))
        self.assertEqual(snapshots["modalities"].tolist(), [0, 2, 3])
        self.assertEqual(state.modality_ids.tolist(), [0, 2, 3])
        self.assertTrue(torch.equal(state._protected, torch.tensor([True, False, False])))

        class InvalidOnSecondCall:
            def __init__(self):
                self.calls = 0

            def __call__(self, **_kwargs):
                self.calls += 1
                return (torch.tensor([True, False]) if self.calls == 1
                        else torch.tensor([False, False]))

        plugin = InvalidOnSecondCall()
        atomic = MultimodalImportance(
            torch.tensor([1.0, 0.0]), torch.tensor([2, 1]), 1,
            selector=plugin,
        )
        history_before = atomic._history_state.clone()
        observed_before = atomic._ever_observed.clone()
        with self.assertRaises(ValueError):
            atomic.update(torch.tensor([0.0, 1.0]), torch.ones(2, dtype=torch.bool))
        self.assertTrue(torch.equal(atomic._history_state, history_before))
        self.assertTrue(torch.equal(atomic._ever_observed, observed_before))
        self.assertEqual(atomic._observed_turns, 0)
        self.assertEqual(atomic._update_calls, 0)

        for invalid_result in (
            torch.tensor([1, 0]),
            torch.tensor([[True, False]]),
            torch.tensor([False, False]),
            torch.tensor([False, True]),
        ):
            invalid = MultimodalImportance(
                torch.ones(2), torch.tensor([0, 1]), 1,
                protected=torch.tensor([True, False]),
                selector=lambda **_kwargs: invalid_result,
            )
            with self.subTest(result=invalid_result), self.assertRaises(ValueError):
                invalid.select()

    def test_legacy_image_wrapper_matches_canonical_engine(self):
        scores = torch.tensor([4.0, 2.0, 1.0, 0.0])
        image = torch.tensor([True, True, False, False])
        protected = torch.tensor([True, False, False, False])
        legacy = RecurrentImportance(
            scores, image, 2, protected=protected, image_floor=0.2, decay=0.7
        )
        canonical = MultimodalImportance(
            scores, torch.tensor([2, 2, 1, 1]), 2,
            protected=protected, prior_floor=0.2, decay=0.7,
        )
        self.assertTrue(torch.equal(legacy._image_mask, image))
        self.assertTrue(torch.allclose(legacy._image_prior, canonical.prior_scores))
        self.assertTrue(torch.equal(legacy.select()[0], canonical.select()[0]))
        legacy.append(1); canonical.append(1)
        attention = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        observed = torch.ones(5, dtype=torch.bool)
        legacy_diag = legacy.update(attention, observed)
        canonical_diag = canonical.update(attention, observed)
        self.assertTrue(torch.equal(legacy.select()[0], canonical.select()[0]))
        for key in (
            "selected_image_tokens", "selected_text_tokens", "image_weight",
            "history_weight", "image_prior_scale", "history_l1", "turnover_count",
        ):
            self.assertEqual(legacy_diag[key], canonical_diag[key])


class RecurrentImportanceTest(unittest.TestCase):
    def test_dynamic_shift_forget_and_image_anchor(self):
        state = RecurrentImportance(
            image_score=torch.tensor([1.0, 0.1, 0.0, 0.0]),
            image_mask=torch.tensor([True, True, False, False]),
            budget=2,
            image_floor=0.35,
            decay=0.5,
        )
        keep0, diag0 = state.select()
        self.assertEqual(selected(keep0), {0, 1})
        self.assertEqual(diag0["history_weight"], 0.0)

        diag1 = state.update(
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            torch.ones(4, dtype=torch.bool),
        )
        keep1, _ = state.select()
        self.assertEqual(selected(keep1), {0, 2})
        self.assertEqual(diag1["turnover_count"], 1)
        self.assertEqual(diag1["selected_image_tokens"], 1)
        self.assertEqual(diag1["selected_text_tokens"], 1)

        # Token 2 receives an observed zero and is forgotten; new evidence at
        # token 3 takes its place.  Strong image token 0 remains as the anchor.
        diag2 = state.update(
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.ones(4, dtype=torch.bool),
        )
        keep2, _ = state.select()
        self.assertEqual(selected(keep2), {0, 3})
        self.assertEqual(diag2["turnover_count"], 1)
        self.assertGreaterEqual(diag2["image_weight"], state.image_floor)

    def test_fixed_budget_protected_tokens_and_append(self):
        protected = torch.tensor([True, False, False])
        state = RecurrentImportance(
            torch.tensor([1.0, 0.1, 0.0]),
            torch.tensor([True, True, False]),
            budget=2,
            protected=protected,
        )
        before, before_diag = state.select()
        bytes_before = before_diag["state_bytes"]
        scale_before = before_diag["image_prior_scale"]
        self.assertEqual(int(before.sum()), 2)
        self.assertTrue(bool(before[0]))

        state.append(2)
        after, after_diag = state.select()
        self.assertEqual(tuple(after.shape), (5,))
        self.assertEqual(int(after.sum()), 2)
        self.assertTrue(bool(after[0]))
        self.assertEqual(selected(before), selected(after[:3]))
        self.assertEqual(after_diag["state_bytes"] - bytes_before, 36)
        self.assertEqual(after_diag["image_prior_scale"], scale_before)

        # Appended tokens have no image prior, but completed interaction
        # evidence can move one into the fixed-size working set.
        state.update(
            torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
            torch.ones(5, dtype=torch.bool),
        )
        moved, moved_diag = state.select()
        self.assertEqual(int(moved.sum()), 2)
        self.assertEqual(selected(moved), {0, 4})
        self.assertEqual(moved_diag["protected_count"], 1)

    def test_unobserved_is_not_observed_zero(self):
        kwargs = dict(
            image_score=torch.tensor([1.0, 0.0]),
            image_mask=torch.tensor([True, False]),
            budget=1,
            image_floor=0.0,
            decay=0.5,
        )
        unknown = RecurrentImportance(**kwargs)
        known_zero = RecurrentImportance(**kwargs)
        evidence = torch.tensor([0.0, 1.0])
        both = torch.tensor([True, True])
        unknown.update(evidence, both)
        known_zero.update(evidence, both)

        d_unknown = unknown.update(
            torch.zeros(2), torch.tensor([False, False])
        )
        d_zero = known_zero.update(
            torch.zeros(2), torch.tensor([False, True])
        )
        self.assertEqual(d_unknown["history_l1"], 1.0)
        self.assertEqual(d_unknown["observed_turns"], 1)
        self.assertEqual(d_zero["history_l1"], 0.5)
        self.assertEqual(d_zero["observed_turns"], 2)

    def test_zero_evidence_does_not_fabricate_history(self):
        state = RecurrentImportance(
            torch.tensor([1.0, 0.5, 0.0]),
            torch.tensor([True, True, False]),
            budget=2,
        )
        diag = state.update(torch.zeros(3), torch.ones(3, dtype=torch.bool))
        self.assertEqual(diag["positive_evidence_count"], 0)
        self.assertEqual(diag["history_nonzero_tokens"], 0)
        self.assertEqual(diag["history_l1"], 0.0)

    def test_protected_attention_sink_does_not_compress_candidate_scale(self):
        state = RecurrentImportance(
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([False, True, False]),
            budget=2,
            protected=torch.tensor([True, False, False]),
            image_floor=0.0,
        )
        attention = torch.tensor([1000.0, 0.0, 1.0])
        observed = torch.ones(3, dtype=torch.bool)
        state.update(attention, observed)
        diag = state.update(attention, observed)
        keep, _ = state.select()
        self.assertEqual(selected(keep), {0, 2})
        self.assertEqual(diag["evidence_scale"], 1.0)
        self.assertEqual(diag["observed_protected_count"], 1)
        self.assertEqual(diag["updated_unprotected_count"], 2)

    def test_deterministic_ties_select_is_pure_and_json_safe(self):
        state = RecurrentImportance(
            torch.ones(5), torch.tensor([True, True, False, False, False]), budget=3
        )
        keep1, diag1 = state.select()
        keep1[4] = True  # returned masks do not alias persistent state
        keep2, diag2 = state.select()
        self.assertEqual(selected(keep2), {0, 1, 2})
        self.assertEqual(diag1, diag2)
        json.dumps(diag2)

    def test_sessions_and_constructor_inputs_are_isolated(self):
        scores = torch.tensor([1.0, 0.0, 0.0])
        mask = torch.tensor([True, False, False])
        left = RecurrentImportance(scores, mask, budget=1, image_floor=0.0)
        right = RecurrentImportance(scores, mask, budget=1, image_floor=0.0)
        scores.zero_()
        mask.zero_()
        left.update(torch.tensor([0.0, 1.0, 0.0]), torch.ones(3, dtype=torch.bool))
        left.update(torch.tensor([0.0, 1.0, 0.0]), torch.ones(3, dtype=torch.bool))
        self.assertEqual(selected(left.select()[0]), {1})
        self.assertEqual(selected(right.select()[0]), {0})

    def test_retain_is_physical_preserves_values_and_last_turnover(self):
        state = RecurrentImportance(
            torch.tensor([4.0, 2.0, 1.0, 0.0]),
            torch.tensor([True, True, False, False]),
            budget=2,
            protected=torch.tensor([True, False, False, False]),
            image_floor=0.0,
        )
        update = state.update(
            torch.tensor([1000.0, 0.0, 1.0, 0.0]),
            torch.ones(4, dtype=torch.bool),
        )
        keep, _ = state.select()
        self.assertEqual(selected(keep), {0, 2})
        self.assertEqual(update["turnover_count"], 1)
        old_tensors = [state._image_prior, state._history_state, state._modality_ids,
                       state._protected, state._ever_observed]
        old_scale = state.select()[1]["image_prior_scale"]

        state.retain(keep)
        after, diag = state.select()
        self.assertEqual(after.tolist(), [True, True])
        self.assertEqual(diag["token_count"], 2)
        self.assertEqual(diag["state_bytes"], 36)
        self.assertEqual(diag["image_prior_scale"], old_scale)
        self.assertEqual(diag["turnover_count"], 1)
        self.assertEqual(diag["observed_turns"], 1)
        self.assertEqual(diag["update_calls"], 1)
        self.assertEqual(state.budget, 2)
        self.assertTrue(torch.allclose(state._image_prior, torch.tensor([1.0, 0.25])))
        self.assertTrue(torch.allclose(state._history_state, torch.tensor([0.0, 1.0])))
        new_tensors = [state._image_prior, state._history_state, state._modality_ids,
                       state._protected, state._ever_observed]
        for old, new in zip(old_tensors, new_tensors):
            self.assertNotEqual(old.data_ptr(), new.data_ptr())
            self.assertEqual(new.untyped_storage().nbytes(), new.numel() * new.element_size())
        old_tensors[0][0] = 99
        self.assertEqual(float(state._image_prior[0]), 1.0)

    def test_retain_then_append_is_bounded_and_deleted_priors_do_not_return(self):
        state = RecurrentImportance(
            torch.tensor([4.0, 3.0, 2.0, 1.0]),
            torch.tensor([True, True, True, True]),
            budget=2,
            image_floor=0.0,
        )
        keep, before = state.select()
        self.assertEqual(before["state_bytes"], 72)
        state.retain(keep)
        self.assertTrue(torch.allclose(state._image_prior, torch.tensor([1.0, 0.75])))
        self.assertEqual(state.select()[1]["state_bytes"], 36)

        state.append(2)
        self.assertTrue(torch.allclose(
            state._image_prior, torch.tensor([1.0, 0.75, 0.0, 0.0])))
        state.update(torch.tensor([0.0, 0.0, 0.0, 1.0]), torch.ones(4, dtype=torch.bool))
        next_keep, _ = state.select()
        self.assertEqual(selected(next_keep), {0, 3})
        state.retain(next_keep)
        self.assertEqual(state.n_tokens, state.budget)
        self.assertEqual(state.select()[1]["state_bytes"], 36)
        self.assertTrue(torch.allclose(state._image_prior, torch.tensor([1.0, 0.0])))

    def test_retain_validation_is_atomic(self):
        state = RecurrentImportance(
            torch.tensor([3.0, 2.0, 1.0]),
            torch.tensor([True, True, False]),
            budget=2,
            protected=torch.tensor([True, False, False]),
        )
        baseline_tensors = [tensor.clone() for tensor in (
            state._image_prior, state._history_state, state._modality_ids,
            state._protected, state._ever_observed)]
        baseline_diag = state.select()[1]
        invalid = (
            torch.tensor([1, 1, 0]),
            torch.tensor([[True, True, False]]),
            torch.tensor([True, True]),
            torch.tensor([True, False, False]),
            torch.tensor([False, True, True]),
        )
        for keep in invalid:
            with self.assertRaises(ValueError):
                state.retain(keep)
            self.assertEqual(state.select()[1], baseline_diag)
            for expected, actual in zip(baseline_tensors, (
                    state._image_prior, state._history_state, state._modality_ids,
                    state._protected, state._ever_observed)):
                self.assertTrue(torch.equal(expected, actual))

    def test_api_has_no_future_question_input(self):
        self.assertEqual(
            list(inspect.signature(MultimodalImportance.select).parameters), ["self"]
        )
        self.assertEqual(
            list(inspect.signature(MultimodalImportance.update).parameters),
            ["self", "attention_mass", "observed"],
        )
        self.assertEqual(
            list(inspect.signature(MultimodalImportance.append).parameters),
            ["self", "n_tokens", "modality_ids", "prior_scores"],
        )
        self.assertEqual(
            list(inspect.signature(RecurrentImportance.select).parameters), ["self"]
        )
        self.assertEqual(
            list(inspect.signature(RecurrentImportance.update).parameters),
            ["self", "attention_mass", "observed"],
        )
        self.assertEqual(
            list(inspect.signature(RecurrentImportance.append).parameters),
            ["self", "n_tokens"],
        )
        self.assertEqual(
            list(inspect.signature(RecurrentImportance.retain).parameters),
            ["self", "keep"],
        )

    def test_validation_and_failed_update_are_atomic(self):
        valid_mask = torch.tensor([True, False])
        for invalid in (
            torch.tensor([float("nan"), 0.0]),
            torch.tensor([float("inf"), 0.0]),
            torch.tensor([-0.1, 0.0]),
            torch.ones(1, 2),
        ):
            with self.assertRaises(ValueError):
                RecurrentImportance(invalid, valid_mask, 1)
        with self.assertRaises(ValueError):
            RecurrentImportance(torch.ones(2), torch.tensor([1, 0]), 1)
        with self.assertRaises(ValueError):
            RecurrentImportance(torch.ones(2), valid_mask, 3)
        with self.assertRaises(ValueError):
            RecurrentImportance(
                torch.ones(2), valid_mask, 1, protected=torch.tensor([True, True])
            )
        with self.assertRaises(ValueError):
            RecurrentImportance(torch.ones(2), valid_mask, 1, image_floor=1.1)
        with self.assertRaises(ValueError):
            RecurrentImportance(torch.ones(2), valid_mask, 1, decay=float("nan"))
        with self.assertRaises(ValueError):
            MultimodalImportance(torch.ones(2), torch.tensor([2, 1]), 1,
                                 selector="not-callable")

        state = RecurrentImportance(torch.ones(2), valid_mask, 1)
        baseline_mask, baseline_diag = state.select()
        for attention, observed in (
            (torch.tensor([1.0]), torch.tensor([True, False])),
            (torch.tensor([1.0, -1.0]), torch.tensor([True, False])),
            (torch.tensor([1.0, float("inf")]), torch.tensor([True, False])),
            (torch.ones(1, 2), torch.tensor([True, False])),
            (torch.ones(2), torch.tensor([1, 0])),
        ):
            with self.assertRaises(ValueError):
                state.update(attention, observed)
            after_mask, after_diag = state.select()
            self.assertTrue(torch.equal(after_mask, baseline_mask))
            self.assertEqual(after_diag, baseline_diag)
        with self.assertRaises(ValueError):
            state.append(-1)
        with self.assertRaises(ValueError):
            state.append(1.5)


if __name__ == "__main__":
    unittest.main()
