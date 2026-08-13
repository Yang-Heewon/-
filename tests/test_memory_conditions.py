import unittest

from vlm_diagnosis.core.memory_conditions import (
    MemoryCondition,
    generate_conditions,
    payload_atoms_for_label,
    position_mode,
)


class MemoryConditionTest(unittest.TestCase):
    def test_answer_aware_is_diagnostic_only(self):
        condition = MemoryCondition(
            payload=("K_v",),
            build_mode="BR_ANSWER_PROBE",
            read_mode="R_INJECT_K",
            position="same_sequence_same_offset",
            blocks="single",
        )
        self.assertTrue(condition.valid)
        self.assertTrue(condition.diagnostic_only)

    def test_read_query_probe_is_diagnostic_only(self):
        condition = MemoryCondition(
            payload=("K_v",),
            build_mode="BR_QUERY",
            read_mode="R_INJECT_K",
            position="same_sequence_same_offset",
            blocks="single",
        )
        self.assertTrue(condition.valid)
        self.assertTrue(condition.diagnostic_only)

    def test_config_label_crosswalk(self):
        self.assertEqual(payload_atoms_for_label("IMAGE+T_visual"), ("I", "T_o", "T_d", "T_u"))
        self.assertEqual(payload_atoms_for_label("FULL_KV"), ("K_p", "K_v", "Z"))
        self.assertEqual(
            position_mode(write_offset=0, read_offset=128, context_changed=True),
            "context_and_offset_change",
        )

    def test_episode_answer_is_labeled_as_carryover(self):
        condition = MemoryCondition(
            payload=("T_q", "T_a", "T_out"),
            build_mode="BW_ANSWER",
            read_mode="R_PREFILL_T",
            position="same_sequence_same_offset",
            blocks="single",
        )
        self.assertTrue(condition.valid)
        self.assertTrue(condition.answer_carryover)

    def test_image_text_condition_is_valid(self):
        condition = MemoryCondition(
            payload=("I", "T_o"),
            build_mode="BW_QUERY",
            read_mode="R_PREFILL_I_THEN_T",
            position="same_offset_context_change",
            blocks="single",
        )
        self.assertTrue(condition.valid, condition.validation_errors())

    def test_missing_payload_dependency_is_rejected(self):
        condition = MemoryCondition(
            payload=("T_o",),
            build_mode="B0_GENERIC",
            read_mode="R_PREFILL_I_THEN_T",
            position="same_sequence_same_offset",
            blocks="single",
        )
        self.assertFalse(condition.valid)
        self.assertIn("image prefill requires I", condition.validation_errors())

    def test_no_memory_has_one_canonical_build_label(self):
        condition = MemoryCondition(
            payload=(),
            build_mode="BR_ANSWER_PROBE",
            read_mode="R_NONE",
            position="same_sequence_same_offset",
            blocks="single",
        )
        self.assertFalse(condition.valid)

    def test_moved_kv_requires_position_metadata(self):
        condition = MemoryCondition(
            payload=("K_v",),
            build_mode="BW_QUERY",
            read_mode="R_INJECT_K",
            position="offset_shift",
            blocks="single",
        )
        self.assertFalse(condition.valid)
        self.assertIn("moved/composed KV requires Z metadata", condition.validation_errors())

    def test_irrelevant_block_composition_requires_position_metadata(self):
        condition = MemoryCondition(
            payload=("K_v",),
            build_mode="BW_QUERY",
            read_mode="R_INJECT_K",
            position="same_sequence_same_offset",
            blocks="single",
            interference="relevant_plus_irrelevant",
        )
        self.assertFalse(condition.valid)
        self.assertIn("moved/composed KV requires Z metadata", condition.validation_errors())

    def test_core_registry_has_unique_valid_conditions_and_required_families(self):
        conditions = list(generate_conditions(scope="core"))
        ids = {condition.condition_id for condition in conditions}
        self.assertEqual(len(ids), len(conditions))
        self.assertTrue(all(condition.valid for condition in conditions))

        payloads = {condition.payload for condition in conditions}
        self.assertIn(("I",), payloads)
        self.assertIn(("I", "T_o", "T_d", "T_u"), payloads)
        self.assertIn(("K_p", "K_v"), payloads)
        self.assertIn(("T_o", "T_d", "T_u"), payloads)
        self.assertIn(("T_q", "T_a", "T_out", "T_traj"), payloads)
        self.assertIn(("I", "T_o", "T_d", "T_u", "T_q", "T_a", "T_out", "T_traj", "K_p", "K_v"), payloads)


if __name__ == "__main__":
    unittest.main()
