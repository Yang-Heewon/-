import unittest
from unittest.mock import patch

import torch

from vlm_diagnosis.exps.core_delta_full_kv import (
    align_image_prefill_score,
    image_prefill_stats,
)


class AlignImagePrefillScoreTest(unittest.TestCase):
    def test_prefix_is_mapped_and_text_suffix_is_ineligible(self):
        image = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
        aligned, eligible = align_image_prefill_score(image, prompt_len=5)
        self.assertEqual(tuple(aligned.shape), (2, 2, 5))
        self.assertTrue(torch.equal(aligned[..., :3], image))
        self.assertTrue(bool(torch.isneginf(aligned[..., 3:]).all()))
        self.assertTrue(bool(eligible[..., :3].all()))
        self.assertFalse(bool(eligible[..., 3:].any()))

    def test_invalid_shape_or_longer_image_prefix_fails_closed(self):
        with self.assertRaises(ValueError):
            align_image_prefill_score(torch.ones(2, 3), prompt_len=5)
        with self.assertRaises(ValueError):
            align_image_prefill_score(torch.ones(2, 2, 6), prompt_len=5)


class ImagePrefillStatsTest(unittest.TestCase):
    def test_forward_runs_once_and_never_receives_text_suffix(self):
        inputs = {
            "input_ids": torch.tensor([[10, 99, 99, 11, 20, 21]]),
            "pixel_values": torch.ones(1),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        }

        class Model:
            config = object()

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)

        class Capture:
            def __enter__(self):
                self.qk = [object(), object()]
                return self

            def __exit__(self, *args):
                return False

        model = Model()
        spans = {"visual": torch.tensor([1, 2]), "vis_end": 2, "L": 6}
        expected = torch.arange(8, dtype=torch.float32).view(2, 1, 4)
        with (
            patch("vlm_diagnosis.exps.core_delta_full_kv.S.vlm_inputs", return_value=inputs),
            patch("vlm_diagnosis.exps.core_delta_full_kv.token_spans", return_value=spans),
            patch("vlm_diagnosis.exps.core_delta_full_kv.mrope_position_ids",
                  return_value=torch.zeros(3, 1, 4, dtype=torch.long)),
            patch("vlm_diagnosis.exps.core_delta_full_kv.QKCapture", Capture),
            patch("vlm_diagnosis.exps.core_delta_full_kv.kv_dims", return_value=(2, 1, 4)),
            patch("vlm_diagnosis.exps.core_delta_full_kv.per_head_column_stats",
                  return_value=(expected, torch.zeros_like(expected))),
        ):
            score, prefix_ids, n_rows = image_prefill_stats(
                model, object(), object(), "cpu")

        self.assertEqual(len(model.calls), 1)
        self.assertTrue(torch.equal(model.calls[0]["input_ids"], inputs["input_ids"][:, :4]))
        self.assertNotIn(20, model.calls[0]["input_ids"].tolist()[0])
        self.assertTrue(torch.equal(score, expected))
        self.assertTrue(torch.equal(prefix_ids, inputs["input_ids"][:, :4]))
        self.assertEqual(n_rows, 3)


if __name__ == "__main__":
    unittest.main()
