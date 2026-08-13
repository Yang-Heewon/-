import unittest

import torch

from vlm_diagnosis.core.signals import spatial_uniform_keep


class SpatialUniformTest(unittest.TestCase):
    def test_keep_count_and_determinism(self):
        grid = torch.tensor([[1, 8, 12]])
        first = spatial_uniform_keep(grid, merge_size=2, keep_count=6)
        second = spatial_uniform_keep(grid, merge_size=2, keep_count=6)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertTrue(all(0 <= index < 24 for index in first))

    def test_full_keep_is_identity_set(self):
        grid = torch.tensor([[1, 4, 6]])
        self.assertEqual(spatial_uniform_keep(grid, 2, 6), set(range(6)))

    def test_invalid_keep_count_is_rejected(self):
        with self.assertRaises(ValueError):
            spatial_uniform_keep(torch.tensor([[1, 4, 4]]), 2, 0)


if __name__ == "__main__":
    unittest.main()
