import copy
import tempfile
import unittest
from pathlib import Path

import torch

from vlm_diagnosis.core.tensor_quantization import (
    dequantize_tensor,
    pack_signed_int4,
    quantize_tensor,
    unpack_signed_int4,
    validate_quantized_tensor,
)


class SignedInt4PackingTest(unittest.TestCase):
    def test_even_and_odd_round_trip_use_low_nibble_first(self):
        even = torch.tensor([-8, -7, -1, 0, 1, 7], dtype=torch.int8)
        packed, padding = pack_signed_int4(even)
        self.assertEqual(padding, 0)
        self.assertEqual(packed.tolist(), [0x98, 0x0F, 0x71])
        self.assertTrue(torch.equal(unpack_signed_int4(packed, 6), even))

        odd = torch.tensor([-7, 0, 7], dtype=torch.int8)
        packed, padding = pack_signed_int4(odd)
        self.assertEqual(padding, 1)
        self.assertEqual(packed.tolist(), [0x09, 0x07])
        self.assertTrue(torch.equal(unpack_signed_int4(packed, 3), odd))

    def test_packer_rejects_non_integer_and_out_of_range_values(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            pack_signed_int4(torch.tensor([1.0]))
        with self.assertRaisesRegex(ValueError, r"\[-8, 7\]"):
            pack_signed_int4(torch.tensor([-9], dtype=torch.int8))

    def test_unpacker_rejects_wrong_physical_byte_count(self):
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            unpack_signed_int4(torch.tensor([0], dtype=torch.uint8), 3)


class PerTokenQuantizationTest(unittest.TestCase):
    def setUp(self):
        self.tensor = torch.tensor(
            [
                [-4.0, -1.5, 0.0, 0.5, 4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.2, -0.1, 0.1, 0.15, 0.2],
            ],
            dtype=torch.float16,
        )

    def test_fp16_int8_and_int4_round_trip_contracts(self):
        sizes = {}
        for scheme in ("fp16", "int8", "int4"):
            with self.subTest(scheme=scheme):
                payload = quantize_tensor(self.tensor, scheme)
                validate_quantized_tensor(payload)
                restored = dequantize_tensor(payload, dtype=torch.float32)
                self.assertEqual(restored.shape, self.tensor.shape)
                self.assertTrue(torch.isfinite(restored).all())
                self.assertTrue(torch.equal(restored[1], torch.zeros(5)))
                metadata = payload["metadata"]
                self.assertEqual(metadata["scheme"], scheme)
                self.assertEqual(metadata["shape"], [3, 5])
                self.assertEqual(metadata["reference_fp16_bytes"], 30)
                self.assertIn("cosine_similarity", metadata["error_stats"])
                sizes[scheme] = metadata["payload_tensor_bytes"]
        self.assertEqual(sizes["fp16"], 30)
        self.assertLess(sizes["int4"], sizes["int8"])

    def test_integer_error_is_bounded_by_half_a_per_token_step(self):
        for scheme in ("int8", "int4"):
            payload = quantize_tensor(self.tensor, scheme)
            restored = dequantize_tensor(payload, dtype=torch.float32)
            error = (restored - self.tensor.float()).abs().amax(dim=1, keepdim=True)
            half_step = payload["scales"] / 2
            self.assertTrue(torch.all(error <= half_step + 1e-6))

    def test_int4_is_physically_packed_and_uses_symmetric_range(self):
        payload = quantize_tensor(self.tensor, "int4")
        metadata = payload["metadata"]
        self.assertEqual(payload["data"].dtype, torch.uint8)
        self.assertEqual(payload["data"].numel(), 8)
        self.assertEqual(metadata["padding_values"], 1)
        unpacked = unpack_signed_int4(payload["data"], self.tensor.numel())
        self.assertGreaterEqual(int(unpacked.min()), -7)
        self.assertLessEqual(int(unpacked.max()), 7)

    def test_validation_rejects_corrupt_scale_and_padding(self):
        int8 = quantize_tensor(self.tensor, "int8")
        broken_scale = copy.deepcopy(int8)
        broken_scale["scales"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            validate_quantized_tensor(broken_scale)

        int4 = quantize_tensor(self.tensor, "int4")
        broken_padding = copy.deepcopy(int4)
        broken_padding["data"][-1] |= 0xF0
        with self.assertRaisesRegex(ValueError, "padding nibble"):
            validate_quantized_tensor(broken_padding)

    def test_non_finite_and_unsupported_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            quantize_tensor(torch.tensor([[float("nan")]]), "int8")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            quantize_tensor(self.tensor, "int3")

    def test_weights_only_serialization_records_actual_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            sizes = {}
            for scheme in ("fp16", "int8", "int4"):
                path = Path(directory) / f"tokens.{scheme}.pt"
                torch.save(quantize_tensor(self.tensor, scheme), path)
                sizes[scheme] = path.stat().st_size
                loaded = torch.load(path, map_location="cpu", weights_only=True)
                validate_quantized_tensor(loaded)
            self.assertTrue(all(size > 0 for size in sizes.values()))


if __name__ == "__main__":
    unittest.main()
