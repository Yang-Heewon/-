import unittest
from pathlib import Path

import torch

from vlm_diagnosis.core.tensor_quantization import quantize_tensor
from vlm_diagnosis.exps.source_denial_embedding import (
    REPRESENTATION,
    assert_question_free,
    decode_projected_visual_tokens,
    inject_projected_visual_tokens,
    merge_prefix_and_raw_prompt,
    projected_package_path,
    projected_quantization_scheme,
    validate_projected_package,
)


IMAGE_TOKEN_ID = 99


def valid_blob():
    return {
        "schema_version": "1.0",
        "representation": REPRESENTATION,
        "sample_id": "sample/1",
        "source_sha256": "a" * 64,
        "model_family": "qwen25vl",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "dtype": "float16",
        "boundary": "future question excluded",
        "prefix_ids": torch.tensor([[1, 10, 99, 99]]),
        "prefix_position_ids": torch.arange(4).view(1, 1, 4).expand(3, -1, -1),
        "prefix_rope_delta": torch.tensor([[-1]]),
        "image_grid_thw": torch.tensor([[1, 4, 2]]),
        "vis_start": 2,
        "vis_end": 3,
        "projected_visual_tokens": torch.ones(2, 3, dtype=torch.float16),
    }


def quantized_blob(scheme):
    blob = valid_blob()
    encoded = quantize_tensor(blob["projected_visual_tokens"], scheme)
    blob["dtype"] = {"fp16": "float16", "int8": "int8", "int4": "int4"}[
        scheme
    ]
    blob["projected_visual_tokens"] = encoded["data"]
    blob["quantization"] = encoded["metadata"]
    if "scales" in encoded:
        blob["projected_visual_scales"] = encoded["scales"]
    return blob


class ProjectedVisualContractTest(unittest.TestCase):
    def test_writer_rejects_future_question_material(self):
        with self.assertRaisesRegex(ValueError, "question-bearing key"):
            assert_question_free({
                "sample_id": "1",
                "image": "allowed-at-write-time.png",
                "questions": [{"question": "future query"}],
            })

    def test_package_name_reuses_path_safe_d0_slug(self):
        first = projected_package_path(Path("packages"), "a/b")
        second = projected_package_path(Path("packages"), "a/b")
        self.assertEqual(first, second)
        self.assertEqual(first.parent, Path("packages"))
        self.assertNotIn("/", first.name)
        self.assertTrue(first.name.endswith(".full_projected_visual.pt"))

    def test_integer_package_names_do_not_collide_with_legacy_fp16(self):
        fp16 = projected_package_path(Path("packages"), "a/b", "fp16")
        int8 = projected_package_path(Path("packages"), "a/b", "int8")
        int4 = projected_package_path(Path("packages"), "a/b", "int4")
        self.assertEqual(len({fp16, int8, int4}), 3)
        self.assertTrue(int8.name.endswith(".full_projected_visual.int8.pt"))
        self.assertTrue(int4.name.endswith(".full_projected_visual.int4.pt"))

    def test_prefix_expands_the_raw_single_placeholder(self):
        prefix = torch.tensor([[1, 10, 99, 99]])
        raw = torch.tensor([[1, 10, 99, 11, 20, 21]])
        full = merge_prefix_and_raw_prompt(prefix, raw, IMAGE_TOKEN_ID)
        self.assertEqual(full.tolist(), [[1, 10, 99, 99, 11, 20, 21]])

    def test_raw_prompt_must_have_one_placeholder(self):
        prefix = torch.tensor([[1, 99, 99]])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            merge_prefix_and_raw_prompt(prefix, torch.tensor([[1, 2]]), IMAGE_TOKEN_ID)

    def test_visual_vectors_replace_only_image_token_embeddings(self):
        ids = torch.tensor([[1, 99, 99, 2]])
        original = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
        projected = torch.tensor([[20, 21, 22], [30, 31, 32]], dtype=torch.float16)
        result = inject_projected_visual_tokens(
            ids, original, projected, IMAGE_TOKEN_ID
        )
        self.assertTrue(torch.equal(result[0, 0], original[0, 0]))
        self.assertTrue(torch.equal(result[0, 3], original[0, 3]))
        self.assertTrue(torch.equal(result[0, 1], projected[0].float()))
        self.assertTrue(torch.equal(result[0, 2], projected[1].float()))
        self.assertFalse(torch.equal(original[0, 1], result[0, 1]))

    def test_visual_count_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            inject_projected_visual_tokens(
                torch.tensor([[99, 99]]),
                torch.zeros(1, 2, 3),
                torch.zeros(1, 3),
                IMAGE_TOKEN_ID,
            )

    def test_valid_package_contract(self):
        validate_projected_package(valid_blob())
        self.assertEqual(projected_quantization_scheme(valid_blob()), "fp16")

    def test_metadata_fp16_int8_and_int4_packages_are_source_free_and_decodable(self):
        for scheme in ("fp16", "int8", "int4"):
            with self.subTest(scheme=scheme):
                blob = quantized_blob(scheme)
                validate_projected_package(blob)
                decoded = decode_projected_visual_tokens(blob)
                self.assertEqual(decoded.dtype, torch.float16)
                self.assertEqual(decoded.shape, (2, 3))
                self.assertEqual(projected_quantization_scheme(blob), scheme)
                self.assertNotIn("image", blob)
                self.assertNotIn("source_path", blob)

    def test_legacy_fp16_package_decodes_without_new_metadata(self):
        blob = valid_blob()
        decoded = decode_projected_visual_tokens(blob)
        self.assertTrue(torch.equal(decoded, blob["projected_visual_tokens"]))

    def test_package_rejects_source_path(self):
        blob = valid_blob()
        blob["source_path"] = "private/image.png"
        with self.assertRaisesRegex(ValueError, "source-bearing key"):
            validate_projected_package(blob)

    def test_package_rejects_wrong_dtype(self):
        blob = valid_blob()
        blob["projected_visual_tokens"] = blob[
            "projected_visual_tokens"
        ].float()
        with self.assertRaisesRegex(RuntimeError, "must be float16"):
            validate_projected_package(blob)

    def test_package_rejects_quantization_dtype_disagreement(self):
        blob = quantized_blob("int8")
        blob["dtype"] = "int4"
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            validate_projected_package(blob)

    def test_package_rejects_corrupt_quantized_shape(self):
        blob = quantized_blob("int4")
        blob["quantization"]["shape"] = [3, 2]
        with self.assertRaisesRegex(RuntimeError, "scales must have"):
            validate_projected_package(blob)


if __name__ == "__main__":
    unittest.main()
