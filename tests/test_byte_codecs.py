import random
import unittest
from io import BytesIO

from PIL import Image

from vlm_diagnosis.core.byte_codecs import (
    CodecUnavailableError,
    available_image_codecs,
    encode_image_to_budget,
    image_codec_available,
    resize_image_long_side,
    truncate_utf8_to_budget,
)


def _noise_image(size=(96, 96), mode="RGB"):
    rng = random.Random(20260818)
    channels = 4 if mode == "RGBA" else 3
    payload = bytes(rng.randrange(256) for _ in range(size[0] * size[1] * channels))
    return Image.frombytes(mode, size, payload)


class ImageByteCodecTest(unittest.TestCase):
    def test_expected_pillow_codecs_are_discoverable(self):
        self.assertTrue(image_codec_available("jpeg"))
        self.assertIn("JPEG", available_image_codecs())
        if image_codec_available("webp"):
            self.assertIn("WEBP", available_image_codecs())

    def _assert_quality_search_and_accounting(self, codec, max_quality):
        image = _noise_image()
        roomy = encode_image_to_budget(image, codec, 1_000_000)
        self.assertTrue(roomy.feasible)
        self.assertEqual(roomy.quality, max_quality)
        self.assertEqual(roomy.serialized_bytes, len(roomy.payload))
        self.assertLessEqual(roomy.serialized_bytes, roomy.target_bytes)

        # One byte below the maximum-quality file forces the search to test a
        # lower quality while retaining a realistic, feasible budget.
        tight_budget = roomy.serialized_bytes - 1
        tight = encode_image_to_budget(image, codec, tight_budget)
        self.assertTrue(tight.feasible)
        self.assertLess(tight.quality, max_quality)
        self.assertEqual(tight.serialized_bytes, len(tight.payload))
        self.assertLessEqual(tight.serialized_bytes, tight_budget)
        self.assertGreater(tight.attempts, 1)
        self.assertLessEqual(tight.budget_utilization, 1.0)

        with Image.open(BytesIO(tight.payload)) as decoded:
            self.assertEqual(decoded.size, image.size)

    def test_jpeg_search_never_returns_over_budget_bytes(self):
        self._assert_quality_search_and_accounting("jpg", 95)

    @unittest.skipUnless(image_codec_available("WEBP"), "Pillow WebP encoder unavailable")
    def test_webp_search_never_returns_over_budget_bytes(self):
        self._assert_quality_search_and_accounting("WEBP", 100)

    def test_infeasible_budget_does_not_expose_over_budget_payload(self):
        result = encode_image_to_budget(_noise_image(), "JPEG", 1)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.payload)
        self.assertIsNone(result.serialized_bytes)
        self.assertIsNone(result.quality)
        self.assertGreater(result.smallest_tested_bytes, result.target_bytes)
        self.assertIsNotNone(result.reason)

    def test_jpeg_alpha_is_composited_without_mutating_input(self):
        image = _noise_image((32, 32), mode="RGBA")
        result = encode_image_to_budget(image, "JPEG", 100_000)
        self.assertTrue(result.feasible)
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(result.source_mode, "RGBA")
        self.assertEqual(result.encoded_mode, "RGB")

    def test_quality_and_format_cannot_be_overridden_in_settings(self):
        image = _noise_image((16, 16))
        with self.assertRaises(ValueError):
            encode_image_to_budget(image, "JPEG", 10_000, encoder_settings={"quality": 1})
        with self.assertRaises(ValueError):
            encode_image_to_budget(image, "JPEG", 10_000, encoder_settings={"format": "PNG"})

    def test_avif_is_used_only_if_registered(self):
        image = _noise_image((32, 32))
        if image_codec_available("AVIF"):
            result = encode_image_to_budget(image, "AVIF", 100_000)
            self.assertTrue(result.feasible)
            self.assertLessEqual(result.serialized_bytes, result.target_bytes)
        else:
            with self.assertRaises(CodecUnavailableError):
                encode_image_to_budget(image, "AVIF", 100_000)

    def test_invalid_budget_and_quality_bounds_are_rejected(self):
        image = _noise_image((16, 16))
        with self.assertRaises(ValueError):
            encode_image_to_budget(image, "JPEG", 0)
        with self.assertRaises(ValueError):
            encode_image_to_budget(image, "JPEG", 1000, quality_min=90, quality_max=80)

    def test_declared_long_side_resize_preserves_aspect_and_never_upsamples(self):
        image = Image.new("RGB", (1080, 1920), "white")
        resized = resize_image_long_side(image, 768)
        self.assertEqual(resized.size, (432, 768))
        self.assertEqual(image.size, (1080, 1920))
        small = Image.new("RGB", (20, 10), "white")
        copied = resize_image_long_side(small, 100)
        self.assertEqual(copied.size, small.size)
        self.assertIsNot(copied, small)

        with self.assertRaises(ValueError):
            resize_image_long_side(image, 0)


class TextByteCodecTest(unittest.TestCase):
    def test_truncation_never_splits_a_utf8_character(self):
        # Byte lengths: A=1, 한=3, 🙂=4, B=1.  A 5-byte prefix lands inside 🙂.
        result = truncate_utf8_to_budget("A한🙂B", 5)
        self.assertEqual(result.text, "A한")
        self.assertEqual(result.payload.decode("utf-8"), result.text)
        self.assertEqual(result.serialized_bytes, 4)
        self.assertLessEqual(result.serialized_bytes, result.target_bytes)
        self.assertTrue(result.truncated)

    def test_text_that_fits_is_unchanged_and_reports_actual_bytes(self):
        result = truncate_utf8_to_budget("OCR 한글", 100)
        self.assertEqual(result.text, "OCR 한글")
        self.assertEqual(result.serialized_bytes, len(result.payload))
        self.assertEqual(result.original_serialized_bytes, len("OCR 한글".encode("utf-8")))
        self.assertFalse(result.truncated)

    def test_zero_budget_is_a_valid_empty_prefix(self):
        result = truncate_utf8_to_budget("🙂", 0)
        self.assertEqual(result.text, "")
        self.assertEqual(result.payload, b"")
        self.assertEqual(result.serialized_bytes, 0)
        self.assertEqual(result.budget_utilization, 0.0)

    def test_negative_text_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            truncate_utf8_to_budget("text", -1)


if __name__ == "__main__":
    unittest.main()
