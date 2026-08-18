"""Byte-bounded codecs for durable visual-memory comparisons.

The helpers in this module account for the bytes produced by Pillow's encoder,
not an analytical size estimate.  Image search is deliberately exhaustive over
the small integer quality range: encoded size is not guaranteed to be monotonic
in ``quality``, so a binary search could silently miss a better feasible result.

No spatial resizing is performed by :func:`encode_image_to_budget`.  A caller
that wants a declared resolution arm must call :func:`resize_image_long_side`
first and record that arm separately.  If even the minimum requested quality
does not fit, the encoder returns an infeasible result without an over-budget
payload instead of changing resolution behind the experiment's back.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Mapping

from PIL import Image

try:  # Optional plugin registers AVIF with Pillow on import.
    import pillow_avif  # noqa: F401
except ImportError:  # AVIF remains an explicit infeasible/unavailable arm.
    pillow_avif = None


_CODEC_ALIASES = {
    "JPG": "JPEG",
    "JPEG": "JPEG",
    "WEBP": "WEBP",
    "AVIF": "AVIF",
}

_QUALITY_LIMITS = {
    "JPEG": (1, 95),
    "WEBP": (1, 100),
    "AVIF": (1, 100),
}

_DEFAULT_ENCODER_SETTINGS: dict[str, dict[str, Any]] = {
    "JPEG": {
        "optimize": True,
        "progressive": False,
        "subsampling": 2,
    },
    "WEBP": {
        "lossless": False,
        "method": 6,
        "exact": True,
    },
    "AVIF": {},
}


class CodecUnavailableError(RuntimeError):
    """Raised when Pillow has no registered encoder for a requested codec."""


class CodecEncodingError(RuntimeError):
    """Raised when a registered encoder cannot produce any candidate."""


@dataclass(frozen=True)
class ImageBudgetResult:
    """Result of a byte-bounded image encoding search.

    ``payload`` is ``None`` when ``feasible`` is false.  Consequently, callers
    cannot accidentally persist the smallest attempted but over-budget image.
    ``serialized_bytes`` is always the actual length of ``payload`` when one is
    returned, rather than an encoder estimate.
    """

    codec: str
    target_bytes: int
    feasible: bool
    payload: bytes | None
    serialized_bytes: int | None
    quality: int | None
    encoder_settings: Mapping[str, Any]
    quality_min: int
    quality_max: int
    attempts: int
    smallest_tested_bytes: int
    width: int
    height: int
    source_mode: str
    encoded_mode: str
    reason: str | None = None

    @property
    def budget_utilization(self) -> float | None:
        """Fraction of the byte budget occupied by the returned payload."""

        if self.serialized_bytes is None:
            return None
        return self.serialized_bytes / self.target_bytes


@dataclass(frozen=True)
class TextBudgetResult:
    """UTF-8 prefix truncated at a physical byte boundary."""

    text: str
    payload: bytes
    target_bytes: int
    serialized_bytes: int
    original_serialized_bytes: int
    truncated: bool
    encoding: str = "utf-8"

    @property
    def budget_utilization(self) -> float:
        """Fraction of the budget used (zero for a zero-byte budget)."""

        if self.target_bytes == 0:
            return 0.0
        return self.serialized_bytes / self.target_bytes


def _canonical_codec(codec: str) -> str:
    if not isinstance(codec, str):
        raise TypeError("codec must be a string")
    canonical = _CODEC_ALIASES.get(codec.strip().upper())
    if canonical is None:
        supported = ", ".join(sorted(_CODEC_ALIASES))
        raise ValueError(f"unsupported codec {codec!r}; expected one of: {supported}")
    return canonical


def image_codec_available(codec: str) -> bool:
    """Return whether Pillow currently has a registered encoder for ``codec``."""

    canonical = _canonical_codec(codec)
    # Pillow loads built-in plugins lazily.  This does not install or import an
    # optional external AVIF implementation; AVIF is used only if registered.
    Image.init()
    return canonical in Image.SAVE


def available_image_codecs() -> tuple[str, ...]:
    """Return supported codecs whose encoders are registered with Pillow."""

    return tuple(codec for codec in _QUALITY_LIMITS if image_codec_available(codec))


def _prepare_image(image: Image.Image, codec: str) -> Image.Image:
    """Convert only modes unsupported by the target container."""

    has_alpha = "A" in image.getbands() or (
        image.mode == "P" and "transparency" in image.info
    )

    if codec == "JPEG":
        if has_alpha:
            rgba = image.convert("RGBA")
            prepared = Image.new("RGB", rgba.size, (255, 255, 255))
            prepared.paste(rgba, mask=rgba.getchannel("A"))
            return prepared
        if image.mode not in ("L", "RGB"):
            return image.convert("RGB")
        return image

    # RGB/RGBA is a conservative common denominator for Pillow's WebP and
    # optional AVIF encoders.  Palette transparency must be expanded first.
    if has_alpha:
        return image if image.mode == "RGBA" else image.convert("RGBA")
    return image if image.mode == "RGB" else image.convert("RGB")


def resize_image_long_side(image: Image.Image, max_long_side: int) -> Image.Image:
    """Return a LANCZOS thumbnail with a declared maximum long side.

    Upsampling is forbidden: if the image already fits, a copy is returned.
    Keeping this operation separate from codec quality search makes resolution
    a visible experimental variable instead of a hidden encoder choice.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(max_long_side, int) or isinstance(max_long_side, bool):
        raise TypeError("max_long_side must be an integer")
    if max_long_side < 1:
        raise ValueError("max_long_side must be positive")
    width, height = image.size
    current = max(width, height)
    if current <= max_long_side:
        return image.copy()
    scale = max_long_side / current
    resized = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return image.resize(resized, Image.Resampling.LANCZOS)


def _encode_candidate(
    image: Image.Image,
    codec: str,
    quality: int,
    encoder_settings: Mapping[str, Any],
) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=codec, quality=quality, **encoder_settings)
    return buffer.getvalue()


def encode_image_to_budget(
    image: Image.Image,
    codec: str,
    target_bytes: int,
    *,
    quality_min: int | None = None,
    quality_max: int | None = None,
    encoder_settings: Mapping[str, Any] | None = None,
) -> ImageBudgetResult:
    """Encode ``image`` at the highest integer quality within ``target_bytes``.

    The search checks qualities from high to low, so it is bounded by at most
    100 Pillow encodes and does not assume that file size is monotonic.  The
    fixed ``encoder_settings`` are used at every quality.  ``quality`` and
    ``format`` are reserved because this function controls them.

    Args:
        image: A loaded Pillow image.  The input object is not mutated.
        codec: ``JPEG``/``JPG``, ``WEBP``, or registered optional ``AVIF``.
        target_bytes: Positive physical payload budget.
        quality_min: Inclusive lower search bound (codec default if omitted).
        quality_max: Inclusive upper search bound (codec default if omitted).
        encoder_settings: Fixed Pillow save options other than quality/format.

    Returns:
        A result with a budget-safe payload, or an explicit infeasible result.

    Raises:
        CodecUnavailableError: if Pillow has no registered encoder.
        CodecEncodingError: if all attempted encodes fail.
        ValueError: for invalid budget, quality bounds, or reserved settings.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not isinstance(target_bytes, int) or isinstance(target_bytes, bool):
        raise TypeError("target_bytes must be an integer")
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")

    canonical = _canonical_codec(codec)
    if not image_codec_available(canonical):
        raise CodecUnavailableError(
            f"Pillow has no registered {canonical} encoder in this environment"
        )

    allowed_min, allowed_max = _QUALITY_LIMITS[canonical]
    q_min = allowed_min if quality_min is None else quality_min
    q_max = allowed_max if quality_max is None else quality_max
    for label, quality in (("quality_min", q_min), ("quality_max", q_max)):
        if not isinstance(quality, int) or isinstance(quality, bool):
            raise TypeError(f"{label} must be an integer")
        if not allowed_min <= quality <= allowed_max:
            raise ValueError(
                f"{label} must be in [{allowed_min}, {allowed_max}] for {canonical}"
            )
    if q_min > q_max:
        raise ValueError("quality_min must be less than or equal to quality_max")

    overrides = dict(encoder_settings or {})
    reserved = {str(key).lower() for key in overrides} & {"format", "quality"}
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"encoder_settings may not override reserved option(s): {names}")
    settings = dict(_DEFAULT_ENCODER_SETTINGS[canonical])
    settings.update(overrides)

    prepared = _prepare_image(image, canonical)
    attempts = 0
    successful_encodes = 0
    smallest_tested: int | None = None
    last_error: Exception | None = None

    for quality in range(q_max, q_min - 1, -1):
        attempts += 1
        try:
            payload = _encode_candidate(prepared, canonical, quality, settings)
        except Exception as exc:  # Pillow plugins use several exception classes.
            last_error = exc
            continue
        successful_encodes += 1
        serialized_bytes = len(payload)
        if smallest_tested is None or serialized_bytes < smallest_tested:
            smallest_tested = serialized_bytes
        if serialized_bytes <= target_bytes:
            # Assert the central contract at the return boundary.
            assert len(payload) == serialized_bytes <= target_bytes
            return ImageBudgetResult(
                codec=canonical,
                target_bytes=target_bytes,
                feasible=True,
                payload=payload,
                serialized_bytes=serialized_bytes,
                quality=quality,
                encoder_settings=settings,
                quality_min=q_min,
                quality_max=q_max,
                attempts=attempts,
                smallest_tested_bytes=smallest_tested,
                width=prepared.width,
                height=prepared.height,
                source_mode=image.mode,
                encoded_mode=prepared.mode,
            )

    if successful_encodes == 0:
        detail = f": {last_error}" if last_error is not None else ""
        raise CodecEncodingError(
            f"{canonical} failed for every quality in [{q_min}, {q_max}]{detail}"
        ) from last_error

    assert smallest_tested is not None
    return ImageBudgetResult(
        codec=canonical,
        target_bytes=target_bytes,
        feasible=False,
        payload=None,
        serialized_bytes=None,
        quality=None,
        encoder_settings=settings,
        quality_min=q_min,
        quality_max=q_max,
        attempts=attempts,
        smallest_tested_bytes=smallest_tested,
        width=prepared.width,
        height=prepared.height,
        source_mode=image.mode,
        encoded_mode=prepared.mode,
        reason="no encoding in the requested quality range fits the byte budget",
    )


def truncate_utf8_to_budget(text: str, target_bytes: int) -> TextBudgetResult:
    """Return the longest UTF-8-safe character prefix within ``target_bytes``."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(target_bytes, int) or isinstance(target_bytes, bool):
        raise TypeError("target_bytes must be an integer")
    if target_bytes < 0:
        raise ValueError("target_bytes must be non-negative")

    original = text.encode("utf-8")
    if len(original) <= target_bytes:
        return TextBudgetResult(
            text=text,
            payload=original,
            target_bytes=target_bytes,
            serialized_bytes=len(original),
            original_serialized_bytes=len(original),
            truncated=False,
        )

    # A prefix of valid UTF-8 can only be invalid at its final, incomplete code
    # point.  ``ignore`` therefore removes that fragment and retains the longest
    # complete character prefix without inserting replacement characters.
    truncated_text = original[:target_bytes].decode("utf-8", errors="ignore")
    payload = truncated_text.encode("utf-8")
    assert len(payload) <= target_bytes
    return TextBudgetResult(
        text=truncated_text,
        payload=payload,
        target_bytes=target_bytes,
        serialized_bytes=len(payload),
        original_serialized_bytes=len(original),
        truncated=True,
    )


__all__ = [
    "CodecEncodingError",
    "CodecUnavailableError",
    "ImageBudgetResult",
    "TextBudgetResult",
    "available_image_codecs",
    "encode_image_to_budget",
    "image_codec_available",
    "resize_image_long_side",
    "truncate_utf8_to_budget",
]
