"""Portable tensor quantization primitives for diagnostic memory baselines.

The integer schemes use one symmetric scale per token (row).  INT8 uses the
range [-127, 127].  INT4 deliberately uses [-7, 7], leaving the two's
complement value -8 unused so that positive and negative ranges are symmetric.
INT4 values are flattened row-major and packed low nibble first; an odd final
value is padded with a zero high nibble.

This module is intentionally model-agnostic.  It returns CPU-only dictionaries
containing tensors and JSON-like metadata, so they can be embedded directly in
a ``torch.save`` package and loaded with ``weights_only=True``.
"""
from __future__ import annotations

import math
from typing import Any

import torch


FORMAT_VERSION = 1
QUANTIZATION_SCHEMES = ("fp16", "int8", "int4")
_ERROR_KEYS = {
    "mse",
    "mae",
    "max_abs_error",
    "relative_l2_error",
    "cosine_similarity",
}


def _shape2(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1
               for item in value)
    ):
        raise ValueError("shape must contain two positive integers")
    return int(value[0]), int(value[1])


def pack_signed_int4(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack signed four-bit integers low-nibble first on the CPU.

    The helper accepts the complete signed nibble range [-8, 7].  The
    quantizer itself emits only [-7, 7].  ``padding_values`` is one when an
    odd number of values required a zero-valued high nibble, otherwise zero.
    """
    if not torch.is_tensor(values):
        raise TypeError("values must be a tensor")
    if values.dtype == torch.bool or values.is_floating_point() or values.is_complex():
        raise ValueError("INT4 packing requires an integer tensor")
    flat = values.detach().to(device="cpu", dtype=torch.int16).contiguous().view(-1)
    if flat.numel() and (int(flat.min()) < -8 or int(flat.max()) > 7):
        raise ValueError("INT4 values must lie in [-8, 7]")
    unsigned = torch.bitwise_and(flat, 0x0F).to(torch.uint8)
    padding_values = int(unsigned.numel() % 2)
    if padding_values:
        unsigned = torch.cat((unsigned, torch.zeros(1, dtype=torch.uint8)))
    packed = unsigned[0::2] | torch.bitwise_left_shift(unsigned[1::2], 4)
    return packed.contiguous(), padding_values


def unpack_signed_int4(packed: torch.Tensor, value_count: int) -> torch.Tensor:
    """Reverse :func:`pack_signed_int4`, returning a flat CPU INT8 tensor."""
    if not torch.is_tensor(packed) or packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("packed INT4 data must be a one-dimensional uint8 tensor")
    if not isinstance(value_count, int) or isinstance(value_count, bool) or value_count < 0:
        raise ValueError("value_count must be a non-negative integer")
    expected_bytes = (value_count + 1) // 2
    if packed.numel() != expected_bytes:
        raise ValueError(
            f"packed INT4 byte count mismatch: {packed.numel()} != {expected_bytes}"
        )
    data = packed.detach().to(device="cpu")
    unsigned = torch.empty(data.numel() * 2, dtype=torch.uint8)
    unsigned[0::2] = torch.bitwise_and(data, 0x0F)
    unsigned[1::2] = torch.bitwise_right_shift(data, 4)
    unsigned = unsigned[:value_count]
    signed = unsigned.to(torch.int8)
    return torch.where(signed >= 8, signed - 16, signed).to(torch.int8)


def _error_stats(reference: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    reference64 = reference.to(torch.float64).reshape(-1)
    reconstructed64 = reconstructed.to(torch.float64).reshape(-1)
    error = reconstructed64 - reference64
    reference_norm = torch.linalg.vector_norm(reference64)
    reconstructed_norm = torch.linalg.vector_norm(reconstructed64)
    error_norm = torch.linalg.vector_norm(error)
    if float(reference_norm) == 0.0:
        relative_l2 = 0.0 if float(error_norm) == 0.0 else 1.0
    else:
        relative_l2 = float(error_norm / reference_norm)
    if float(reference_norm) == 0.0 and float(reconstructed_norm) == 0.0:
        cosine = 1.0
    elif float(reference_norm) == 0.0 or float(reconstructed_norm) == 0.0:
        cosine = 0.0
    else:
        cosine = float(
            torch.dot(reference64, reconstructed64)
            / (reference_norm * reconstructed_norm)
        )
    return {
        "mse": float(torch.mean(error.square())),
        "mae": float(torch.mean(error.abs())),
        "max_abs_error": float(error.abs().max()),
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
    }


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _decode_without_validation(payload: dict[str, Any]) -> torch.Tensor:
    metadata = payload["metadata"]
    shape = tuple(metadata["shape"])
    scheme = metadata["scheme"]
    if scheme == "fp16":
        return payload["data"].reshape(shape).to(torch.float32)
    scales = payload["scales"].to(torch.float32)
    if scheme == "int8":
        quantized = payload["data"].to(torch.float32)
    else:
        quantized = unpack_signed_int4(
            payload["data"], metadata["value_count"]
        ).reshape(shape).to(torch.float32)
    return quantized * scales


def quantize_tensor(tensor: torch.Tensor, scheme: str) -> dict[str, Any]:
    """Encode a rank-two token-by-hidden tensor into a portable CPU payload."""
    scheme = str(scheme).lower()
    if scheme not in QUANTIZATION_SCHEMES:
        raise ValueError(
            f"unsupported quantization scheme {scheme!r}; "
            f"choose from {QUANTIZATION_SCHEMES}"
        )
    if not torch.is_tensor(tensor) or tensor.ndim != 2:
        raise ValueError("tensor must have shape (tokens, hidden_size)")
    if tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError("tensor dimensions must be non-empty")
    reference = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(reference).all():
        raise ValueError("tensor contains non-finite values")
    n_tokens, hidden_size = reference.shape

    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "scheme": scheme,
        "shape": [int(n_tokens), int(hidden_size)],
        "value_count": int(reference.numel()),
        "reference_dtype": "float16",
        "reference_fp16_bytes": int(reference.numel() * 2),
    }
    if scheme == "fp16":
        data = reference.to(torch.float16).contiguous()
        payload: dict[str, Any] = {"metadata": metadata, "data": data}
        metadata.update({
            "bits": 16,
            "granularity": "none",
            "symmetric": False,
            "storage_dtype": "float16",
            "packed": False,
            "padding_values": 0,
            "scale_dtype": None,
            "qmin": None,
            "qmax": None,
        })
    else:
        bits = 8 if scheme == "int8" else 4
        qmax = (1 << (bits - 1)) - 1
        max_abs = reference.abs().amax(dim=1, keepdim=True)
        scales = torch.where(
            max_abs == 0,
            torch.ones_like(max_abs),
            max_abs / qmax,
        ).to(torch.float32).contiguous()
        quantized = torch.round(reference / scales).clamp(-qmax, qmax).to(torch.int8)
        if scheme == "int8":
            data = quantized.contiguous()
            padding_values = 0
            storage_dtype = "int8"
            packed = False
        else:
            data, padding_values = pack_signed_int4(quantized)
            storage_dtype = "uint8_packed_int4"
            packed = True
        payload = {"metadata": metadata, "data": data, "scales": scales}
        metadata.update({
            "bits": bits,
            "granularity": "per_token",
            "symmetric": True,
            "storage_dtype": storage_dtype,
            "packed": packed,
            "packing_order": "row_major_low_nibble_first" if packed else None,
            "padding_values": padding_values,
            "scale_dtype": "float32",
            "qmin": -qmax,
            "qmax": qmax,
        })

    reconstructed = _decode_without_validation(payload)
    metadata["error_stats"] = _error_stats(reference, reconstructed)
    payload_bytes = _tensor_bytes(payload["data"])
    if "scales" in payload:
        payload_bytes += _tensor_bytes(payload["scales"])
    metadata["payload_tensor_bytes"] = payload_bytes
    metadata["payload_compression_ratio_vs_fp16"] = (
        metadata["reference_fp16_bytes"] / payload_bytes
    )
    validate_quantized_tensor(payload)
    return payload


def validate_quantized_tensor(payload: dict[str, Any]) -> None:
    """Fail closed on malformed or internally inconsistent payloads."""
    if not isinstance(payload, dict):
        raise ValueError("quantized tensor payload must be a dictionary")
    if "metadata" not in payload or "data" not in payload:
        raise ValueError("quantized tensor payload requires metadata and data")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("quantization metadata must be a dictionary")
    required = {
        "format_version", "scheme", "shape", "value_count", "bits",
        "granularity", "symmetric", "storage_dtype", "packed",
        "padding_values", "scale_dtype", "qmin", "qmax",
        "reference_dtype", "reference_fp16_bytes", "payload_tensor_bytes",
        "payload_compression_ratio_vs_fp16", "error_stats",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"incomplete quantization metadata: {sorted(missing)}")
    if metadata["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"unsupported quantization format_version={metadata['format_version']!r}"
        )
    scheme = metadata["scheme"]
    if scheme not in QUANTIZATION_SCHEMES:
        raise ValueError(f"unsupported quantization scheme={scheme!r}")
    shape = _shape2(metadata["shape"])
    value_count = shape[0] * shape[1]
    if metadata["value_count"] != value_count:
        raise ValueError("quantization value_count disagrees with shape")
    if metadata["reference_dtype"] != "float16":
        raise ValueError("reference_dtype must be float16")
    if metadata["reference_fp16_bytes"] != value_count * 2:
        raise ValueError("reference_fp16_bytes disagrees with shape")

    data = payload["data"]
    if not torch.is_tensor(data) or data.device.type != "cpu":
        raise ValueError("quantized data must be a CPU tensor")
    if scheme == "fp16":
        if data.dtype != torch.float16 or tuple(data.shape) != shape:
            raise ValueError("FP16 data must have the declared shape and float16 dtype")
        if metadata["bits"] != 16 or metadata["granularity"] != "none":
            raise ValueError("invalid FP16 quantization metadata")
        if metadata["symmetric"] or metadata["packed"]:
            raise ValueError("FP16 payload cannot be symmetric or packed")
        if metadata["storage_dtype"] != "float16":
            raise ValueError("FP16 storage_dtype must be float16")
        if any(metadata[key] is not None for key in ("scale_dtype", "qmin", "qmax")):
            raise ValueError("FP16 payload cannot declare scales or integer bounds")
        if "scales" in payload:
            raise ValueError("FP16 payload cannot contain scales")
        if metadata["padding_values"] != 0:
            raise ValueError("FP16 payload cannot contain padding")
    else:
        bits = 8 if scheme == "int8" else 4
        qmax = (1 << (bits - 1)) - 1
        if metadata["bits"] != bits or metadata["granularity"] != "per_token":
            raise ValueError("invalid integer quantization metadata")
        if not metadata["symmetric"]:
            raise ValueError("integer quantization must be symmetric")
        if metadata["qmin"] != -qmax or metadata["qmax"] != qmax:
            raise ValueError("integer quantization bounds are inconsistent")
        scales = payload.get("scales")
        if (
            not torch.is_tensor(scales)
            or scales.device.type != "cpu"
            or scales.dtype != torch.float32
            or tuple(scales.shape) != (shape[0], 1)
        ):
            raise ValueError("per-token scales must have CPU float32 shape (tokens, 1)")
        if not torch.isfinite(scales).all() or not torch.all(scales > 0):
            raise ValueError("per-token scales must be finite and positive")
        if metadata["scale_dtype"] != "float32":
            raise ValueError("integer scale_dtype must be float32")
        if scheme == "int8":
            if data.dtype != torch.int8 or tuple(data.shape) != shape:
                raise ValueError("INT8 data must have the declared shape and int8 dtype")
            if int(data.min()) < -qmax or int(data.max()) > qmax:
                raise ValueError("INT8 data lies outside the symmetric range")
            if metadata["storage_dtype"] != "int8" or metadata["packed"]:
                raise ValueError("invalid INT8 storage metadata")
            if metadata["padding_values"] != 0:
                raise ValueError("INT8 payload cannot contain padding")
        else:
            expected_bytes = (value_count + 1) // 2
            if data.dtype != torch.uint8 or data.ndim != 1 or data.numel() != expected_bytes:
                raise ValueError("packed INT4 data has the wrong dtype, rank, or byte count")
            if metadata["storage_dtype"] != "uint8_packed_int4" or not metadata["packed"]:
                raise ValueError("invalid INT4 storage metadata")
            expected_padding = value_count % 2
            if metadata["padding_values"] != expected_padding:
                raise ValueError("INT4 padding metadata disagrees with shape")
            if metadata.get("packing_order") != "row_major_low_nibble_first":
                raise ValueError("unsupported INT4 packing order")
            if expected_padding and int(torch.bitwise_right_shift(data[-1], 4)) != 0:
                raise ValueError("INT4 padding nibble must encode zero")
            unpacked = unpack_signed_int4(data, value_count)
            if int(unpacked.min()) < -qmax or int(unpacked.max()) > qmax:
                raise ValueError("INT4 data lies outside the symmetric range")

    actual_payload_bytes = _tensor_bytes(data)
    if "scales" in payload:
        actual_payload_bytes += _tensor_bytes(payload["scales"])
    if metadata["payload_tensor_bytes"] != actual_payload_bytes:
        raise ValueError("payload_tensor_bytes disagrees with stored tensors")
    ratio = metadata["reference_fp16_bytes"] / actual_payload_bytes
    if not math.isclose(
        float(metadata["payload_compression_ratio_vs_fp16"]), ratio, rel_tol=1e-12
    ):
        raise ValueError("payload compression ratio is inconsistent")
    stats = metadata["error_stats"]
    if not isinstance(stats, dict) or _ERROR_KEYS.difference(stats):
        raise ValueError("quantization error_stats are incomplete")
    if any(not math.isfinite(float(stats[key])) for key in _ERROR_KEYS):
        raise ValueError("quantization error_stats must be finite")
    if any(float(stats[key]) < 0 for key in _ERROR_KEYS - {"cosine_similarity"}):
        raise ValueError("quantization error magnitudes cannot be negative")
    if not -1.000001 <= float(stats["cosine_similarity"]) <= 1.000001:
        raise ValueError("quantization cosine_similarity lies outside [-1, 1]")


def dequantize_tensor(
    payload: dict[str, Any], dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Validate and reconstruct a quantized tensor on the CPU."""
    validate_quantized_tensor(payload)
    if dtype not in {torch.float16, torch.float32, torch.float64, torch.bfloat16}:
        raise ValueError("dequantized dtype must be floating point")
    return _decode_without_validation(payload).to(dtype=dtype).contiguous()

