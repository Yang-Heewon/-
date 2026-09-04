"""Materialize query-blind OCR views as physically byte-accounted payloads.

``frontier_paddleocr`` deliberately writes a rich audit JSONL containing the
recognized text, layout view, boxes, provenance, and accounting fields.  That
audit file is not a fair storage payload: evaluating a string embedded inside
it can report fewer bytes than are physically retained.  This module closes
that gap by writing one minimal UTF-8 file per sample and representation, plus
a source-free manifest containing the file's exact size and SHA-256 digest.

The optional byte cap is enforced while writing the file.  A read worker must
therefore consume the whole materialized file; it cannot claim a smaller
in-memory prefix while leaving a larger hidden package on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from vlm_diagnosis.core.byte_codecs import truncate_utf8_to_budget
from vlm_diagnosis.exps.source_denial_kv import ROOT, assert_source_free


SCHEMA_VERSION = "1.0"
REPRESENTATION_KEYS = {
    "plain": "recognized_text",
    "layout": "layout_text",
}
JSON_KWARGS = {
    "ensure_ascii": False,
    "sort_keys": True,
    "separators": (",", ":"),
    "allow_nan": False,
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_sample_id(sample_id: Any) -> str:
    raw = str(sample_id)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "sample"
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:80]}-{suffix}"


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _reject_data_output(path: Path) -> None:
    data_root = (ROOT / "data").resolve()
    if _inside(path.resolve(), data_root):
        raise ValueError(f"materialized OCR payloads cannot be written under data/: {path}")


def _load_input(path: Path) -> list[tuple[dict[str, Any], bytes]]:
    rows: list[tuple[dict[str, Any], bytes]] = []
    seen: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid UTF-8 JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"OCR row must be an object: {path}:{line_number}")
            if row.get("record_type") == "run_metadata":
                continue
            assert_source_free(row)
            if "sample_id" not in row:
                raise ValueError(f"OCR row has no sample_id: {path}:{line_number}")
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate OCR sample_id {sample_id!r}")
            seen.add(sample_id)
            reported = row.get("package_bytes")
            if reported is not None and reported != len(raw):
                raise ValueError(
                    f"producer package_bytes mismatch at {path}:{line_number}: "
                    f"reported={reported!r} actual={len(raw)}"
                )
            rows.append((row, raw))
    return rows


def _selected_view(row: Mapping[str, Any], representation: str) -> str:
    key = REPRESENTATION_KEYS[representation]
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(
            f"OCR sample {row.get('sample_id')!r} needs string field {key!r}"
        )
    return value


def _bounded_payload(text: str, byte_cap: int | None) -> tuple[bytes, bool]:
    if byte_cap is None:
        return text.encode("utf-8"), False
    bounded = truncate_utf8_to_budget(text, byte_cap)
    return bounded.payload, bounded.truncated


def _validate_output_locations(out_manifest: Path, payload_dir: Path) -> None:
    manifest_parent = out_manifest.parent.resolve()
    resolved_payload_dir = payload_dir.resolve()
    if not _inside(resolved_payload_dir, manifest_parent):
        raise ValueError(
            "payload directory must be contained by the package-manifest directory"
        )
    _reject_data_output(out_manifest)
    _reject_data_output(payload_dir)


def materialize_packages(
    *,
    input_manifest: Path,
    out_manifest: Path,
    payload_dir: Path,
    representations: Iterable[str] = ("plain", "layout"),
    byte_cap: int | None = None,
    overwrite: bool = False,
) -> None:
    """Write minimal payload files and their source-free integrity manifest."""

    selected_representations = tuple(dict.fromkeys(representations))
    if not selected_representations:
        raise ValueError("at least one representation is required")
    unknown = set(selected_representations).difference(REPRESENTATION_KEYS)
    if unknown:
        raise ValueError(f"unknown representation(s): {sorted(unknown)}")
    if byte_cap is not None and byte_cap < 0:
        raise ValueError("byte_cap must be non-negative")

    input_manifest = input_manifest.resolve()
    out_manifest = out_manifest.resolve()
    payload_dir = payload_dir.resolve()
    _validate_output_locations(out_manifest, payload_dir)
    rows = _load_input(input_manifest)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    payload_dir.mkdir(parents=True, exist_ok=True)
    if out_manifest.exists() and not overwrite:
        raise FileExistsError(f"package manifest already exists: {out_manifest}")

    input_sha256 = _sha256_path(input_manifest)
    manifest_rows: list[dict[str, Any]] = []
    payloads: list[tuple[Path, bytes]] = []
    for row, raw in rows:
        sample_id = str(row["sample_id"])
        slug = _safe_sample_id(sample_id)
        descriptors: dict[str, dict[str, Any]] = {}
        for representation in selected_representations:
            text = _selected_view(row, representation)
            original_bytes = len(text.encode("utf-8"))
            payload, truncated = _bounded_payload(text, byte_cap)
            destination = payload_dir / f"{slug}.{representation}.utf8"
            if destination.exists() and not overwrite:
                raise FileExistsError(f"payload already exists: {destination}")
            relative = destination.relative_to(out_manifest.parent).as_posix()
            descriptors[representation] = {
                "payload_relpath": relative,
                "payload_bytes": len(payload),
                "payload_sha256": _sha256_bytes(payload),
                "encoding": "utf-8",
                "file_count": 1,
                "original_utf8_bytes": original_bytes,
                "byte_cap": byte_cap,
                "truncated": truncated,
            }
            payloads.append((destination, payload))

        output_row = {
            "record_type": "materialized_text_memory_package",
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "representations": descriptors,
            "source_path_stored": False,
            "future_questions_visible": False,
            "input_record_sha256": _sha256_bytes(raw),
        }
        assert_source_free(output_row)
        manifest_rows.append(output_row)

    for destination, payload in payloads:
        mode = "wb" if overwrite else "xb"
        with destination.open(mode) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    metadata = {
        "record_type": "run_metadata",
        "schema_version": SCHEMA_VERSION,
        "stage": "OCR_PAYLOAD_MATERIALIZATION",
        "input_manifest_sha256": input_sha256,
        "representations": list(selected_representations),
        "byte_cap_per_selected_representation": byte_cap,
        "budget_scope": "one_complete_materialized_representation_payload",
        "source_path_available": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded_rows = [metadata, *manifest_rows]
    manifest_payload = b"".join(
        json.dumps(item, **JSON_KWARGS).encode("utf-8") + b"\n"
        for item in encoded_rows
    )
    temporary = out_manifest.with_name(f".{out_manifest.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(manifest_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out_manifest)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, help="frontier_paddleocr JSONL")
    parser.add_argument("--out-manifest", required=True, help="materialized package JSONL")
    parser.add_argument(
        "--payload-dir",
        help="payload directory; default is <out-manifest stem>_payloads beside it",
    )
    parser.add_argument(
        "--representation",
        action="append",
        choices=sorted(REPRESENTATION_KEYS),
        dest="representations",
        help="repeat to select arms; default materializes plain and layout",
    )
    parser.add_argument("--byte-cap", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_manifest = Path(args.out_manifest)
    payload_dir = (
        Path(args.payload_dir)
        if args.payload_dir
        else out_manifest.parent / f"{out_manifest.stem}_payloads"
    )
    materialize_packages(
        input_manifest=Path(args.input_manifest),
        out_manifest=out_manifest,
        payload_dir=payload_dir,
        representations=args.representations or ("plain", "layout"),
        byte_cap=args.byte_cap,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
