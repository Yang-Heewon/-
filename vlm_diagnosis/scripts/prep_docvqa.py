"""Create a revision-pinned DocVQA JSONL manifest and local image subset.

This replaces the legacy ``data/d4_mini/meta.jsonl`` writer.  It emits the
common manifest fields from ``experiments/manifests/README.md`` and keeps all
questions from the selected documents so M2-A/M3 can derive stage-specific
subsets without silently resampling documents.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "lmms-lab/DocVQA"
DATASET_CONFIG = "DocVQA"


def _stream(*, source_split: str, dataset_revision: str):
    return load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=source_split,
        revision=dataset_revision,
        streaming=True,
    )


def _select_doc_ids(
    *,
    source_split: str,
    dataset_revision: str,
    scan_rows: int,
    k_min: int,
    n_docs: int,
    seed: int,
) -> tuple[list[object], int, int]:
    metadata = _stream(source_split=source_split, dataset_revision=dataset_revision).select_columns(
        ["questionId", "docId"]
    )
    counts: dict[object, int] = {}
    order: list[object] = []
    rows_seen = 0
    for rows_seen, example in enumerate(metadata, start=1):
        if rows_seen > scan_rows:
            rows_seen = scan_rows
            break
        doc_id = example["docId"]
        if doc_id not in counts:
            counts[doc_id] = 0
            order.append(doc_id)
        counts[doc_id] += 1

    eligible = [doc_id for doc_id in order if counts[doc_id] >= k_min]
    if len(eligible) < n_docs:
        raise RuntimeError(
            f"only {len(eligible)} documents have at least {k_min} questions "
            f"within {rows_seen} scanned rows; requested {n_docs}"
        )
    chosen_set = set(random.Random(seed).sample(eligible, n_docs))
    chosen = [doc_id for doc_id in order if doc_id in chosen_set]
    return chosen, rows_seen, len(eligible)


def build_manifest(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    image_dir = args.image_dir.resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    chosen, rows_seen, eligible_count = _select_doc_ids(
        source_split=args.source_split,
        dataset_revision=args.dataset_revision,
        scan_rows=args.scan_rows,
        k_min=args.k_min,
        n_docs=args.n_docs,
        seed=args.seed,
    )
    chosen_set = set(chosen)
    documents = {doc_id: {"questions": []} for doc_id in chosen}

    for row_index, example in enumerate(
        _stream(source_split=args.source_split, dataset_revision=args.dataset_revision)
    ):
        if row_index >= args.scan_rows:
            break
        doc_id = example["docId"]
        if doc_id not in chosen_set:
            continue
        document = documents[doc_id]
        if "image" not in document:
            image_path = image_dir / f"{doc_id}.png"
            example["image"].convert("RGB").save(image_path)
            try:
                document["image"] = str(image_path.relative_to(ROOT))
            except ValueError:
                document["image"] = str(image_path)
        document["questions"].append(
            {
                "question_id": str(example["questionId"]),
                "question": example["question"],
                "answers": example["answers"],
            }
        )

    incomplete = [
        doc_id
        for doc_id, document in documents.items()
        if "image" not in document or len(document["questions"]) < args.k_min
    ]
    if incomplete:
        raise RuntimeError(f"selected documents were incomplete during pass 2: {incomplete}")

    with manifest.open("w", encoding="utf-8") as handle:
        for doc_id in chosen:
            document = documents[doc_id]
            record = {
                "dataset": "DocVQA",
                "dataset_id": DATASET_ID,
                "dataset_revision": args.dataset_revision,
                "source_split": args.source_split,
                "split": args.output_split,
                "sample_id": str(doc_id),
                "image": document["image"],
                "question_ids": [item["question_id"] for item in document["questions"]],
                "questions": document["questions"],
                "task_types": ["OCR"],
                "pair_labels": [],
                "selection_seed": args.seed,
                "exclusion_reason": None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata_path = manifest.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "generator": "vlm_diagnosis.scripts.prep_docvqa",
                "dataset_id": DATASET_ID,
                "dataset_revision": args.dataset_revision,
                "source_split": args.source_split,
                "output_split": args.output_split,
                "selection_seed": args.seed,
                "scan_rows": args.scan_rows,
                "rows_seen": rows_seen,
                "k_min": args.k_min,
                "eligible_documents": eligible_count,
                "selected_documents": len(chosen),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(chosen)} documents to {manifest}")
    print(f"wrote generation metadata to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--output-split", default="diagnostic")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data" / "docvqa_manifest")
    parser.add_argument("--n-docs", type=int, default=32)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--scan-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    build_manifest(parser.parse_args())


if __name__ == "__main__":
    main()
