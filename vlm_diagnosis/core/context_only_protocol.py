"""Validation and non-destructive output contracts for context-only experiments."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
from pathlib import Path

import torch

SCHEMA = "context_only_v2"


def configuration_digest(metadata):
    # Runtime identity and dirty/untracked output files are not experiment settings.
    ignored = {"run_id", "started_at", "record_type", "code_dirty", "configuration_sha256"}
    config = {k: v for k, v in metadata.items() if k not in ignored}
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


class ExperimentLog:
    """Exclusive new output; resume only matching, fully committed context records.

    An interrupted context is never silently deleted, overwritten, or appended twice.
    Such a log is preserved and requires a new output path. flock rejects concurrent writers.
    """

    def __init__(self, path, metadata, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(metadata, record_type="run", schema_version=SCHEMA)
        metadata["configuration_sha256"] = configuration_digest(metadata)
        self.done = set()
        self.stream = None
        exists = self.path.exists()
        try:
            self.stream = self.path.open("r+" if resume and exists else "x+")
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume and exists:
                rows = []
                for number, line in enumerate(self.stream, 1):
                    if not line.endswith("\n"):
                        raise ValueError(f"incomplete JSONL line {number}; use a new --out")
                    try:
                        rows.append(json.loads(line))
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"malformed JSONL line {number}; use a new --out") from exc
                if not rows or rows[0].get("record_type") != "run":
                    raise ValueError("missing run header; use a new --out")
                previous = rows[0]
                if previous.get("schema_version") != SCHEMA or configuration_digest(previous) != metadata["configuration_sha256"]:
                    raise ValueError("resume configuration mismatch; existing output was not modified")
                seen_contexts = set()
                for row in rows[1:]:
                    if row.get("run_id") != previous["run_id"] or row.get("record_type") in ("run", "error"):
                        raise ValueError("mixed/failed run cannot be resumed; use a new --out")
                    cid = str(row.get("context_id", ""))
                    if not cid or cid in self.done:
                        raise ValueError("duplicate or out-of-order completed context; use a new --out")
                    seen_contexts.add(cid)
                    if row["record_type"] == "context_done":
                        self.done.add(cid)
                if seen_contexts != self.done:
                    raise ValueError("interrupted context preserved; resume requires a new --out")
                metadata = previous
                self.stream.seek(0, 2)
            else:
                self.stream.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False) + "\n")
                self.stream.flush()
            self.metadata = metadata
            self.run_id = metadata["run_id"]
        except BaseException:
            if self.stream is not None:
                self.stream.close()
            raise

    def emit(self, record):
        if record.get("run_id", self.run_id) != self.run_id:
            raise ValueError("foreign run record")
        self.stream.write(json.dumps({**record, "run_id": self.run_id}, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def parity_tolerances(dtype, *, logit_atol=None, logit_rtol=None, nll_atol=None):
    # Numerical gates, not evidence of semantic/accuracy equivalence. Record all values.
    defaults = {
        torch.float64: (1e-8, 1e-7, 1e-8),
        torch.float32: (1e-4, 1e-4, 1e-4),
        torch.float16: (0.5, 0.02, 0.05),
        torch.bfloat16: (2.0, 0.05, 0.2),
    }
    if dtype not in defaults:
        raise ValueError(f"unsupported parity dtype {dtype}")
    values = [fallback if value is None else value for value, fallback in zip(
        (logit_atol, logit_rtol, nll_atol), defaults[dtype])]
    if any(not math.isfinite(float(value)) or value < 0 for value in values):
        raise ValueError("parity tolerances must be finite and nonnegative")
    return dict(zip(("logit_atol", "logit_rtol", "nll_atol"), map(float, values)))


def check_full_parity(cached_logits, dense_logits, cached_logp, dense_logp, positions_match, tolerances):
    """Fail closed on positions, all suffix logits, first answer argmax, and each gold logp."""
    tensors = (cached_logits, dense_logits, cached_logp, dense_logp)
    if any(not torch.isfinite(t).all() or t.numel() == 0 for t in tensors):
        raise RuntimeError("parity failed: empty or nonfinite logits/log-probabilities")
    if cached_logits.shape != dense_logits.shape or cached_logp.shape != dense_logp.shape:
        raise RuntimeError("parity failed: shape mismatch")
    if not positions_match:
        raise RuntimeError("parity failed: logical/RoPE positions differ")
    if int(cached_logits[-1].argmax()) != int(dense_logits[-1].argmax()):
        raise RuntimeError("parity failed: first answer token differs")
    if not torch.allclose(cached_logits, dense_logits, atol=tolerances["logit_atol"], rtol=tolerances["logit_rtol"]):
        raise RuntimeError("parity failed: suffix logits exceed recorded tolerances")
    if not torch.allclose(cached_logp, dense_logp, atol=tolerances["nll_atol"], rtol=0):
        raise RuntimeError("parity failed: answer token log-probabilities exceed tolerance")
    if abs(float(cached_logp.mean() - dense_logp.mean())) > tolerances["nll_atol"]:
        raise RuntimeError("parity failed: answer NLL exceeds tolerance")
