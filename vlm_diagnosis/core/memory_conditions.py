"""Factorial registry for visual-memory storage and reuse experiments.

This module enumerates the condition space described in PLAN.md section 1.2.
It does not run model inference.  The registry makes every experimental factor
explicit and rejects combinations that cannot be executed as written.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


PAYLOAD_ATOMS = (
    "I",
    "T_o",
    "T_d",
    "T_u",
    "T_q",
    "T_a",
    "T_out",
    "T_traj",
    "K_p",
    "K_v",
    "K_w",
    "Z",
)
VISUAL_TEXT_ATOMS = frozenset({"T_o", "T_d", "T_u"})
EPISODE_TEXT_ATOMS = frozenset({"T_q", "T_a", "T_out", "T_traj"})
TEXT_ATOMS = VISUAL_TEXT_ATOMS | EPISODE_TEXT_ATOMS
KV_ATOMS = frozenset({"K_p", "K_v", "K_w"})

BUILD_MODES = (
    "B0_GENERIC",
    "BW_QUERY",
    "BW_ANSWER",
    "BR_QUERY",
    "BR_ANSWER_PROBE",
)

READ_MODES = (
    "R_NONE",
    "R_PREFILL_I",
    "R_PREFILL_T",
    "R_PREFILL_I_THEN_T",
    "R_PREFILL_T_THEN_I",
    "R_INJECT_K",
    "R_INJECT_K_THEN_T",
    "R_PREFILL_T_THEN_INJECT_K",
    "R_PREFILL_I_THEN_INJECT_K",
    "R_INJECT_K_THEN_PREFILL_I",
    "R_PREFILL_I_THEN_T_THEN_INJECT_K",
    "R_PREFILL_I_THEN_INJECT_K_THEN_T",
    "R_PREFILL_T_THEN_I_THEN_INJECT_K",
    "R_PREFILL_T_THEN_INJECT_K_THEN_I",
    "R_INJECT_K_THEN_PREFILL_I_THEN_T",
    "R_INJECT_K_THEN_PREFILL_T_THEN_I",
)

READ_REQUIREMENTS = {
    "R_NONE": frozenset(),
    "R_PREFILL_I": frozenset({"image"}),
    "R_PREFILL_T": frozenset({"text"}),
    "R_PREFILL_I_THEN_T": frozenset({"image", "text"}),
    "R_PREFILL_T_THEN_I": frozenset({"image", "text"}),
    "R_INJECT_K": frozenset({"kv"}),
    "R_INJECT_K_THEN_T": frozenset({"kv", "text"}),
    "R_PREFILL_T_THEN_INJECT_K": frozenset({"kv", "text"}),
    "R_PREFILL_I_THEN_INJECT_K": frozenset({"image", "kv"}),
    "R_INJECT_K_THEN_PREFILL_I": frozenset({"image", "kv"}),
    "R_PREFILL_I_THEN_T_THEN_INJECT_K": frozenset({"image", "text", "kv"}),
    "R_PREFILL_I_THEN_INJECT_K_THEN_T": frozenset({"image", "text", "kv"}),
    "R_PREFILL_T_THEN_I_THEN_INJECT_K": frozenset({"image", "text", "kv"}),
    "R_PREFILL_T_THEN_INJECT_K_THEN_I": frozenset({"image", "text", "kv"}),
    "R_INJECT_K_THEN_PREFILL_I_THEN_T": frozenset({"image", "text", "kv"}),
    "R_INJECT_K_THEN_PREFILL_T_THEN_I": frozenset({"image", "text", "kv"}),
}

POSITIONS = (
    "same_sequence_same_offset",
    "same_offset_context_change",
    "offset_shift",
    "context_and_offset_change",
)

BLOCK_MODES = (
    "single",
    "independent_2",
    "independent_4",
    "joint_2",
    "joint_4",
    "relevant_plus_irrelevant",
)

CORE_PAYLOADS = (
    (),
    ("I",),
    ("T_o",),
    ("T_d",),
    ("T_u",),
    ("T_o", "T_d", "T_u"),
    ("T_q",),
    ("T_a",),
    ("T_out",),
    ("T_traj",),
    ("T_q", "T_a", "T_out", "T_traj"),
    ("I", "T_o", "T_d", "T_u"),
    ("I", "T_q", "T_a", "T_out", "T_traj"),
    ("K_v",),
    ("K_p", "K_v"),
    ("K_p", "K_v", "K_w"),
    ("K_p", "K_v", "T_o", "T_d", "T_u"),
    ("K_p", "K_v", "T_q", "T_a", "T_out", "T_traj"),
    ("I", "K_v"),
    ("I", "K_p", "K_v"),
    (
        "I",
        "T_o",
        "T_d",
        "T_u",
        "T_q",
        "T_a",
        "T_out",
        "T_traj",
        "K_p",
        "K_v",
    ),
    (
        "I",
        "T_o",
        "T_d",
        "T_u",
        "T_q",
        "T_a",
        "T_out",
        "T_traj",
        "K_p",
        "K_v",
        "K_w",
    ),
)


def _ordered_payload(payload: Iterable[str]) -> tuple[str, ...]:
    payload_set = set(payload)
    unknown = payload_set.difference(PAYLOAD_ATOMS)
    if unknown:
        raise ValueError(f"unknown payload atom(s): {sorted(unknown)}")
    return tuple(atom for atom in PAYLOAD_ATOMS if atom in payload_set)


@dataclass(frozen=True)
class MemoryCondition:
    payload: tuple[str, ...]
    build_mode: str
    read_mode: str
    position: str
    blocks: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _ordered_payload(self.payload))

    @property
    def diagnostic_only(self) -> bool:
        return self.build_mode == "BR_ANSWER_PROBE"

    @property
    def answer_carryover(self) -> bool:
        return "T_a" in self.payload

    @property
    def condition_id(self) -> str:
        payload = ",".join(self.payload) if self.payload else "EMPTY"
        return (
            f"store[{payload}]__build[{self.build_mode}]__read[{self.read_mode}]"
            f"__position[{self.position}]__blocks[{self.blocks}]"
        )

    def validation_errors(self) -> list[str]:
        p = set(self.payload)
        has_image = "I" in p
        has_text = bool(p & TEXT_ATOMS)
        has_kv = bool(p & KV_ATOMS)
        requirements = READ_REQUIREMENTS.get(self.read_mode, frozenset())
        reads_image = "image" in requirements
        reads_text = "text" in requirements
        injects_kv = "kv" in requirements
        moved = self.position in {"offset_shift", "context_and_offset_change"}
        multiple_blocks = self.blocks != "single"

        errors: list[str] = []
        if self.build_mode not in BUILD_MODES:
            errors.append("unknown build mode")
        if self.read_mode not in READ_MODES:
            errors.append("unknown read mode")
        if self.position not in POSITIONS:
            errors.append("unknown position mode")
        if self.blocks not in BLOCK_MODES:
            errors.append("unknown block mode")
        if not p and self.read_mode != "R_NONE":
            errors.append("empty payload cannot be read")
        if p and self.read_mode == "R_NONE":
            errors.append("stored payload is unused")
        if reads_image and not has_image:
            errors.append("image prefill requires I")
        if reads_text and not has_text:
            errors.append("text prefill requires at least one T_*")
        if injects_kv and not has_kv:
            errors.append("KV injection requires at least one K_*")
        if has_image and not reads_image:
            errors.append("stored I is unused by read mode")
        if has_text and not reads_text:
            errors.append("stored T_* is unused by read mode")
        if has_kv and not injects_kv:
            errors.append("stored K_* is unused by read mode")
        if "K_w" in p and not ({"K_p", "K_v"} & p):
            errors.append("K_w requires a preceding prefix or visual KV block")
        if injects_kv and (moved or multiple_blocks) and "Z" not in p:
            errors.append("moved/composed KV requires Z metadata")
        if self.read_mode == "R_NONE" and (
            self.position != "same_sequence_same_offset" or self.blocks != "single"
        ):
            errors.append("no-memory control has no position/block manipulation")
        if self.read_mode == "R_NONE" and self.build_mode != "B0_GENERIC":
            errors.append("no-memory control uses only the generic build label")
        if not injects_kv and self.position == "offset_shift":
            errors.append("pure offset-shift condition is defined for injected KV")
        if not injects_kv and self.position == "context_and_offset_change":
            errors.append("context+offset condition is defined for injected KV")
        return errors

    @property
    def valid(self) -> bool:
        return not self.validation_errors()

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record.update(
            condition_id=self.condition_id,
            diagnostic_only=self.diagnostic_only,
            answer_carryover=self.answer_carryover,
            valid=self.valid,
            validation_errors=self.validation_errors(),
        )
        return record


def payload_powerset(*, include_position_metadata: bool = True) -> Iterator[tuple[str, ...]]:
    atoms = PAYLOAD_ATOMS if include_position_metadata else PAYLOAD_ATOMS[:-1]
    for size in range(len(atoms) + 1):
        yield from itertools.combinations(atoms, size)


def generate_conditions(
    *,
    scope: str = "core",
    include_invalid: bool = False,
) -> Iterator[MemoryCondition]:
    if scope == "core":
        payloads: Iterable[Sequence[str]] = CORE_PAYLOADS
    elif scope == "full":
        payloads = payload_powerset()
    else:
        raise ValueError("scope must be 'core' or 'full'")

    seen: set[str] = set()
    for payload, build, read, position, blocks in itertools.product(
        payloads, BUILD_MODES, READ_MODES, POSITIONS, BLOCK_MODES
    ):
        # Z is required metadata, not a standalone representation. Add it to
        # executable moved/composed KV conditions instead of duplicating every
        # canonical payload in CORE_PAYLOADS.
        p = set(payload)
        if scope == "core" and ("kv" in READ_REQUIREMENTS[read]) and (
            position in {"offset_shift", "context_and_offset_change"} or blocks != "single"
        ):
            p.add("Z")
        condition = MemoryCondition(tuple(p), build, read, position, blocks)
        if condition.condition_id in seen:
            continue
        seen.add(condition.condition_id)
        if include_invalid or condition.valid:
            yield condition


def write_registry(path: Path, conditions: Iterable[MemoryCondition]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for condition in conditions:
            handle.write(json.dumps(condition.to_record(), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("core", "full"), default="core")
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = write_registry(
        args.output,
        generate_conditions(scope=args.scope, include_invalid=args.include_invalid),
    )
    print(f"wrote {count} conditions to {args.output}")


if __name__ == "__main__":
    main()
