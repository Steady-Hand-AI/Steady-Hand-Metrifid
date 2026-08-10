"""Chunked byte comparison of two private complete-MJB artifact files.

Both artifacts can be over a hundred megabytes, so the comparison streams fixed-size chunks and
accumulates counters. It never builds a list of differing offsets and never holds either whole
artifact in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Self, cast

from ..json_values import CanonicalValue
from ..operational import _require_exact_object_fields

BYTE_COMPARISON_SCHEMA = "metrifid.compiled_artifact_byte_comparison"
BYTE_COMPARISON_SCHEMA_VERSION = 1

_CHUNK_BYTES = 1 << 20

_COMPARISON_MEMBERS = (
    "schema",
    "schema_version",
    "equal",
    "baseline_mjb_size_bytes",
    "candidate_mjb_size_bytes",
    "compared_byte_count",
    "first_differing_byte_offset",
    "differing_byte_count",
)


def _validate_byte_counts(comparison: ByteComparison) -> int:
    """Validate byte counts and return the compared overlap."""
    for name in (
        "baseline_mjb_size_bytes",
        "candidate_mjb_size_bytes",
        "compared_byte_count",
        "differing_byte_count",
    ):
        value = getattr(comparison, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    overlap = min(comparison.baseline_mjb_size_bytes, comparison.candidate_mjb_size_bytes)
    if comparison.compared_byte_count != overlap:
        raise ValueError("compared_byte_count must equal the overlap of the two artifacts")
    offset = comparison.first_differing_byte_offset
    if offset is not None and (type(offset) is not int or not 0 <= offset <= overlap):
        raise ValueError("first_differing_byte_offset must lie within the compared range")
    return overlap


def _validate_difference_shape(comparison: ByteComparison) -> None:
    """Validate equal and differing comparison evidence relationships."""
    offset = comparison.first_differing_byte_offset
    if comparison.equal:
        if comparison.differing_byte_count != 0 or offset is not None:
            raise ValueError("an equal comparison reports no differing byte and no offset")
        if comparison.baseline_mjb_size_bytes != comparison.candidate_mjb_size_bytes:
            raise ValueError("an equal comparison requires equal artifact sizes")
        return
    if comparison.differing_byte_count <= 0:
        raise ValueError("a differing comparison reports at least one differing byte")
    length_delta = abs(comparison.baseline_mjb_size_bytes - comparison.candidate_mjb_size_bytes)
    if comparison.differing_byte_count < length_delta:
        raise ValueError("differing_byte_count must account for the length difference")
    if offset is None and length_delta == 0:
        raise ValueError("a differing comparison records a first differing offset")


@dataclass(frozen=True, slots=True)
class ByteComparison:
    """The complete result of comparing two artifacts byte for byte."""

    schema: str
    schema_version: int
    equal: bool
    baseline_mjb_size_bytes: int
    candidate_mjb_size_bytes: int
    compared_byte_count: int
    first_differing_byte_offset: int | None
    differing_byte_count: int

    def __post_init__(self) -> None:
        """Cross-check byte counts, first difference, bounded spans, and artifact digests."""
        if self.schema != BYTE_COMPARISON_SCHEMA:
            raise ValueError("invalid byte comparison schema")
        if (
            type(self.schema_version) is not int
            or self.schema_version != BYTE_COMPARISON_SCHEMA_VERSION
        ):
            raise ValueError("invalid byte comparison schema_version")
        if type(self.equal) is not bool:
            raise TypeError("equal must be a boolean")
        _validate_byte_counts(self)
        _validate_difference_shape(self)

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Parse one byte comparison strictly, requiring its exact member set."""
        obj = _require_exact_object_fields(value, set(_COMPARISON_MEMBERS), "ByteComparison")
        if type(obj["schema"]) is not str:
            raise TypeError("schema must be a string")
        if type(obj["schema_version"]) is not int:
            raise TypeError("schema_version must be an integer")
        if type(obj["equal"]) is not bool:
            raise TypeError("equal must be a boolean")
        for name in (
            "baseline_mjb_size_bytes",
            "candidate_mjb_size_bytes",
            "compared_byte_count",
            "differing_byte_count",
        ):
            if type(obj[name]) is not int:
                raise TypeError(f"{name} must be an integer")
        offset = obj["first_differing_byte_offset"]
        if offset is not None and type(offset) is not int:
            raise TypeError("first_differing_byte_offset must be an integer or null")
        return cls(
            schema=obj["schema"],
            schema_version=obj["schema_version"],
            equal=obj["equal"],
            baseline_mjb_size_bytes=cast(int, obj["baseline_mjb_size_bytes"]),
            candidate_mjb_size_bytes=cast(int, obj["candidate_mjb_size_bytes"]),
            compared_byte_count=cast(int, obj["compared_byte_count"]),
            first_differing_byte_offset=offset,
            differing_byte_count=cast(int, obj["differing_byte_count"]),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit exact byte-comparison counts, witnesses, and role artifact hashes."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "equal": self.equal,
            "baseline_mjb_size_bytes": self.baseline_mjb_size_bytes,
            "candidate_mjb_size_bytes": self.candidate_mjb_size_bytes,
            # Offsets present in both artifacts. Any length difference is counted separately
            # below rather than being silently treated as equal or as a single difference.
            "compared_byte_count": self.compared_byte_count,
            "first_differing_byte_offset": self.first_differing_byte_offset,
            "differing_byte_count": self.differing_byte_count,
        }


def compare_artifact_bytes(baseline: Path, candidate: Path) -> ByteComparison:
    """Compare two complete artifacts in bounded chunks."""
    import numpy as np

    baseline_size = baseline.stat().st_size
    candidate_size = candidate.stat().st_size
    overlap = min(baseline_size, candidate_size)
    first_offset: int | None = None
    differing = abs(baseline_size - candidate_size)
    with baseline.open("rb") as left, candidate.open("rb") as right:
        offset = 0
        while offset < overlap:
            span = min(_CHUNK_BYTES, overlap - offset)
            left_chunk = _exact_read(left, span)
            right_chunk = _exact_read(right, span)
            if left_chunk != right_chunk:
                mask = np.frombuffer(left_chunk, dtype=np.uint8) != np.frombuffer(
                    right_chunk, dtype=np.uint8
                )
                differing += int(np.count_nonzero(mask))
                if first_offset is None:
                    first_offset = offset + int(np.argmax(mask))
            offset += span
    if first_offset is None and baseline_size != candidate_size:
        first_offset = overlap
    return ByteComparison(
        schema=BYTE_COMPARISON_SCHEMA,
        schema_version=BYTE_COMPARISON_SCHEMA_VERSION,
        equal=baseline_size == candidate_size and differing == 0,
        baseline_mjb_size_bytes=baseline_size,
        candidate_mjb_size_bytes=candidate_size,
        compared_byte_count=overlap,
        first_differing_byte_offset=first_offset,
        differing_byte_count=differing,
    )


def _exact_read(stream: BinaryIO, span: int) -> bytes:
    """Read exactly one requested comparison span or fail on premature EOF."""
    chunk = stream.read(span)
    if len(chunk) != span:
        raise OSError("artifact file shrank while it was being compared")
    return bytes(chunk)


__all__ = ["BYTE_COMPARISON_SCHEMA", "ByteComparison", "compare_artifact_bytes"]
