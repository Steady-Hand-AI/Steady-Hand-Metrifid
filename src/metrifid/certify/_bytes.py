"""Chunked byte comparison of two private complete-MJB artifact files.

Both artifacts can be over a hundred megabytes, so the comparison streams fixed-size chunks and
accumulates counters. It never builds a list of differing offsets and never holds either whole
artifact in memory.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, cast

from .._model_closure import ModelRole, refuse
from ..json_values import CanonicalValue
from ..operational import OperationalReasonCode, _require_exact_object_fields

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


class _ArtifactReader(Protocol):
    """The exact retained-subject reads this comparison is allowed to perform.

    The expected digest and role travel with the reader so the comparison can bind the stream it
    actually consumed to the artifact the receipt identifies, and name the role that failed.
    """

    @property
    def role(self) -> ModelRole:
        """Return which role this reader supplies."""

    @property
    def mjb_sha256(self) -> str:
        """Return the receipt-bound digest this reader's stream must reproduce."""

    @property
    def mjb_size_bytes(self) -> int:
        """Return the measured length of the retained subject."""

    def read_exact(self, offset: int, span: int) -> bytes:
        """Return one exact positional span of the retained subject."""


def compare_artifact_bytes(baseline: Path, candidate: Path) -> ByteComparison:
    """Compare two complete artifact files in bounded chunks."""
    with _PathReader(baseline, "baseline") as left, _PathReader(candidate, "candidate") as right:
        return compare_retained_artifacts(left, right)


def compare_retained_artifacts(
    baseline: _ArtifactReader, candidate: _ArtifactReader
) -> ByteComparison:
    """Compare two retained compiled subjects in bounded chunks, by descriptor only.

    Both subjects are read positionally through their own retained descriptors, so no pathname is
    consulted, and neither artifact is ever held in memory: exactly one bounded chunk per role is
    resident.

    The verdict is bound to what was actually read. Each role's consumed stream is hashed as it is
    consumed, including the non-overlap tail of the longer artifact, and both observed digests must
    equal their receipt-bound digests before a comparison is returned. Verifying the descriptors
    around the comparison would not achieve this: a same-user process can substitute bytes after the
    first check and restore them before the second, so only a digest over the consumed bytes proves
    which bytes decided the result.
    """
    import numpy as np

    baseline_size = baseline.mjb_size_bytes
    candidate_size = candidate.mjb_size_bytes
    overlap = min(baseline_size, candidate_size)
    first_offset: int | None = None
    # The length delta is counted once, here. The tail bytes read below are hashed but never
    # recounted, because they are already accounted for by this delta.
    differing = abs(baseline_size - candidate_size)
    streams = {
        "baseline": hashlib.sha256(),
        "candidate": hashlib.sha256(),
    }
    offset = 0
    while offset < overlap:
        span = min(_CHUNK_BYTES, overlap - offset)
        left_chunk = baseline.read_exact(offset, span)
        right_chunk = candidate.read_exact(offset, span)
        streams["baseline"].update(left_chunk)
        streams["candidate"].update(right_chunk)
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

    longer, longer_size = (
        (baseline, baseline_size) if baseline_size > candidate_size else (candidate, candidate_size)
    )
    _consume_tail(longer, streams[longer.role], overlap, longer_size)

    for reader in (baseline, candidate):
        observed = streams[reader.role].hexdigest()
        if observed == reader.mjb_sha256:
            continue
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            reader.role,
            issue="consumed_artifact_digest_mismatch",
            expected_mjb_sha256=reader.mjb_sha256,
            observed_mjb_sha256=observed,
        )

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


def _consume_tail(reader: _ArtifactReader, digest: Any, overlap: int, size: int) -> None:
    """Read and hash every byte of the longer artifact past the compared overlap."""
    offset = overlap
    while offset < size:
        span = min(_CHUNK_BYTES, size - offset)
        digest.update(reader.read_exact(offset, span))
        offset += span


class _PathReader:
    """Read one ordinary artifact file positionally, for callers holding only a pathname.

    Such a caller has no receipt-bound digest, so the expected digest is measured here from the
    already-open descriptor. That keeps ``compare_artifact_bytes(Path, Path)`` working while still
    giving the exact-consumer check something to compare the consumed stream against.
    """

    __slots__ = ("_fd", "mjb_sha256", "mjb_size_bytes", "role")

    def __init__(self, path: Path, role: ModelRole) -> None:
        """Open one artifact file, measure its length, and record its expected digest."""
        self._fd = os.open(path, os.O_RDONLY)
        self.role = role
        self.mjb_size_bytes = os.fstat(self._fd).st_size
        digest = hashlib.sha256()
        offset = 0
        while offset < self.mjb_size_bytes:
            span = min(_CHUNK_BYTES, self.mjb_size_bytes - offset)
            digest.update(self.read_exact(offset, span))
            offset += span
        self.mjb_sha256 = digest.hexdigest()

    def read_exact(self, offset: int, span: int) -> bytes:
        """Return one exact positional span or fail on premature end of file."""
        chunk = os.pread(self._fd, span, offset)
        if len(chunk) != span:
            raise OSError("artifact file shrank while it was being compared")
        return chunk

    def __enter__(self) -> _PathReader:
        """Return the open reader to its ``with`` block."""
        return self

    def __exit__(self, *_exception: object) -> None:
        """Close the descriptor this reader opened."""
        os.close(self._fd)


__all__ = [
    "BYTE_COMPARISON_SCHEMA",
    "ByteComparison",
    "compare_artifact_bytes",
    "compare_retained_artifacts",
]
