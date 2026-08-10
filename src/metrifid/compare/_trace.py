"""Immutable trace evidence and exact repeat signatures for comparison."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ..json_values import Binary64, CanonicalValue, ExactRational, canonical_sha256

Role: TypeAlias = Literal["baseline", "candidate"]


@dataclass(slots=True)
class _BoundaryRows:
    """The per-boundary arrays one role accumulates while it replays."""

    qpos_rows: list[npt.NDArray[np.float64]]
    qvel_rows: list[npt.NDArray[np.float64]]
    act_rows: list[npt.NDArray[np.float64]]
    boundaries: list[int]
    canonical_times: list[ExactRational]
    observed_times: list[Binary64]
    warnings: list[tuple[tuple[int, int, int], ...]]

    @classmethod
    def empty(cls) -> _BoundaryRows:
        """Return an empty mutable boundary accumulator for one replay."""
        return cls([], [], [], [], [], [], [])


def _trace_boundary_count(trace: RoleTrace) -> int:
    """Validate scalar trace dimensions and return the captured count."""
    if trace.role not in {"baseline", "candidate"}:
        raise ValueError("role must be baseline or candidate")
    if type(trace.expected_boundary_count) is not int or trace.expected_boundary_count < 2:
        raise ValueError("expected_boundary_count must be at least two")
    count = len(trace.boundary_indices)
    if trace.boundary_indices != tuple(range(count)):
        raise ValueError("boundary indices must be the contiguous captured prefix")
    scalar_counts = (
        len(trace.canonical_times),
        len(trace.observed_time_bits),
        len(trace.warning_snapshots),
    )
    if any(value != count for value in scalar_counts):
        raise ValueError("trace scalar evidence lengths must match")
    return count


def _validate_trace_arrays(trace: RoleTrace, count: int) -> None:
    """Validate immutable, boundary-aligned trace arrays."""
    for field, array in (("qpos", trace.qpos), ("qvel", trace.qvel), ("act", trace.act)):
        if not isinstance(array, np.ndarray) or array.dtype.str != "<f8" or array.ndim != 2:
            raise TypeError(f"{field} must be a two-dimensional little-endian float64 array")
        if array.shape[0] != count or not array.flags.c_contiguous or array.flags.writeable:
            raise ValueError(f"{field} must be read-only, C-contiguous, and boundary-aligned")


def _validate_invalid_trace_evidence(trace: RoleTrace, count: int) -> None:
    """Validate invalid-trace evidence and captured error messages."""
    if trace.invalid_kind is None:
        if trace.invalid_boundary_index is not None:
            raise ValueError("invalid boundary requires invalid_kind")
    else:
        if not trace.invalid_kind or trace.invalid_boundary_index is None:
            raise ValueError("invalid trace evidence is incomplete")
        if trace.invalid_boundary_index < 0 or trace.invalid_boundary_index > count:
            raise ValueError("invalid boundary is outside the captured prefix")
    if type(trace.initial_state_preserved) is not bool:
        raise TypeError("initial_state_preserved must be a boolean")
    for message in trace.error_logs:
        if not isinstance(message, str) or not message:
            raise ValueError("error logs must be nonempty strings")


@dataclass(frozen=True, slots=True, eq=False)
class RoleTrace:
    """One role-local repeat with only complete captured boundaries."""

    role: Role
    expected_boundary_count: int
    boundary_indices: tuple[int, ...]
    canonical_times: tuple[ExactRational, ...]
    observed_time_bits: tuple[Binary64, ...]
    qpos: npt.NDArray[np.float64]
    qvel: npt.NDArray[np.float64]
    act: npt.NDArray[np.float64]
    warning_snapshots: tuple[tuple[tuple[int, int, int], ...], ...]
    error_logs: tuple[str, ...]
    invalid_kind: str | None
    invalid_boundary_index: int | None
    initial_state_preserved: bool

    def __post_init__(self) -> None:
        """Validate boundary alignment, immutable arrays, and invalid-trace evidence."""
        count = _trace_boundary_count(self)
        _validate_trace_arrays(self, count)
        _validate_invalid_trace_evidence(self, count)

    @property
    def complete(self) -> bool:
        """Return whether replay completed every declared control interval."""
        return (
            len(self.boundary_indices) == self.expected_boundary_count
            and self.invalid_kind is None
            and self.initial_state_preserved
        )

    @property
    def has_warning(self) -> bool:
        """Return whether any captured MuJoCo warning counter is nonzero."""
        return any(snapshot for snapshot in self.warning_snapshots)

    def signature_sha256(self) -> str:
        """Hash every decision-bearing trace field using exact array bytes."""
        return canonical_sha256(self.signature_primitive())

    def signature_primitive(self) -> dict[str, CanonicalValue]:
        """Emit every decision-bearing trace field for exact repeat hashing."""
        return {
            "schema": "metrifid.role_trace_signature",
            "schema_version": 1,
            "role": self.role,
            "expected_boundary_count": self.expected_boundary_count,
            "boundary_indices": list(self.boundary_indices),
            "canonical_times": [item.to_primitive() for item in self.canonical_times],
            "observed_time_bits": [item.to_primitive() for item in self.observed_time_bits],
            "qpos": _array_identity(self.qpos),
            "qvel": _array_identity(self.qvel),
            "act": _array_identity(self.act),
            "warning_snapshots": [
                [
                    {"warning_index": item[0], "number": item[1], "lastinfo": item[2]}
                    for item in snapshot
                ]
                for snapshot in self.warning_snapshots
            ],
            "error_logs": list(self.error_logs),
            "invalid_kind": self.invalid_kind,
            "invalid_boundary_index": self.invalid_boundary_index,
            "initial_state_preserved": self.initial_state_preserved,
        }


@dataclass(frozen=True, slots=True)
class RoleRepeatSet:
    """Group all independent replay traces for one comparison role."""

    role: Role
    traces: tuple[RoleTrace, ...]

    def __post_init__(self) -> None:
        """Require the configured nonempty repeat count and typed role traces."""
        if self.role not in {"baseline", "candidate"}:
            raise ValueError("role must be baseline or candidate")
        if len(self.traces) < 2 or any(trace.role != self.role for trace in self.traces):
            raise ValueError("repeat set requires at least two traces for one role")

    @property
    def signatures(self) -> tuple[str, ...]:
        """Return exact decision-bearing signatures in replay order."""
        return tuple(trace.signature_sha256() for trace in self.traces)

    @property
    def stable(self) -> bool:
        """Return whether every repeat produced the same exact trace signature."""
        return len(set(self.signatures)) == 1

    @property
    def representative(self) -> RoleTrace:
        """Return the first replay trace as the role's metric representative."""
        return self.traces[0]


def readonly_matrix(rows: list[npt.NDArray[np.float64]], width: int) -> npt.NDArray[np.float64]:
    """Build a read-only little-endian float64 matrix from complete boundary rows."""
    if rows:
        result = np.ascontiguousarray(np.stack(rows), dtype="<f8")
    else:
        result = np.empty((0, width), dtype="<f8")
    result.setflags(write=False)
    return cast(npt.NDArray[np.float64], result)


def build_role_trace(
    rows: _BoundaryRows,
    role: Role,
    expected_boundary_count: int,
    widths: tuple[int, int, int],
    error_logs: list[str],
    invalid_kind: str | None,
    invalid_boundary: int | None,
    initial_state_preserved: bool,
) -> RoleTrace:
    """Freeze accumulated boundary rows into one immutable role trace."""
    return RoleTrace(
        role=role,
        expected_boundary_count=expected_boundary_count,
        boundary_indices=tuple(rows.boundaries),
        canonical_times=tuple(rows.canonical_times),
        observed_time_bits=tuple(rows.observed_times),
        qpos=readonly_matrix(rows.qpos_rows, widths[0]),
        qvel=readonly_matrix(rows.qvel_rows, widths[1]),
        act=readonly_matrix(rows.act_rows, widths[2]),
        warning_snapshots=tuple(rows.warnings),
        error_logs=tuple(error_logs),
        invalid_kind=invalid_kind,
        invalid_boundary_index=invalid_boundary,
        initial_state_preserved=initial_state_preserved,
    )


def _array_identity(array: npt.NDArray[np.float64]) -> dict[str, CanonicalValue]:
    """Describe a trace array by dtype, shape, and exact C-order byte hash."""
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


__all__ = ["RoleRepeatSet", "RoleTrace", "Role", "readonly_matrix"]
