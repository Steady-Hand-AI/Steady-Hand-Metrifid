"""Exact artifact admission physical-time admission and boundary recurrence without simulation stepping."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, TypeAlias

from ._npz import refuse
from .json_values import Binary64, CanonicalValue, ExactRational, canonical_sha256
from .operational import OperationalReasonCode

TimeRole: TypeAlias = Literal["baseline", "candidate"]


def _validate_grid_periods(grid: TimeGrid) -> None:
    """Validate exact and compiled timestep fields."""
    for field, period in (
        ("baseline_step_dt", grid.baseline_step_dt),
        ("candidate_step_dt", grid.candidate_step_dt),
        ("control_dt", grid.control_dt),
    ):
        if not isinstance(period, ExactRational) or period.numerator <= 0:
            raise ValueError(f"{field} must be a strictly positive ExactRational")
    if not isinstance(grid.baseline_compiled_timestep, Binary64):
        raise TypeError("baseline_compiled_timestep must be Binary64")
    if not isinstance(grid.candidate_compiled_timestep, Binary64):
        raise TypeError("candidate_compiled_timestep must be Binary64")
    _require_direct_timestep_match(
        grid.baseline_step_dt, grid.baseline_compiled_timestep, "baseline"
    )
    _require_direct_timestep_match(
        grid.candidate_step_dt, grid.candidate_compiled_timestep, "candidate"
    )


def _validate_grid_counts(grid: TimeGrid) -> None:
    """Validate substep, boundary, and horizon relationships."""
    for field, value in (
        ("baseline_substeps_per_control", grid.baseline_substeps_per_control),
        ("candidate_substeps_per_control", grid.candidate_substeps_per_control),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if type(grid.control_intervals) is not int or not 1 <= grid.control_intervals <= 100_000:
        raise ValueError("control_intervals must be between 1 and 100000")
    if grid.boundary_count != grid.control_intervals + 1:
        raise ValueError("boundary_count must equal control_intervals + 1")
    if grid.horizon != grid.control_dt.multiplied_by_int(grid.control_intervals):
        raise ValueError("horizon must equal control_intervals * control_dt")
    if (
        _integral_ratio(grid.control_dt, grid.baseline_step_dt)
        != grid.baseline_substeps_per_control
    ):
        raise ValueError("baseline substep count is inconsistent")
    if (
        _integral_ratio(grid.control_dt, grid.candidate_step_dt)
        != grid.candidate_substeps_per_control
    ):
        raise ValueError("candidate substep count is inconsistent")


@dataclass(frozen=True, slots=True)
class TimeGrid:
    """One immutable exact control grid and the two role-local binary64 recurrences."""

    baseline_step_dt: ExactRational
    candidate_step_dt: ExactRational
    control_dt: ExactRational
    baseline_compiled_timestep: Binary64
    candidate_compiled_timestep: Binary64
    baseline_substeps_per_control: int
    candidate_substeps_per_control: int
    control_intervals: int
    boundary_count: int
    horizon: ExactRational

    def __post_init__(self) -> None:
        """Validate exact timing fields, positive substep counts, and boundary count."""
        _validate_grid_periods(self)
        _validate_grid_counts(self)

    def sha256(self) -> str:
        """Hash every exact or binary64 field that defines this admitted grid."""
        return canonical_sha256(self.to_primitive())

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the deterministic internal time-grid semantic object."""
        return {
            "schema": "metrifid.time_grid",
            "schema_version": 1,
            "baseline_step_dt": self.baseline_step_dt.to_primitive(),
            "candidate_step_dt": self.candidate_step_dt.to_primitive(),
            "control_dt": self.control_dt.to_primitive(),
            "baseline_compiled_timestep": self.baseline_compiled_timestep.to_primitive(),
            "candidate_compiled_timestep": self.candidate_compiled_timestep.to_primitive(),
            "baseline_substeps_per_control": self.baseline_substeps_per_control,
            "candidate_substeps_per_control": self.candidate_substeps_per_control,
            "control_intervals": self.control_intervals,
            "boundary_count": self.boundary_count,
            "horizon": self.horizon.to_primitive(),
        }

    def iter_canonical_boundaries(self) -> Iterator[ExactRational]:
        """Yield exact canonical boundary times k * control_dt for k in [0, N]."""
        for boundary in range(self.boundary_count):
            yield self.control_dt.multiplied_by_int(boundary)

    def iter_role_boundary_time_bits(self, role: TimeRole) -> Iterator[Binary64]:
        """Yield role-local boundary bits from positive zero and repeated step additions."""
        if role == "baseline":
            step = self.baseline_compiled_timestep.to_float()
            substeps = self.baseline_substeps_per_control
        elif role == "candidate":
            step = self.candidate_compiled_timestep.to_float()
            substeps = self.candidate_substeps_per_control
        else:
            raise ValueError("role must be baseline or candidate")
        current = 0.0
        yield Binary64.from_float(current)
        for _ in range(self.control_intervals):
            for _ in range(substeps):
                current += step
            yield Binary64.from_float(current)


def build_time_grid(
    *,
    baseline_step_dt: ExactRational,
    candidate_step_dt: ExactRational,
    control_dt: ExactRational,
    baseline_compiled_timestep: float,
    candidate_compiled_timestep: float,
    control_intervals: int,
) -> TimeGrid:
    """Admit an exact integral two-role control grid without epsilon or resampling."""
    baseline_bits = _validate_declared_step(
        baseline_step_dt, baseline_compiled_timestep, "baseline"
    )
    candidate_bits = _validate_declared_step(
        candidate_step_dt, candidate_compiled_timestep, "candidate"
    )
    if not isinstance(control_dt, ExactRational) or control_dt.numerator <= 0:
        raise refuse(
            OperationalReasonCode.CONTROL_DT_INVALID,
            issue="control_dt_not_strictly_positive",
        )
    if type(control_intervals) is not int or not 1 <= control_intervals <= 100_000:
        raise refuse(
            OperationalReasonCode.CONTROL_INTERVAL_COUNT_INVALID,
            control_intervals=(
                control_intervals if type(control_intervals) is int else "non_integer"
            ),
        )
    baseline_substeps = _admit_integral_grid(control_dt, baseline_step_dt, "baseline")
    candidate_substeps = _admit_integral_grid(control_dt, candidate_step_dt, "candidate")
    return TimeGrid(
        baseline_step_dt,
        candidate_step_dt,
        control_dt,
        baseline_bits,
        candidate_bits,
        baseline_substeps,
        candidate_substeps,
        control_intervals,
        control_intervals + 1,
        control_dt.multiplied_by_int(control_intervals),
    )


def _require_direct_timestep_match(
    declared: ExactRational, compiled: Binary64, role: TimeRole
) -> None:
    """Require exact equality when a compiled timestep has a direct rational representation."""
    try:
        rounded = float(Fraction(declared.numerator, declared.denominator))
    except OverflowError as exc:
        raise ValueError(f"{role} declared timestep is not representable as binary64") from exc
    if (
        not math.isfinite(rounded)
        or rounded <= 0.0
        or Binary64.from_float(rounded).bits != compiled.bits
    ):
        raise ValueError(f"{role} compiled timestep bits do not match declared timestep")


def _validate_declared_step(
    declared: ExactRational,
    compiled: float,
    role: TimeRole,
) -> Binary64:
    """Match the declared exact timestep to the compiled binary64 value without guessing."""
    if not isinstance(declared, ExactRational) or declared.numerator <= 0:
        raise refuse(
            OperationalReasonCode.DECLARED_STEP_DT_INVALID,
            role,
            issue="declared_step_not_strictly_positive",
        )
    expected = _rounded_positive_binary64(declared, role)
    if type(compiled) is not float or not math.isfinite(compiled) or compiled <= 0.0:
        raise refuse(
            OperationalReasonCode.DECLARED_STEP_DT_MISMATCH,
            role,
            issue="compiled_timestep_invalid",
        )
    actual = Binary64.from_float(compiled)
    if actual.bits != expected.bits:
        raise refuse(
            OperationalReasonCode.DECLARED_STEP_DT_MISMATCH,
            role,
            declared_bits=expected.bits,
            compiled_bits=actual.bits,
        )
    return actual


def _rounded_positive_binary64(period: ExactRational, role: TimeRole) -> Binary64:
    """Round an exact positive period to finite positive binary64 for comparison."""
    try:
        rounded = float(Fraction(period.numerator, period.denominator))
    except OverflowError as exc:
        raise refuse(
            OperationalReasonCode.DECLARED_STEP_DT_INVALID,
            role,
            issue="declared_step_overflows_binary64",
        ) from exc
    if not math.isfinite(rounded) or rounded <= 0.0:
        raise refuse(
            OperationalReasonCode.DECLARED_STEP_DT_INVALID,
            role,
            issue="declared_step_not_positive_finite_binary64",
        )
    return Binary64.from_float(rounded)


def _admit_integral_grid(
    control_dt: ExactRational,
    step_dt: ExactRational,
    role: TimeRole,
) -> int:
    """Require an exact positive integer number of simulator steps per control interval."""
    ratio = _integral_ratio(control_dt, step_dt)
    if ratio is None:
        raise refuse(
            OperationalReasonCode.CONTROL_GRID_NONINTEGRAL,
            role,
            control_dt=control_dt.to_primitive(),
            step_dt=step_dt.to_primitive(),
        )
    return ratio


def _integral_ratio(control_dt: ExactRational, step_dt: ExactRational) -> int | None:
    """Return the exact integral control/step ratio, or ``None`` when nonintegral."""
    numerator = control_dt.numerator * step_dt.denominator
    denominator = control_dt.denominator * step_dt.numerator
    quotient, remainder = divmod(numerator, denominator)
    if remainder or quotient <= 0:
        return None
    return quotient


__all__ = ["TimeGrid", "build_time_grid"]
