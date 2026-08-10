"""Numerical primitives for deterministic comparison metrics."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import numpy as np

from .._alignment import AlignedJoint
from .._timegrid import TimeGrid
from ..errors import ReasonCode, ReasonRecord, ReasonRole
from ..json_values import (
    Binary64,
    CanonicalValue,
    ExactRational,
    FrozenCanonicalObject,
    freeze_canonical,
)


def _canonical_offsets(joints: tuple[AlignedJoint, ...]) -> dict[str, tuple[int, int]]:
    """Assign each aligned joint's qpos and qvel offsets in retained trace rows."""
    result: dict[str, tuple[int, int]] = {}
    qpos = 0
    qvel = 0
    for joint in joints:
        result[joint.canonical_name] = (qpos, qvel)
        qpos += joint.baseline_qpos[1]
        qvel += joint.baseline_qvel[1]
    return result


def _metric_specs(
    joint_type: str, qpos_offset: int, qvel_offset: int
) -> tuple[tuple[str, str, int, bool], ...]:
    """Return joint-type-specific metric channels, offsets, and angular wrapping rules."""
    if joint_type == "HINGE":
        return (
            ("angle_rad", "qpos", qpos_offset, True),
            ("angular_velocity_rad_s", "qvel", qvel_offset, False),
        )
    if joint_type == "SLIDE":
        return (
            ("translation_m", "qpos", qpos_offset, False),
            ("linear_velocity_m_s", "qvel", qvel_offset, False),
        )
    raise ValueError("comparison metrics support only aligned HINGE and SLIDE joints")


def _absolute_errors(left: np.ndarray, right: np.ndarray, *, wrapped: bool) -> np.ndarray:
    """Compute elementwise absolute error, wrapping angular differences when required."""
    delta = right - left
    if wrapped:
        return cast("np.ndarray", np.abs(np.arctan2(np.sin(delta), np.cos(delta))))
    return cast("np.ndarray", np.abs(delta))


def _summarize_metric(
    values: np.ndarray,
    tolerance: ExactRational,
    time_grid: TimeGrid,
    joint_name: str,
    metric: str,
) -> tuple[dict[str, CanonicalValue], dict[str, CanonicalValue] | None]:
    """Summarize maximum error and first strict tolerance crossing for one metric."""
    max_index = int(np.argmax(values))
    max_error = float(values[max_index])
    first_index = next(
        (index for index, value in enumerate(values) if _strictly_exceeds(float(value), tolerance)),
        None,
    )
    row: dict[str, CanonicalValue] = {
        "maximum_error": Binary64.from_float(max_error).to_primitive(),
        "worst_boundary_index": max_index,
        "worst_time": time_grid.control_dt.multiplied_by_int(max_index).to_primitive(),
        "first_crossing_boundary_index": first_index,
        "first_crossing_time": (
            None
            if first_index is None
            else time_grid.control_dt.multiplied_by_int(first_index).to_primitive()
        ),
        "tolerance": tolerance.to_primitive(),
        "maximum_ratio": _ratio(max_error, tolerance).to_primitive(),
    }
    if first_index is None:
        return row, None
    error = float(values[first_index])
    return row, {
        "joint_name": joint_name,
        "metric": metric,
        "boundary_index": first_index,
        "time": time_grid.control_dt.multiplied_by_int(first_index).to_primitive(),
        "error": Binary64.from_float(error).to_primitive(),
        "tolerance": tolerance.to_primitive(),
        "ratio": _ratio(error, tolerance).to_primitive(),
    }


def _strictly_exceeds(value: float, tolerance: ExactRational) -> bool:
    """Compare binary64 error to an exact rational tolerance without rounding ambiguity."""
    numerator, denominator = value.as_integer_ratio()
    return numerator * tolerance.denominator > tolerance.numerator * denominator


def _ratio(value: float, tolerance: ExactRational) -> Binary64:
    """Return the dimensionless binary64 error-to-exact-tolerance ratio."""
    exact_tolerance = Fraction(tolerance.numerator, tolerance.denominator)
    return Binary64.from_float(float(Fraction(*value.as_integer_ratio()) / exact_tolerance))


def _build_metric_reason_record(
    code: ReasonCode,
    role: str,
    *,
    object_type: str | None = None,
    object_name: str | None = None,
    metric: str | None = None,
    boundary_index: int | None = None,
    evidence: dict[str, CanonicalValue],
) -> ReasonRecord:
    """Build a strictly typed metric reason with canonical evidence."""
    frozen = freeze_canonical(evidence)
    return ReasonRecord(
        code=code,
        role=cast(ReasonRole, role),
        object_type=object_type,
        object_name=object_name,
        metric=metric,
        boundary_index=boundary_index,
        evidence=cast(FrozenCanonicalObject, frozen),
    )
