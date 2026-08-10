"""Exact preexecution step and retained-trace budgets for the comparison comparator."""

from __future__ import annotations

from collections.abc import Sequence as _Sequence
from typing import cast as _cast

from .._model_closure import (
    AlignedActuator as _AlignedActuator,
)
from .._model_closure import (
    AlignedJoint as _AlignedJoint,
)
from .._timegrid import TimeGrid as _TimeGrid
from ..errors import (
    ReasonCode as _ReasonCode,
)
from ..errors import (
    ReasonRecord as _ReasonRecord,
)
from ..errors import (
    ordered_reasons as _ordered_reasons,
)
from ..json_values import CanonicalValue as _CanonicalValue
from ..json_values import FrozenCanonicalObject as _FrozenCanonicalObject
from ..json_values import freeze_canonical as _freeze_canonical

MAX_TOTAL_INTERNAL_STEPS = 10_000_000
MAX_TRACE_FLOAT64_BYTES = 268_435_456

_ROLE_COUNT = 2
_FLOAT64_BYTES_PER_VALUE = 8


def evaluate_preexecution_budgets(
    time_grid: _TimeGrid,
    repeats: int,
    aligned_joints: _Sequence[_AlignedJoint],
    aligned_actuators: _Sequence[_AlignedActuator],
) -> tuple[_ReasonRecord, ...]:
    """Return exact preexecution budget reasons without stepping or allocating traces."""
    if not isinstance(time_grid, _TimeGrid):
        raise TypeError("time_grid must be a TimeGrid")
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("repeats must be a positive integer")

    qpos_width, qvel_width, activation_width = _canonical_trace_widths(
        aligned_joints, aligned_actuators
    )
    requested_internal_steps, requested_trace_bytes = _budget_requests(
        time_grid, repeats, qpos_width, qvel_width, activation_width
    )
    reasons: list[_ReasonRecord] = []
    if requested_internal_steps > MAX_TOTAL_INTERNAL_STEPS:
        reasons.append(_step_budget_reason(time_grid, repeats, requested_internal_steps))
    if requested_trace_bytes > MAX_TRACE_FLOAT64_BYTES:
        reasons.append(
            _trace_budget_reason(
                time_grid, repeats, qpos_width, qvel_width, activation_width, requested_trace_bytes
            )
        )
    return _ordered_reasons(reasons)


def _budget_requests(
    time_grid: _TimeGrid,
    repeats: int,
    qpos_width: int,
    qvel_width: int,
    activation_width: int,
) -> tuple[int, int]:
    """Compute exact internal-step and retained-trace requests."""
    steps = (
        time_grid.control_intervals
        * repeats
        * (time_grid.baseline_substeps_per_control + time_grid.candidate_substeps_per_control)
    )
    trace_bytes = (
        time_grid.boundary_count
        * repeats
        * _ROLE_COUNT
        * _FLOAT64_BYTES_PER_VALUE
        * (qpos_width + qvel_width + activation_width)
    )
    return steps, trace_bytes


def _step_budget_reason(time_grid: _TimeGrid, repeats: int, requested: int) -> _ReasonRecord:
    """Build the exact internal-step budget refusal reason."""
    return _comparison_reason(
        _ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED,
        {
            "baseline_substeps_per_control": time_grid.baseline_substeps_per_control,
            "candidate_substeps_per_control": time_grid.candidate_substeps_per_control,
            "control_intervals": time_grid.control_intervals,
            "maximum_total_internal_steps": MAX_TOTAL_INTERNAL_STEPS,
            "repeats": repeats,
            "requested_total_internal_steps": requested,
        },
    )


def _trace_budget_reason(
    time_grid: _TimeGrid,
    repeats: int,
    qpos_width: int,
    qvel_width: int,
    activation_width: int,
    requested: int,
) -> _ReasonRecord:
    """Build the exact retained-trace memory budget refusal reason."""
    return _comparison_reason(
        _ReasonCode.TRACE_MEMORY_BUDGET_EXCEEDED,
        {
            "boundary_count": time_grid.boundary_count,
            "canonical_activation_width": activation_width,
            "canonical_qpos_width": qpos_width,
            "canonical_qvel_width": qvel_width,
            "float64_bytes_per_value": _FLOAT64_BYTES_PER_VALUE,
            "maximum_trace_float64_bytes": MAX_TRACE_FLOAT64_BYTES,
            "repeats": repeats,
            "role_count": _ROLE_COUNT,
            "requested_trace_float64_bytes": requested,
        },
    )


def _canonical_trace_widths(
    joints: _Sequence[_AlignedJoint],
    actuators: _Sequence[_AlignedActuator],
) -> tuple[int, int, int]:
    """Compute retained qpos, qvel, and activation widths from canonical alignment."""
    qpos_width, qvel_width = _joint_trace_widths(joints)
    activation_width = sum(_actuator_trace_width(actuator) for actuator in actuators)
    return qpos_width, qvel_width, activation_width


def _joint_trace_widths(joints: _Sequence[_AlignedJoint]) -> tuple[int, int]:
    """Validate aligned joints and return their canonical position and velocity widths."""
    qpos_width = 0
    qvel_width = 0
    for joint in joints:
        if not isinstance(joint, _AlignedJoint):
            raise TypeError("aligned_joints must contain AlignedJoint values")
        baseline_qpos = _slice_width(joint.baseline_qpos, "baseline_qpos")
        candidate_qpos = _slice_width(joint.candidate_qpos, "candidate_qpos")
        baseline_qvel = _slice_width(joint.baseline_qvel, "baseline_qvel")
        candidate_qvel = _slice_width(joint.candidate_qvel, "candidate_qvel")
        if baseline_qpos != candidate_qpos:
            raise ValueError("aligned joint qpos widths are incompatible")
        if baseline_qvel != candidate_qvel:
            raise ValueError("aligned joint qvel widths are incompatible")
        qpos_width += baseline_qpos
        qvel_width += baseline_qvel
    return qpos_width, qvel_width


def _actuator_trace_width(actuator: _AlignedActuator) -> int:
    """Validate one aligned actuator and return its canonical activation width."""
    if not isinstance(actuator, _AlignedActuator):
        raise TypeError("aligned_actuators must contain AlignedActuator values")
    width = actuator.activation_width
    if type(width) is not int or width < 0:
        raise ValueError("aligned actuator activation width must be nonnegative")
    baseline_address = actuator.baseline_activation_address
    candidate_address = actuator.candidate_activation_address
    if width == 0 and (baseline_address is not None or candidate_address is not None):
        raise ValueError("stateless aligned actuator has an activation address")
    if width > 0 and (baseline_address is None or candidate_address is None):
        raise ValueError("stateful aligned actuator lacks a role-local activation address")
    return width


def _slice_width(value: tuple[int, int], field: str) -> int:
    """Validate and return the positive width from an aligned state slice."""
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{field} must be a two-integer slice")
    start, width = value
    if type(start) is not int or start < 0 or type(width) is not int or width <= 0:
        raise ValueError(f"{field} must contain a nonnegative start and positive width")
    return width


def _comparison_reason(code: _ReasonCode, evidence: dict[str, _CanonicalValue]) -> _ReasonRecord:
    """Build one comparison-scoped preexecution budget reason."""
    frozen = _freeze_canonical(evidence)
    return _ReasonRecord(
        code=code,
        role="comparison",
        object_type=None,
        object_name=None,
        metric=None,
        boundary_index=None,
        evidence=_cast(_FrozenCanonicalObject, frozen),
    )


__all__ = [
    "MAX_TOTAL_INTERNAL_STEPS",
    "MAX_TRACE_FLOAT64_BYTES",
    "evaluate_preexecution_budgets",
]
